import streamlit as st
import os
import time
import torch
import re
import tempfile

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

try:
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from qdrant_client import QdrantClient
except ImportError:
    QdrantVectorStore = None

from src.core.config import EMBEDDING_MODEL, OLLAMA_HOST, DATA_DIR, QDRANT_URL, QDRANT_COLLECTION, DB_DIR
from src.core.rag import build_rag_chain
from src.cli.ingest import run_pipeline
from src.indexing.router import DocumentRouter

# 3.1 Caching
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
os.makedirs(DB_DIR, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=os.path.join(DB_DIR, "langchain_cache.db")))

st.set_page_config(page_title="OmniRAG", layout="wide")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def load_db():
    """Загрузка векторной базы Qdrant с Hybrid Search (Dense + Sparse) и ParentDocumentRetriever"""
    if QdrantVectorStore is None:
        return None, "Библиотека langchain-qdrant не установлена."
        
    try:
        embeddings = get_embeddings()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(QDRANT_COLLECTION):
            return None, "База данных не найдена или коллекция пуста. Загрузите документы."
            
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_classic.storage import LocalFileStore
        from langchain_classic.storage._lc_store import create_kv_docstore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        store = create_kv_docstore(LocalFileStore(os.path.join(DB_DIR, "doc_store")))
        
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=250),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
        )
        
        return retriever, "Success"
    except Exception as e:
        return None, f"Критическая ошибка инициализации: {e}"

def process_temp_files(uploaded_files):
    """Обработка временных файлов для InMemory VectorStore"""
    if "temp_vectorstore" not in st.session_state:
        st.session_state.temp_vectorstore = InMemoryVectorStore(embedding=get_embeddings())
        st.session_state.temp_files_names = set()

    router = DocumentRouter()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for uf in uploaded_files:
        if uf.name not in st.session_state.temp_files_names:
            # Сохраняем во временный файл для парсинга
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.name)[1]) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            
            # Парсим
            parsed_data = router.process(tmp_path)
            os.remove(tmp_path)
            
            if parsed_data and "text" in parsed_data:
                doc = Document(
                    page_content=parsed_data["text"], 
                    metadata={"source": uf.name}
                )
                chunks = splitter.split_documents([doc])
                st.session_state.temp_vectorstore.add_documents(chunks)
                st.session_state.temp_files_names.add(uf.name)

# --- Инициализация UI ---
st.title("OmniRAG")
st.caption("Задайте вопрос, и получите ответ, основанный на реальных данных!")

top_k = 4  # Резко снижаем количество кусков, чтобы маленькая модель не теряла фокус
with st.spinner("Подключение к базе данных..."):
    main_retriever, status_msg = load_db()

if not main_retriever:
    st.error(status_msg)
    st.stop()

# 3.2 Временные файлы для чата
with st.expander("Прикрепить временный файл"):
    temp_files = st.file_uploader("Файлы только для текущего сеанса", accept_multiple_files=True, key="temp_uploader")
    if temp_files:
        process_temp_files(temp_files)
        st.success(f"Загружено файлов: {len(st.session_state.temp_files_names)}. Они будут автоматически учтены при ответе.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if prompt := st.chat_input("Например: Какие побочные эффекты у тренболона?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Анализ базы данных..."):
            try:
                # Устанавливаем search_kwargs для глобального ретривера
                if not hasattr(main_retriever, "search_kwargs") or main_retriever.search_kwargs is None:
                    main_retriever.search_kwargs = {}
                main_retriever.search_kwargs.update({'k': top_k * 3})

                if "temp_vectorstore" in st.session_state and st.session_state.temp_files_names:
                    from langchain_classic.retrievers import MergerRetriever
                    temp_retriever = st.session_state.temp_vectorstore.as_retriever(
                        search_kwargs={'k': top_k * 3}
                    )
                    active_retriever = MergerRetriever(retrievers=[main_retriever, temp_retriever])
                else:
                    active_retriever = main_retriever

                # Собираем chain на лету с активным ретривером
                rag_chain, _ = build_rag_chain(active_retriever, top_k)

                chat_history = []
                # Ограничиваем историю только последними 2 сообщениями (1 пара вопрос-ответ)
                for msg in st.session_state.messages[-3:-1]: 
                    if msg["role"] == "user":
                        chat_history.append(("human", msg["content"]))
                    else:
                        chat_history.append(("ai", msg["content"]))
                        
                last_update_time = time.time()
                UPDATE_INTERVAL = 0.1
                
                for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        if time.time() - last_update_time > UPDATE_INTERVAL:
                            message_placeholder.markdown(full_response + "▌")
                            last_update_time = time.time()
                            
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
                st.rerun()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Произошла ошибка при генерации ответа: {e}")

