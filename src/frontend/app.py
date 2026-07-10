import streamlit as st
import os
import time
import torch
import re
import tempfile

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

st.set_page_config(page_title="OmniRAG", page_icon="🧠", layout="wide")

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
st.title("🧠 OmniRAG")
st.caption("Задайте вопрос, и получите ответ, основанный на реальных данных!")

if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None

col_chat, col_viewer = st.columns([6, 4])

# --- Сайдбар (Настройки и Загрузка) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2836/2836856.png", width=60)
    st.title("Управление")
    
    st.header("⚙️ Настройки AI")
    temperature = st.slider("Креативность", 0.0, 1.0, 0.0, 0.1, help="0.0 - строгие факты из базы. 1.0 - больше свободы для нейросети.")
    top_k = st.slider("Источники (Top K)", 1, 30, 10, 1, help="Сколько фрагментов текста находить в базе.")
    
    st.divider()
    st.header("📄 База знаний (Глобальная)")
    uploaded_files = st.file_uploader("Загрузить документы", accept_multiple_files=True, type=["pdf", "docx", "txt", "xml", "html"], key="global_uploader")
    if uploaded_files:
        if st.button("Индексировать в базу", type="primary"):
            os.makedirs(DATA_DIR, exist_ok=True)
            for uf in uploaded_files:
                with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            with st.spinner("Идет индексация документов в Qdrant... Это может занять несколько минут."):
                run_pipeline()
            st.success("База успешно обновлена!")
            time.sleep(2)
            st.rerun()

with st.spinner("Подключение к базе данных..."):
    main_retriever, status_msg = load_db()

if not main_retriever:
    st.error(status_msg)
    st.stop()

with col_chat:
    # 3.2 Временные файлы для чата
    with st.expander("📎 Прикрепить временный файл (Поиск через @filename)"):
        temp_files = st.file_uploader("Файлы только для текущего сеанса", accept_multiple_files=True, key="temp_uploader")
        if temp_files:
            process_temp_files(temp_files)
            st.success(f"Загружено файлов: {len(st.session_state.temp_files_names)}. Используйте @имя_файла в запросе.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.markdown("**🔍 Источники:**")
                cols = st.columns(4)
                for cite_idx, citation in enumerate(msg["citations"]):
                    with cols[cite_idx % 4]:
                        if st.button(f"📄 {citation['source']}", key=f"btn_{msg_idx}_{cite_idx}", use_container_width=True):
                            st.session_state.selected_citation = citation

    if prompt := st.chat_input("Например: Какие побочные эффекты у тренболона? (или @doc.pdf сделай саммари)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            citations = []
            seen_sources = set()
            
            with st.spinner("Анализ базы данных..."):
                try:
                    # Проверяем наличие @filename в запросе
                    tag_match = re.search(r'@([\w.-]+)', prompt)
                    
                    if tag_match and "temp_vectorstore" in st.session_state:
                        # Поиск ТОЛЬКО по временному файлу
                        target_file = tag_match.group(1)
                        def filter_by_source(doc):
                            return doc.metadata.get("source") == target_file
                            
                        # InMemoryVectorStore filter requires callable or exact matching depending on version.
                        # Simple way: get retriever with filtering logic
                        temp_retriever = st.session_state.temp_vectorstore.as_retriever(
                            search_kwargs={'k': top_k, 'filter': filter_by_source}
                        )
                        active_retriever = temp_retriever
                    else:
                        active_retriever = main_retriever

                    # Собираем chain на лету с активным ретривером
                    rag_chain, _ = build_rag_chain(active_retriever, top_k, temperature)

                    chat_history = []
                    for msg in st.session_state.messages[:-1]: 
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
                            
                        if "context" in chunk:
                            for doc in chunk["context"]:
                                source = doc.metadata.get("source", "Неизвестный источник")
                                content_hash = hash(doc.page_content)
                                if content_hash not in seen_sources:
                                    citations.append({
                                        "source": source,
                                        "content": doc.page_content
                                    })
                                    seen_sources.add(content_hash)
                                
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "citations": citations
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Произошла ошибка при генерации ответа: {e}")

with col_viewer:
    st.subheader("📖 Просмотр источника")
    if st.session_state.selected_citation:
        st.markdown(f"**Источник:** `{st.session_state.selected_citation['source']}`")
        st.info(st.session_state.selected_citation['content'])
    else:
        st.info("👈 Нажмите на любой источник в чате, чтобы прочитать точную цитату, из которой нейросеть взяла информацию.")
