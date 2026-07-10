from src.core.logger import setup_logger
from src.core.config import QDRANT_URL, QDRANT_COLLECTION, DB_DIR
import os
import re
from typing import List, Dict, Any, Optional
import torch

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from qdrant_client import QdrantClient
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_classic.storage import LocalFileStore
    from langchain_classic.storage._lc_store import create_kv_docstore
except ImportError:
    pass

logger = setup_logger(__name__)

class VectorIndexer:
    """
    Класс для токен-чанкинга нормализованных документов и их векторизации 
    с сохранением в локальную базу данных Qdrant (Hybrid Search)
    и использованием ParentDocumentRetriever.
    """
    def __init__(self):
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logger.info(f"Инициализация HuggingFaceEmbeddings ({self.model_name})...")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Инициализация FastEmbedSparse для гибридного поиска Qdrant...")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        self.fs = LocalFileStore(os.path.join(DB_DIR, "doc_store"))
        self.store = create_kv_docstore(self.fs)
        
        # Parent/Child Splitters (Small-to-Big Retrieval)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    def _protect_tables(self, text: str) -> str:
        """Предварительная обработка Markdown таблиц."""
        table_pattern = re.compile(r'((?:\|.*\|\n?)+)')
        def chunk_table(match):
            lines = match.group(1).strip().split('\n')
            if len(lines) <= 5: 
                return match.group(1)
            header = lines[0]
            separator = lines[1]
            if not set(separator.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')) == set():
                return match.group(1)
            data_rows = lines[2:]
            chunked_tables = []
            chunk_size = 5
            for i in range(0, len(data_rows), chunk_size):
                chunk = [header, separator] + data_rows[i:i+chunk_size]
                chunked_tables.append('\n'.join(chunk))
            return '\n\n'.join(chunked_tables) + '\n'
        return table_pattern.sub(chunk_table, text)

    def build_and_save_index(self, docs: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        if not docs:
            logger.warning("Пустой список документов.")
            return None

        langchain_docs = []
        for d in docs:
            text = self._protect_tables(d.get("text", ""))
            metadata = d.get("metadata", {})
            metadata.pop("id", None)
            metadata.pop("_id", None)
            doc_id = d.get("id", "unknown_id")
            metadata["doc_id"] = doc_id
            if text:
                langchain_docs.append(Document(page_content=text, metadata=metadata))

        if not langchain_docs:
            return None

        logger.info(f"Векторизация {len(langchain_docs)} документов через ParentDocumentRetriever...")
        
        try:
            if not self.qdrant_client.collection_exists(QDRANT_COLLECTION):
                logger.info("Коллекция Qdrant не найдена. Создаю новую...")
                QdrantVectorStore.from_texts(
                    ["test_init_dummy"],
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    url=QDRANT_URL,
                    collection_name=QDRANT_COLLECTION,
                )
                from qdrant_client.models import Filter
                self.qdrant_client.delete(collection_name=QDRANT_COLLECTION, points_selector=Filter())

            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=QDRANT_COLLECTION,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID
            )

            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=self.store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )

            doc_ids = [d.metadata["doc_id"] for d in langchain_docs]
            retriever.add_documents(langchain_docs)
            
            logger.info("Документы успешно добавлены в Qdrant и doc_store.")
            
            # Возвращаем dummy dict для совместимости с ingest.py
            return {d_id: 1 for d_id in doc_ids}
                
        except Exception as e:
            import traceback
            logger.error(f"Ошибка во время векторизации в Qdrant: {e}\n{traceback.format_exc()}")
            return False
