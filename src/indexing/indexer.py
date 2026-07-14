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
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from qdrant_client import QdrantClient
except ImportError:
    pass

logger = setup_logger(__name__)

class VectorIndexer:
    """
    Класс для токен-чанкинга нормализованных документов и их векторизации 
    с сохранением в локальную базу данных Qdrant (Hybrid Search).
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
        
        # Сплиттер для чанков
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)

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
            
            source_file = metadata.get("source", "Неизвестный документ")
            filename = os.path.basename(source_file)
            
            if text:
                doc = Document(page_content=text, metadata=metadata)
                chunks = self.splitter.split_documents([doc])
                
                for chunk in chunks:
                    chunk.page_content = f"[Источник: {filename}]\n{chunk.page_content}"
                    langchain_docs.append(chunk)

        if not langchain_docs:
            return None

        logger.info(f"Векторизация {len(langchain_docs)} чанков напрямую в Qdrant...")
        
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

            vector_store.add_documents(langchain_docs)
            
            logger.info("Чанки успешно добавлены в Qdrant.")
            
            doc_chunk_counts = {}
            for chunk in langchain_docs:
                d_id = chunk.metadata.get("doc_id")
                if d_id:
                    doc_chunk_counts[d_id] = doc_chunk_counts.get(d_id, 0) + 1
                    
            return doc_chunk_counts
                
        except Exception as e:
            import traceback
            logger.error(f"Ошибка во время векторизации в Qdrant: {e}\n{traceback.format_exc()}")
            return False
