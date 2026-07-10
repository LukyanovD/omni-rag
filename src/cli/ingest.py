import os
import json
import uuid
import hashlib
from pathlib import Path
import concurrent.futures

from src.indexing.router import DocumentRouter
from src.indexing.indexer import VectorIndexer
from src.core.config import DATA_DIR, DB_DIR, STATE_FILE, DLQ_FILE, EMBEDDING_MODEL
from src.core.logger import setup_logger
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = setup_logger(__name__)



def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Не удалось вычислить хэш для {filepath}: {e}")
        return ""

def load_json(filepath: Path, default: dict) -> dict:
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def garbage_collection(state: dict) -> dict:
    """
    Удаляет из базы Qdrant и doc_store документы, которых больше нет на диске в папке data/.
    Возвращает обновленный state.
    """
    logger.info("=== Этап 0: Сборка мусора (Garbage Collection) ===")
    deleted_keys = []
    ids_to_delete = []

    for filepath_str, info in state.items():
        if not os.path.exists(os.path.join(DATA_DIR, filepath_str)):
            logger.info(f"Файл {filepath_str} удален с диска. Подготовка к удалению из Qdrant.")
            doc_id = info.get("doc_id")
            if doc_id:
                ids_to_delete.append(doc_id)
            deleted_keys.append(filepath_str)

    if ids_to_delete:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        from src.core.config import QDRANT_URL, QDRANT_COLLECTION, DB_DIR
        try:
            # 1. Удаление чанков из Qdrant по метаданным
            client = QdrantClient(url=QDRANT_URL)
            if client.collection_exists(QDRANT_COLLECTION):
                client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.doc_id",
                                match=models.MatchAny(any=ids_to_delete)
                            )
                        ]
                    )
                )
            
            # 2. Удаление Parent-документов из LocalFileStore
            from langchain.storage import LocalFileStore
            from langchain.storage._lc_store import create_kv_docstore
            store = create_kv_docstore(LocalFileStore(os.path.join(DB_DIR, "doc_store")))
            store.mdelete(ids_to_delete)
            
            logger.info(f"Успешно удалено {len(ids_to_delete)} устаревших документов из Qdrant и doc_store.")
        except Exception as e:
            logger.error(f"Ошибка при удалении документов: {e}")
            return state # В случае ошибки отменяем удаление из state

    for key in deleted_keys:
        del state[key]
        
    if not deleted_keys:
        logger.info("Удаленных файлов не найдено.")
        
    return state

def process_file_task(filepath_str: str) -> dict:
    router = DocumentRouter()
    return router.process(filepath_str)

def run_pipeline():
    DATA_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)

    state = load_json(STATE_FILE, {})
    dlq = load_json(DLQ_FILE, {})
    
    # 0. Сборка мусора
    state = garbage_collection(state)
    
    router = DocumentRouter()
    docs_to_index = []
    
    # Мы больше не пересоздаем state с нуля, мы его обновляем,
    # чтобы не потерять файлы, которые не изменились.
    updated_state = state.copy()

    logger.info("=== Этап 1: Обход директории data/ и парсинг ===")
    # Собираем список файлов для обработки
    files_to_process = []
    for filepath in DATA_DIR.rglob("*"):
        if not filepath.is_file():
            continue
            
        file_hash = get_file_hash(filepath)
        if not file_hash:
            continue
            
        filename = str(filepath.relative_to(DATA_DIR))
        
        # Проверка идемпотентности
        if filename in state and state[filename].get("hash") == file_hash:
            logger.info(f"Пропуск файла (не изменился): {filename}")
            continue
            
        logger.info(f"Обнаружен новый/измененный файл: {filename}")
        files_to_process.append((filepath, filename, file_hash))

    if files_to_process:
        logger.info(f"Запуск параллельного парсинга для {len(files_to_process)} файлов...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Маппинг futures к данным файлов
            future_to_file = {
                executor.submit(process_file_task, str(filepath)): (filename, file_hash)
                for filepath, filename, file_hash in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                filename, file_hash = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        doc_id = str(uuid.uuid4())
                        result["id"] = doc_id
                        docs_to_index.append((filename, file_hash, doc_id, result))
                    else:
                        logger.warning(f"Файл {filename} отброшен в DLQ.")
                        dlq[filename] = {"reason": "router_returned_none", "hash": file_hash}
                except Exception as exc:
                    logger.error(f"Файл {filename} вызвал ошибку: {exc}")
                    dlq[filename] = {"reason": f"exception: {exc}", "hash": file_hash}
            
    if not docs_to_index:
        logger.info("✅ Нет новых файлов для векторизации. Пайплайн завершен.")
        save_json(STATE_FILE, updated_state)
        save_json(DLQ_FILE, dlq)
        return

    logger.info(f"=== Этап 2: Векторизация ({len(docs_to_index)} новых документов) ===")
    try:
        indexer = VectorIndexer()
        raw_docs = [item[3] for item in docs_to_index]
        
        doc_chunk_counts = indexer.build_and_save_index(raw_docs)
        
        if doc_chunk_counts is not None:
            for filename, file_hash, doc_id, _ in docs_to_index:
                chunk_count = doc_chunk_counts.get(doc_id, 0)
                updated_state[filename] = {
                    "hash": file_hash,
                    "doc_id": doc_id,
                    "chunk_count": chunk_count
                }
            save_json(STATE_FILE, updated_state)
            logger.info("✅ Пайплайн успешно завершен!")
        else:
            logger.error("❌ Векторизация завершилась неудачно. State не обновлен.")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка пайплайна: {e}")
        
    finally:
        save_json(DLQ_FILE, dlq)

if __name__ == "__main__":
    run_pipeline()
