import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env файла, если он существует
load_dotenv()

# Базовые директории (отсчет от корня проекта, так как скрипты запускаются оттуда)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
DB_DIR = BASE_DIR / os.getenv("DB_DIR", "db")
STATE_FILE = DB_DIR / "state.json"
DLQ_FILE = DB_DIR / "dlq.json"

# Настройки Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "omni_rag_collection")

# Настройки парсинга и векторизации
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
