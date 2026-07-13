FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей (требуются для сборки некоторых C-пакетов и FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ARG USE_GPU=false
RUN if [ "$USE_GPU" = "true" ] ; then \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt ; \
    fi

# Копируем исходный код проекта
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Настраиваем PYTHONPATH для импортов src.*
ENV PYTHONPATH=/app

# Запуск Streamlit по умолчанию
CMD ["streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0"]
