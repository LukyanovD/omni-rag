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
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Копируем исходный код проекта
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Настраиваем PYTHONPATH для импортов src.*
ENV PYTHONPATH=/app

# Запуск Streamlit по умолчанию
CMD ["streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0"]
