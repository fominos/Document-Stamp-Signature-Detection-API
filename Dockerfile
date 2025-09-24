# Базовый образ
FROM python:3.11-slim

# Установка системных зависимостей (критично для OpenCV)
# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    poppler-utils \
    poppler-data \
    libpoppler-cpp-dev \
    libpoppler-glib-dev && \
    rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости first для кэширования
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Проверяем что файлы скопировались
RUN echo "Содержимое рабочей директории:" && \
    ls -la && \
    echo "Проверка наличия основных файлов:" && \
    ls -la main.py *.pt || echo "Файлы не найдены!"

# Открываем порт 80
EXPOSE 8000

# Запускаем приложение на порту 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]