# 1. Базовый образ
FROM python:3.12-slim

# 2. Настройки окружения Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Рабочая директория внутри контейнера
WORKDIR /app

# 4. Копирование списка зависимостей
COPY requirements.txt .

# 5. Установка библиотек
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копирование кода сервиса
COPY app.py DataManager.py ./

# 7. Документирование порта
EXPOSE 5001

# 8. Команда запуска
CMD ["python", "app.py"]