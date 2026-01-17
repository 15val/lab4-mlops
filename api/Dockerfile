# Використовуємо офіційний легкий образ Python 3.10 як базу для контейнера
FROM python:3.10-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Встановлюємо всі необхідні бібліотеки для роботи FastAPI, PyTorch, Transformers, pandas, joblib, scikit-learn та інших залежностей
RUN pip install --no-cache-dir fastapi uvicorn torch transformers==4.35.2 joblib scikit-learn safetensors pandas python-multipart

# Копіюємо файл з кодом FastAPI-сервісу (інференсу) у робочу директорію контейнера
COPY api/app.py .

# Вказуємо команду для запуску FastAPI-сервера через Uvicorn на порту 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]