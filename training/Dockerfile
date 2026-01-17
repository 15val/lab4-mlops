# Використовуємо офіційний образ Python 3.10 як базу для контейнера
FROM python:3.10-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Встановлюємо всі необхідні бібліотеки для тренування моделі:
# - torch, transformers: для роботи з нейромережами та BERT
# - datasets: для обробки датасетів
# - scikit-learn, pandas: для підготовки даних та роботи з таблицями
# - mlflow: для логування експериментів
# - joblib: для збереження енкодера міток
# - accelerate: для оптимізації тренування
RUN pip install --no-cache-dir torch transformers datasets scikit-learn pandas mlflow joblib accelerate minio s3fs boto3
# Копіюємо скрипт для тренування моделі у робочу директорію контейнера
COPY training/train.py .

# Вказуємо команду для запуску тренувального скрипта
CMD ["python", "train.py"]