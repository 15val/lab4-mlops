from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import shutil
import os
import boto3

SOURCE_PATH = '/opt/airflow/input/dataset_train.csv'
TARGET_DIR = '/opt/airflow/data'

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "mlops"


def upload_to_minio(local_path, object_name):
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    # Створити bucket, якщо не існує
    buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
    if MINIO_BUCKET not in buckets:
        s3.create_bucket(Bucket=MINIO_BUCKET)
    s3.upload_file(local_path, MINIO_BUCKET, object_name)


def update_data():
    os.makedirs(TARGET_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = f"{TARGET_DIR}/dataset_train_{timestamp}.csv"
    shutil.copy(SOURCE_PATH, target_path)
    shutil.copy(target_path, f"{TARGET_DIR}/dataset_train_latest.csv")
    # Завантажити у MinIO
    upload_to_minio(target_path, f"dataset_train_{timestamp}.csv")
    upload_to_minio(f"{TARGET_DIR}/dataset_train_latest.csv", "dataset_train_latest.csv")


with DAG(
    dag_id='update_training_data',
    start_date=datetime(2024, 1, 1),
    schedule_interval='*/5 * * * *',
    catchup=False
) as dag:
    update_task = PythonOperator(
        task_id='update_training_data_task',
        python_callable=update_data
    )
