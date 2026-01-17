from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import shutil
import os
import boto3
import requests

SOURCE_PATH = '/opt/airflow/input/dataset_train.csv'
TARGET_DIR = '/opt/airflow/data'

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "mlops"

def upload_to_minio(local_path, object_name):
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
    if MINIO_BUCKET not in buckets:
        s3.create_bucket(Bucket=MINIO_BUCKET)
    s3.upload_file(local_path, MINIO_BUCKET, object_name)

def update_data():
    os.makedirs(TARGET_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = f"{TARGET_DIR}/dataset_train_{timestamp}.csv"
    shutil.copy(SOURCE_PATH, target_path)
    latest_path = f"{TARGET_DIR}/dataset_train_latest.csv"
    if os.path.exists(latest_path):
        os.remove(latest_path)
    shutil.copy(target_path, latest_path)
    upload_to_minio(target_path, f"dataset_train_{timestamp}.csv")
    upload_to_minio(latest_path, "dataset_train_latest.csv")

def trigger_github_actions():
    url = "https://api.github.com/repos/15val/lab4-mlops/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token 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"
    }
    data = {
        "event_type": "minio_data_updated"
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.status_code, response.text)

with DAG(
    dag_id='update_training_data',
    start_date=datetime(2024, 1, 1),
    schedule_interval='0,30 * * * *',
    catchup=False
) as dag:
    update_task = PythonOperator(
        task_id='update_training_data_task',
        python_callable=update_data
    )

    trigger_pipeline_task = PythonOperator(
        task_id='trigger_github_pipeline',
        python_callable=trigger_github_actions
    )

    update_task >> trigger_pipeline_task
#ngrok http 9000