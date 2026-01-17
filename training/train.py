import mlflow
import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib
import os
import glob

# --- MinIO settings ---
S3_ENDPOINT_URL = "http://minio:9000"
S3_BUCKET = "mlops"
S3_OBJECT = "dataset_train_latest.csv"
S3_KEY = "minioadmin"
S3_SECRET = "minioadmin"

# --- Read dataset from MinIO ---
s3_path = f"s3://{S3_BUCKET}/{S3_OBJECT}"

df = pd.read_csv(
    s3_path,
    storage_options={
        "key": S3_KEY,
        "secret": S3_SECRET,
        "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}
    }
)

mlflow.set_experiment("bert-tier-classification")

le = LabelEncoder()
df["label"] = le.fit_transform(df["tier"])
dataset = Dataset.from_pandas(df)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=128
    )


dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3
)

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    logging_steps=5
)

with mlflow.start_run() as run:
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    trainer.train()

    # --- Save model, tokenizer, label encoder locally ---
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    files = glob.glob(os.path.join(model_dir, '*'))
    for f in files:
        os.remove(f)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    joblib.dump(le, f"{model_dir}/label_encoder.pkl")
    mlflow.log_param("model", "bert-base-multilingual-cased")

    # --- Log model artifacts to MLflow/MinIO ---
    mlflow.log_artifacts(model_dir, artifact_path="model")

print("Тренування завершено!")
