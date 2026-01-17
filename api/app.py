from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = "model"
    app.state.model = BertForSequenceClassification.from_pretrained(model_dir)
    app.state.tokenizer = BertTokenizer.from_pretrained(model_dir)
    app.state.le = joblib.load(f"{model_dir}/label_encoder.pkl")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    le = request.app.state.le
    df = pd.read_csv(file.file, header=None, names=["country", "text"])
    texts = df["text"].tolist()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).tolist()
    tiers = le.inverse_transform(preds)
    results = [
        {"country": country, "text": text, "tier": tier}
        for country, text, tier in zip(df["country"], df["text"], tiers)
    ]
    return {"results": results}