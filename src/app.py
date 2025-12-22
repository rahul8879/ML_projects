from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_PATH = Path("/Users/rahultiwari/Documents/02_Freelancing/ML_case_studies/notebook/models/credit_risk_model.pkl")


class PredictionRequest(BaseModel):
    category: str = Field(..., examples=["food_dining"])
    amt: float = Field(..., examples=[150.75])
    gender: str = Field(..., examples=["male"])
    state: str = Field(..., examples=["CA"])
    city_pop: float = Field(..., examples=[100000])
    job: str = Field(..., examples=["engineer"])
    distance: float = Field(..., examples=[5.0])
    trans_hour: int = Field(..., examples=[12])
    trans_minute: int = Field(..., examples=[30])
    trans_second: int = Field(..., examples=[45])


app = FastAPI(title="Fraud Prediction API", version="1.0.0")
model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    global model
    model_path = MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    data = pd.DataFrame([payload.model_dump()])
    proba = model.predict_proba(data)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "fraud_probability": float(proba[0]),
        "prediction": int(pred[0]),
    }


@app.post("/predict-bulk")
def predict_bulk(payloads: list[PredictionRequest]):
    data = pd.DataFrame([payload.model_dump() for payload in payloads])
    proba = model.predict_proba(data)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return [
        {"fraud_probability": float(score), "prediction": int(pred)}
        for score, pred in zip(proba, preds, strict=False)
    ]
