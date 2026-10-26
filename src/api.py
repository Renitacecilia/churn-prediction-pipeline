"""FastAPI service that serves churn predictions."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config


class Customer(BaseModel):
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "SeniorCitizen": 0, "tenure": 5, "MonthlyCharges": 85.5, "TotalCharges": 420.0,
                "gender": "Female", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str


def risk_level(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))
    app.state.model = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.train` first.")
    row = pd.DataFrame([customer.model_dump()])[config.FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(p, 4), will_churn=p >= 0.5, risk_level=risk_level(p))
