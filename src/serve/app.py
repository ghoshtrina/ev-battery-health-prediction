from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define the input schema
class BatteryFeatures(BaseModel):
    cycle_count: int
    avg_temperature: float
    charge_rate: float
    discharge_rate: float
    depth_of_discharge: float
    internal_resistance: float

# Try to load the trained model
model_path = "artifacts/random_forest.joblib"
feats_path = "artifacts/feature_names.txt"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None  # placeholder if not trained yet

if os.path.exists(feats_path):
    feature_names = [ln.strip() for ln in open(feats_path).read().splitlines()]
else:
    feature_names = None


app = FastAPI(title="EV Battery Health Prediction API")

@app.get("/")
def root():
    return {"message": "API is running. Go to /docs for interactive docs."}

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "feature_names_loaded": feature_names is not None,
        "expected_features": feature_names
    }

@app.post("/predict")
def predict(features: BatteryFeatures):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model or feature names not found. Please train the model first by running: python -m src.pipelines.train_pipeline",
        )
    
    payload = features.dict()
    try:
        X = np.array([[payload[name] for name in feature_names]], dtype=float)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missuing feature: {e}")

    # Predict SOH
    soh = float(model.predict(X)[0])

    # Bucket logic
    if soh >= 85:
        bucket = "Healthy"
    elif soh >= 70:
        bucket = "Moderate"
    else:
        bucket = "End-of-Life"

    return {
        "soh": float(soh),
        "bucket": bucket,
        "input": payload
    }
