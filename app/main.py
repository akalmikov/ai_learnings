# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_species, predict_with_confidence

app = FastAPI()

class IrisFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: IrisFeatures):
    species = predict_species(data.features )
    return {"species": species}

@app.post("/predict")
def predict(data: IrisFeatures):
    result = predict_with_confidence(data.features)
    return result
