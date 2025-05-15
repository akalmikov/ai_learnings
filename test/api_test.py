# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app
from app.model import predict_with_confidence

client = TestClient(app)

def test_api_response():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert response.json()["species"] in ["setosa", "versicolor", "virginica"]

def test_confidence_output():
    result = predict_with_confidence([5.1, 3.5, 1.4, 0.2])
    assert 0 <= result["confidence"] <= 1

def test_invalid_input():
    response = client.post("/predict", json={"features": [1, 2]})  # too few features
    assert response.status_code == 422
