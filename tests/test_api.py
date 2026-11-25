from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_explain_valid_payload():
    payload = {
        "tenure": 1,
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/predict-explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_category"] in {"Low", "Medium", "High"}
    assert len(data["top_attributions"]) == 3


def test_predict_explain_invalid_payload():
    payload = {
        "tenure": -5,
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/predict-explain", json=payload)
    assert response.status_code == 422
