from __future__ import annotations

from fastapi.testclient import TestClient

from sign_language.api.main import app


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    model_response = client.get("/model/info")
    assert model_response.status_code == 200


def test_api_predict_endpoint() -> None:
    client = TestClient(app)
    sequence = [[float(i) for i in range(42)] for _ in range(16)]
    response = client.post("/predict", json={"sequence": sequence})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["logits"]) == 5
    assert 0 <= payload["predicted_class"] < 5
