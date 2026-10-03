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
