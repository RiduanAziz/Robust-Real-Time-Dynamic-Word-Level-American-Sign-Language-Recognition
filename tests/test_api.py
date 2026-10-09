from __future__ import annotations

import json
from fastapi.testclient import TestClient

from sign_language.api.main import app


def test_api_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["process_alive"] is True
        assert data["backend_initialized"] is True
        assert "mediapipe_available" in data
        assert "trained_model_loaded" in data


def test_api_model_info() -> None:
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "num_classes" in data
        assert "sequence_length" in data
        assert "input_dim" in data
        assert data["num_classes"] > 0


def test_api_predict_endpoint_valid() -> None:
    with TestClient(app) as client:
        sequence = [[0.0] * 4977 for _ in range(64)]
        response = client.post("/predict", json={"sequence": sequence, "confidence_threshold": 0.1})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["logits"]) == 100
        assert 0 <= payload["predicted_class"] < 100
        assert "top_k" in payload
        assert len(payload["top_k"]) <= 5


def test_api_predict_endpoint_invalid() -> None:
    with TestClient(app) as client:
        # Empty sequence
        response = client.post("/predict", json={"sequence": []})
        assert response.status_code == 422

        # Invalid shape (1D instead of 2D)
        response = client.post("/predict", json={"sequence": [1.0, 2.0, 3.0]})
        assert response.status_code == 422


def test_websocket_live_connection_and_handshake() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as websocket:
            # 1. Receive initial handshake
            data = websocket.receive_json()
            assert data["type"] == "session_started"
            assert "session_id" in data
            assert data["active_mode"] == "guided"
            assert "sequence_length" in data

            # 2. Send config update
            websocket.send_json({
                "type": "config",
                "mode": "guided",
                "confidence_threshold": 0.5,
                "debounce_frames": 4,
            })
            config_resp = websocket.receive_json()
            assert config_resp["type"] == "config_updated"
            assert config_resp["confidence_threshold"] == 0.5
            assert config_resp["debounce_frames"] == 4

            # 3. Send reset
            websocket.send_json({"type": "reset"})
            reset_resp = websocket.receive_json()
            assert reset_resp["type"] == "reset_ack"


def test_frontend_static_serving() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "SignFlow" in response.text


def test_api_vocabulary_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/vocabulary")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["count"] == 100
        assert "drink" in data["vocabulary"]
        assert "book" in data["vocabulary"]


def test_api_robustness_experiment() -> None:
    with TestClient(app) as client:
        sequence = [[0.0] * 4977 for _ in range(64)]
        response = client.post(
            "/api/robustness/experiment",
            json={
                "sequence": sequence,
                "perturbation_type": "coordinate_jitter",
                "severity": 0.2,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "original_prediction" in payload
        assert "perturbed_prediction" in payload
        assert "prediction_consistent" in payload
        assert "processing_time_ms" in payload
        assert payload["perturbation_type"] == "coordinate_jitter"
        assert "clean_quality" in payload


def test_websocket_replay_buffer_request() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as websocket:
            init = websocket.receive_json()
            assert init["type"] == "session_started"

            websocket.send_json({"type": "request_replay"})
            resp = websocket.receive_json()
            assert resp["type"] == "replay_buffer"
            assert isinstance(resp["frames"], list)


