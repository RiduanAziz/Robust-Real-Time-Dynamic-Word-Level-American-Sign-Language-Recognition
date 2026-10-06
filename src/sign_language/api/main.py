from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from sign_language.api.inference import RealTimePredictor, create_prediction_payload

app = FastAPI(title="Sign Language Recognition API")


class PredictionRequest(BaseModel):
    sequence: list[list[float]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, str]:
    return {"name": "sign-language-recognition", "status": "ready"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, int | float | list[float]]:
    sequence = np.asarray(request.sequence, dtype=np.float32)
    payload = create_prediction_payload(sequence)
    predictor = RealTimePredictor()
    logits = predictor.predict(sequence)
    predicted_class = int(np.argmax(logits))
    return {
        "sequence_length": payload["sequence_length"],
        "feature_dim": payload["feature_dim"],
        "logits": [float(value) for value in logits],
        "predicted_class": predicted_class,
    }
