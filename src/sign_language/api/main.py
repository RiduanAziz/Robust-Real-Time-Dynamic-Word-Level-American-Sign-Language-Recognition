from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sign_language.api.inference import RealTimePredictor, create_prediction_payload

logger = logging.getLogger(__name__)

# Global singleton predictor instance
_predictor: RealTimePredictor | None = None
_init_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _init_error
    model_path = os.environ.get("MODEL_PATH")
    config_path = os.environ.get("CONFIG_PATH", "configs/base.yaml")
    
    # Check default model paths if not set in environment
    if not model_path:
        candidates = [
            "models/temporal_transformer_best.pt",
            "models/temporal_transformer_trained.pt",
            "models/robust_holistic_fusion_best.pt",
            "models/temporal_transformer.pt",
        ]
        for candidate in candidates:
            if Path(candidate).is_file():
                model_path = candidate
                break

    try:
        if model_path and Path(model_path).is_file():
            _predictor = RealTimePredictor(model_path=model_path, config_path=config_path)
            logger.info("RealTimePredictor successfully initialized with model %s", model_path)
        else:
            _predictor = RealTimePredictor(config_path=config_path)
            logger.info("RealTimePredictor initialized from config %s (no checkpoint specified)", config_path)
        _init_error = None
    except Exception as exc:
        _init_error = str(exc)
        logger.warning("Failed to initialize predictor at startup: %s", exc)

    yield


app = FastAPI(title="Sign Language Recognition API", lifespan=lifespan)


def get_predictor() -> RealTimePredictor:
    global _predictor, _init_error
    if _predictor is None:
        # Lazy fallback attempt if lifespan didn't run (e.g. TestClient without lifespan)
        try:
            _predictor = RealTimePredictor(config_path="configs/base.yaml")
            _init_error = None
        except Exception as exc:
            _init_error = str(exc)
            raise HTTPException(status_code=503, detail=f"Predictor service unavailable: {_init_error}")
    return _predictor


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="2D sequence of landmark features [T, F]")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")


@app.get("/health")
def health() -> dict[str, str]:
    if _init_error is not None:
        return {"status": "degraded", "error": _init_error}
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "name": predictor.config.model.name,
        "status": "ready",
        "num_classes": predictor.num_classes,
        "input_dim": predictor.input_dim,
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "class_names_sample": predictor.class_names[:5] if predictor.class_names else [],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.sequence:
        raise HTTPException(status_code=422, detail="Sequence must not be empty.")

    try:
        sequence = np.asarray(request.sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric sequence data: {exc}")

    if sequence.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array [time_steps, features], got {sequence.shape}",
        )

    if not np.all(np.isfinite(sequence)):
        raise HTTPException(status_code=422, detail="Input sequence contains non-finite values (NaN or Inf).")

    predictor = get_predictor()

    try:
        prediction_result = predictor.predict_label(
            sequence, confidence_threshold=request.confidence_threshold
        )
        logits = predictor.predict_logits(sequence)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = create_prediction_payload(sequence)

    return {
        "sequence_length": payload["sequence_length"],
        "feature_dim": payload["feature_dim"],
        "logits": [float(v) for v in logits],
        "predicted_class": prediction_result["predicted_class"],
        "predicted_label": prediction_result["predicted_label"],
        "confidence": prediction_result["confidence"],
        "meets_threshold": prediction_result["meets_threshold"],
    }
