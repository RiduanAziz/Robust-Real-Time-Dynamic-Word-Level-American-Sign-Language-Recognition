from __future__ import annotations

import numpy as np


def create_prediction_payload(sequence: np.ndarray) -> dict[str, int | float | str]:
    """Prepare a prediction payload with fixed feature sizing metadata."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Prediction sequence must be a 2D array of shape [time_steps, features].")
    return {
        "sequence_length": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]),
        "dtype": str(arr.dtype),
    }


class RealTimePredictor:
    """Minimal runtime predictor for real-time inference scaffolding."""

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = int(num_classes)

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Prediction sequence must be a 2D array of shape [time_steps, features].")

        logits = np.linspace(0.0, 1.0, self.num_classes, dtype=np.float32)
        return logits
