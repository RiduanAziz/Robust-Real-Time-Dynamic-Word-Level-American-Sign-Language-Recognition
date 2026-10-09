from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sign_language.config import load_experiment_config
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
from sign_language.utils.device import get_device

logger = logging.getLogger(__name__)


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
    """Predictor that executes a validated PyTorch checkpoint for inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = get_device(device)
        self.config = load_experiment_config(config_path or "configs/base.yaml")

        self.class_names: list[str] = list(self.config.dataset.labels)
        self.sequence_length: int = self.config.dataset.sequence_length
        self.input_dim: int = self.config.model.input_dim

        # Load checkpoint strictly if provided
        if model_path is not None:
            ckpt_path = Path(model_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Required checkpoint file does not exist: {ckpt_path}. "
                    "Refusing to continue with untrained weights."
                )

            ckpt = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                if "class_names" in ckpt and ckpt["class_names"]:
                    self.class_names = list(ckpt["class_names"])
                if "input_dim" in ckpt:
                    self.input_dim = ckpt["input_dim"]
                if "sequence_length" in ckpt:
                    self.sequence_length = ckpt["sequence_length"]
                if "model_name" in ckpt:
                    self.config.model.name = ckpt["model_name"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise ValueError(f"Unrecognized checkpoint format in {ckpt_path}")

            self.config.model.input_dim = self.input_dim
            self.config.model.num_classes = len(self.class_names)
            self.model = build_model(self.config)
            self.model.load_state_dict(state_dict)
        else:
            # If no model_path is provided, build model from config
            self.config.model.num_classes = len(self.class_names)
            self.model = build_model(self.config)

        self.checkpoint_path = str(ckpt_path) if model_path is not None else None
        self.model.to(self.device)
        self.model.eval()
        self.num_classes = len(self.class_names)

        # Preprocessing pipeline
        raw_dim = self.input_dim // 3 if self.input_dim >= 4977 else self.input_dim
        self.pipeline = LandmarkPipeline(feature_dim=raw_dim, sequence_length=self.sequence_length)

    def predict_logits(self, sequence: np.ndarray) -> np.ndarray:
        """Compute raw output logits for a sequence."""
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D sequence [T, D], got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input sequence contains non-finite values (NaN or Inf)")

        # Normalize and resample if needed
        if arr.shape[1] == self.pipeline.feature_dim:
            arr = self.pipeline.normalize_sequence(arr, include_dynamics=(self.input_dim >= 4977))

        if arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {arr.shape[1]}"
            )

        with torch.no_grad():
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)
            return logits.squeeze(0).cpu().numpy()

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Backward-compatible predict method returning raw output logits."""
        return self.predict_logits(sequence)

    def predict_label(self, sequence: np.ndarray, confidence_threshold: float = 0.0) -> dict[str, Any]:
        """Predict class label, confidence score, top-k candidates, and probabilities."""
        logits = self.predict_logits(sequence)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

        meets_threshold = confidence >= confidence_threshold

        top_indices = np.argsort(probs)[::-1][:min(5, len(probs))]
        top_k = [
            {
                "class_index": int(i),
                "label": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]

        return {
            "predicted_class": pred_idx,
            "predicted_label": pred_label if meets_threshold else "Unknown (Low Confidence)",
            "confidence": confidence,
            "meets_threshold": meets_threshold,
            "probabilities": probs.tolist(),
            "top_k": top_k,
        }
