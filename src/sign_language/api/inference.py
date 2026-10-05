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
<<<<<<< HEAD
    """Predictor that runs the trained PyTorch model for real-time inference."""

    def __init__(self, model_path: str | Path | None = None, config_path: str | Path | None = None) -> None:
        import torch
        from sign_language.config import load_experiment_config
        from sign_language.models import build_model
        
        # Load a default configuration if none provided
        self.config = load_experiment_config(config_path or "configs/base.yaml")
        self.model = build_model(self.config)
        
        if model_path:
            import os
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
            else:
                import logging
                logging.warning(f"Model path {model_path} not found. Using untrained model.")
                
        self.model.eval()
        self.num_classes = self.config.model.num_classes

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        import torch
=======
    """Minimal runtime predictor for real-time inference scaffolding."""

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = int(num_classes)

    def predict(self, sequence: np.ndarray) -> np.ndarray:
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Prediction sequence must be a 2D array of shape [time_steps, features].")

<<<<<<< HEAD
        with torch.no_grad():
            # Add batch dimension
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)
            return logits.squeeze(0).numpy()
=======
        logits = np.linspace(0.0, 1.0, self.num_classes, dtype=np.float32)
        return logits
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
