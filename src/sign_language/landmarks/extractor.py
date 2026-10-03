from __future__ import annotations

import numpy as np


class LandmarkExtractor:
    """Minimal landmark extraction abstraction for MediaPipe-style pipelines."""

    def __init__(self, feature_dim: int = 42) -> None:
        self.feature_dim = feature_dim

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Return a placeholder landmark vector for a frame."""
        if frame.ndim == 2:
            height, width = frame.shape
            return np.zeros((self.feature_dim,), dtype=np.float32)
        height, width, _ = frame.shape
        return np.zeros((self.feature_dim,), dtype=np.float32)
