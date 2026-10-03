from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: np.ndarray,
    mode: str = "center_scale",
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize landmark coordinates with a configurable geometric strategy."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if mode == "center_scale":
        centered = arr - arr.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        scale = np.where(scale < eps, 1.0, scale)
        return centered / scale

    if mode == "none":
        return arr

    raise ValueError(f"Unsupported normalization mode: {mode}")
