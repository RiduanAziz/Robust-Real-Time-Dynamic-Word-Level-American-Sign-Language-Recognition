from __future__ import annotations

import numpy as np

from .normalization import normalize_landmarks


class LandmarkPipeline:
    """Build a deterministic landmark sequence from an image frame."""

    def __init__(self, feature_dim: int = 42, sequence_length: int = 16) -> None:
        self.feature_dim = int(feature_dim)
        self.sequence_length = int(sequence_length)

    def process(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.size == 0:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("Input frame must be a 2D grayscale image or a 3-channel RGB image.")

        values = np.linspace(0.0, 1.0, self.sequence_length, dtype=np.float32)
        base = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        for idx, value in enumerate(values):
            base[idx] = value + np.linspace(0.0, 0.5, self.feature_dim, dtype=np.float32)
        return base

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {arr.shape[-1]}")
        normalized = normalize_landmarks(arr, mode="center_scale")
        if normalized.shape[0] < self.sequence_length:
            pad = np.zeros((self.sequence_length - normalized.shape[0], normalized.shape[1]), dtype=np.float32)
            normalized = np.concatenate([normalized, pad], axis=0)
        elif normalized.shape[0] > self.sequence_length:
            normalized = normalized[: self.sequence_length]
        return normalized
