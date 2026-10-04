from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: np.ndarray,
    mode: str = "center_scale",
    eps: float = 1e-6,
    mask: np.ndarray | None = None,
    reference_index: int | None = None,
) -> np.ndarray:
    """Normalize coordinates per frame using visible points and geometric scale."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if mode == "none":
        return arr
    if mode != "center_scale":
        raise ValueError(f"Unsupported normalization mode: {mode}")

    if arr.ndim == 3:
        coordinates = arr.copy()
        valid = np.ones(coordinates.shape[:2], dtype=bool)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.shape != coordinates.shape:
                raise ValueError("mask must have the same shape as landmarks")
            valid = mask_arr.any(axis=-1)
        for frame_index in range(coordinates.shape[0]):
            visible = valid[frame_index]
            if not visible.any():
                coordinates[frame_index] = 0.0
                continue
            if reference_index is not None and visible[reference_index]:
                center = coordinates[frame_index, reference_index]
            else:
                center = coordinates[frame_index, visible].mean(axis=0)
            centered = coordinates[frame_index] - center
            scale = np.sqrt(np.mean(np.square(centered[visible])))
            if scale >= eps:
                coordinates[frame_index] = centered / scale
            else:
                coordinates[frame_index] = centered
            coordinates[frame_index, ~visible] = 0.0
        return coordinates

    centered = arr - arr.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    return centered / scale
