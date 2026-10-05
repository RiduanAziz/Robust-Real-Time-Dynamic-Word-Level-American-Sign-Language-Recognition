from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LandmarkObservation:
    """Flattened landmark values and a same-shaped mask for one frame."""

    landmarks: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.landmarks.shape != self.mask.shape:
            raise ValueError("landmarks and mask must have the same shape")
        if self.landmarks.ndim != 1:
            raise ValueError("landmarks and mask must be one-dimensional")


@dataclass(frozen=True)
class LandmarkSequence:
    landmarks: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.landmarks.shape != self.mask.shape:
            raise ValueError("landmarks and mask must have the same shape")
        if self.landmarks.ndim != 2:
            raise ValueError("landmarks and mask must have shape [T, F]")