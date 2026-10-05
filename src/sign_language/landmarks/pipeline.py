from __future__ import annotations

import numpy as np

from .extractor import LandmarkExtractor
from .normalization import normalize_landmarks
from .schema import LandmarkSequence


class LandmarkPipeline:
    """Build a deterministic landmark sequence from an image frame."""

    def __init__(self, feature_dim: int = 42, sequence_length: int = 16, extractor: LandmarkExtractor | None = None) -> None:
        self.extractor = extractor or LandmarkExtractor(feature_dim=feature_dim)
        self.feature_dim = int(feature_dim or self.extractor.feature_dim)
        self.sequence_length = int(sequence_length)

    def process(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.size == 0:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        observation = self.extractor.extract_observation(arr)
        if observation.landmarks.shape[0] != self.feature_dim:
            raise ValueError(
                f"Extractor produced {observation.landmarks.shape[0]} features; expected {self.feature_dim}"
            )
        return observation.landmarks.reshape(1, -1).repeat(self.sequence_length, axis=0)

    def process_video(self, video_path: str) -> LandmarkSequence:
        import cv2

        capture = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")
            while True:
                success, frame = capture.read()
                if not success:
                    break
                observation = self.extractor.extract_observation(frame)
                frames.append(observation.landmarks)
                masks.append(observation.mask)
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return LandmarkSequence(np.stack(frames), np.stack(masks))

<<<<<<< HEAD
    def normalize_sequence(self, sequence: np.ndarray, include_dynamics: bool = True) -> np.ndarray:
=======
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {arr.shape[-1]}")
<<<<<<< HEAD
            
        # We assume normalize_landmarks exists in .normalization
        # The previous code called normalize_landmarks with mode="center_scale"
        # However normalize_landmarks only takes landmarks and eps. Let's fix that.
        from .normalization import normalize_landmarks
        normalized = normalize_landmarks(arr)
        
=======
        normalized = normalize_landmarks(arr, mode="center_scale")
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
        if normalized.shape[0] < self.sequence_length:
            pad = np.zeros((self.sequence_length - normalized.shape[0], normalized.shape[1]), dtype=np.float32)
            normalized = np.concatenate([normalized, pad], axis=0)
        elif normalized.shape[0] > self.sequence_length:
            normalized = normalized[: self.sequence_length]
<<<<<<< HEAD
            
        if include_dynamics:
            velocities = np.zeros_like(normalized)
            velocities[1:] = normalized[1:] - normalized[:-1]
            accelerations = np.zeros_like(velocities)
            accelerations[1:] = velocities[1:] - velocities[:-1]
            return np.concatenate([normalized, velocities, accelerations], axis=-1)
            
=======
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
        return normalized
