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
