from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from .schema import LandmarkObservation


class LandmarkExtractor:
    """Extract holistic landmarks with MediaPipe Tasks when an asset is configured."""

    _MODALITY_COUNTS: ClassVar[dict[str, int]] = {
        "pose": 33,
        "face": 478,
        "left_hand": 21,
        "right_hand": 21,
    }

    def __init__(
        self,
        feature_dim: int | None = None,
        model_asset_path: str | Path | None = None,
        representation: str = "hands",
    ) -> None:
        if representation not in {"hands", "hands_pose", "holistic"}:
            raise ValueError(f"Unsupported landmark representation: {representation}")
        self.model_asset_path = Path(model_asset_path) if model_asset_path is not None else None
        self.representation = representation
        self.feature_dim = int(feature_dim or self._feature_dim(representation))
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self._landmarker = None

    @classmethod
    def _feature_dim(cls, representation: str) -> int:
        point_count = cls._MODALITY_COUNTS["left_hand"] + cls._MODALITY_COUNTS["right_hand"]
        if representation in {"hands_pose", "holistic"}:
            point_count += cls._MODALITY_COUNTS["pose"]
        if representation == "holistic":
            point_count += cls._MODALITY_COUNTS["face"]
        return point_count * 3

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        if self.model_asset_path is None:
            raise RuntimeError("A MediaPipe HolisticLandmarker .task asset is required for extraction")
        if not self.model_asset_path.is_file():
            raise FileNotFoundError(f"MediaPipe model asset does not exist: {self.model_asset_path}")

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HolisticLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_asset_path)),
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        return self._landmarker

    @staticmethod
    def _points(values: list | None, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.zeros((expected_count, 3), dtype=np.float32)
        mask = np.zeros((expected_count, 3), dtype=np.float32)
        if values:
            for index, point in enumerate(values[:expected_count]):
                coordinates[index] = (point.x, point.y, point.z)
                mask[index] = 1.0
        return coordinates, mask

    def extract_observation(self, frame: np.ndarray) -> LandmarkObservation:
        frame = np.asarray(frame)
        if frame.size == 0:
            return LandmarkObservation(
                np.zeros(self.feature_dim, dtype=np.float32),
                np.zeros(self.feature_dim, dtype=np.float32),
            )
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Input frame must be a 2D grayscale image or a 3-channel RGB image")
        if self.model_asset_path is None:
            return LandmarkObservation(
                np.zeros(self.feature_dim, dtype=np.float32),
                np.zeros(self.feature_dim, dtype=np.float32),
            )

        import mediapipe as mp

        result = self._get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
        modalities = {
            "left_hand": getattr(result, "left_hand_landmarks", None),
            "right_hand": getattr(result, "right_hand_landmarks", None),
            "pose": getattr(result, "pose_landmarks", None),
            "face": getattr(result, "face_landmarks", None),
        }
        values: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        selected = ["left_hand", "right_hand"]
        if self.representation in {"hands_pose", "holistic"}:
            selected.append("pose")
        if self.representation == "holistic":
            selected.append("face")
        for name in selected:
            points, mask = self._points(modalities[name], self._MODALITY_COUNTS[name])
            values.append(points.reshape(-1))
            masks.append(mask.reshape(-1))
        landmarks = np.concatenate(values)
        mask = np.concatenate(masks)
        if landmarks.shape[0] != self.feature_dim:
            if landmarks.shape[0] > self.feature_dim:
                landmarks = landmarks[: self.feature_dim]
                mask = mask[: self.feature_dim]
            else:
                padding = self.feature_dim - landmarks.shape[0]
                landmarks = np.pad(landmarks, (0, padding))
                mask = np.pad(mask, (0, padding))
        return LandmarkObservation(landmarks, mask)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return self.extract_observation(frame).landmarks
