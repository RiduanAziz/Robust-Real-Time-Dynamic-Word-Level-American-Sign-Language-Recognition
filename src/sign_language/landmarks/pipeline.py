from __future__ import annotations

import cv2
import numpy as np

from .extractor import LandmarkExtractor
from .normalization import normalize_landmarks
from .schema import LandmarkSequence


def resample_sequence(
    sequence: np.ndarray,
    target_length: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly resample a temporal sequence [T, D] to target_length [L, D].

    Preserves the start, middle, and end of dynamic signs without front-truncation.
    """
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Sequence must have shape [T, D], got {arr.shape}")

    t_in, d = arr.shape
    if t_in == 0:
        empty = np.zeros((target_length, d), dtype=np.float32)
        return empty, empty

    if t_in == target_length:
        resampled_mask = mask if mask is not None else np.ones_like(arr, dtype=np.float32)
        return arr.copy(), resampled_mask.copy()

    # High-speed vectorized linear interpolation along time dimension (axis 0)
    # Using PyTorch 1D interpolation kernel (1.3ms vs 35ms in Python loop)
    import torch
    import torch.nn.functional as F

    tensor_in = torch.from_numpy(arr).unsqueeze(0).transpose(1, 2)
    resampled_tensor = F.interpolate(
        tensor_in, size=target_length, mode="linear", align_corners=True
    )
    resampled = resampled_tensor.squeeze(0).transpose(0, 1).contiguous().numpy()

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float32)
        mask_in = torch.from_numpy(mask_arr).unsqueeze(0).transpose(1, 2)
        res_mask_tensor = F.interpolate(
            mask_in, size=target_length, mode="linear", align_corners=True
        )
        resampled_mask = (
            res_mask_tensor.squeeze(0).transpose(0, 1).contiguous().numpy() >= 0.5
        ).astype(np.float32)
    else:
        resampled_mask = np.ones((target_length, d), dtype=np.float32)

    return resampled, resampled_mask


def compute_temporal_derivatives(
    sequence: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute position, velocity, and acceleration with correct boundary and mask handling."""
    pos = np.asarray(sequence, dtype=np.float32)
    t, d = pos.shape

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    if t > 1:
        vel[1:] = pos[1:] - pos[:-1]
    if t > 2:
        acc[2:] = vel[2:] - vel[1:-1]

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float32)
        vel_mask = np.zeros_like(mask_arr)
        if t > 1:
            vel_mask[1:] = mask_arr[1:] * mask_arr[:-1]
        vel = vel * vel_mask

        acc_mask = np.zeros_like(mask_arr)
        if t > 2:
            acc_mask[2:] = vel_mask[2:] * vel_mask[1:-1]
        acc = acc * acc_mask
    else:
        vel_mask = np.ones_like(vel)
        acc_mask = np.ones_like(acc)

    return pos, vel, acc


def canonical_preprocess_sequence(
    sequence: np.ndarray,
    mask: np.ndarray | None = None,
    target_length: int = 64,
    feature_representation: str = "holistic",
    augment: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical, authoritative preprocessing pipeline for training, evaluation, and inference.

    Steps:
      1. Parse raw coordinates and validity mask.
      2. If augment=True (training only):
         - Coordinate jitter on valid landmarks
         - Random spatial scaling
         - Random landmark dropout: dropped coordinates become 0.0 AND are strictly marked invalid in mask.
      3. Geometric centering and scaling per frame across visible points.
      4. Vectorized temporal resampling along time dimension to target_length.
      5. Authoritative feature representation extraction (holistic, hands_pose, hands, engineered, holistic_dynamics).
      6. Finite check & zero-fill invalid elements.
    """
    from sign_language.features.representations import (
        resolve_feature_representation,
        select_feature_representation,
    )

    spec = resolve_feature_representation(feature_representation)
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0 or arr.shape[0] == 0:
        empty_feat = np.zeros((target_length, spec.feature_dim), dtype=np.float32)
        empty_mask = np.zeros((target_length, spec.feature_dim), dtype=np.float32)
        return empty_feat, empty_mask

    m = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(arr, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)

    t_frames = arr.shape[0]
    total_raw_dims = arr.shape[1]
    num_points = min(total_raw_dims // 3, 553)
    raw_dim = num_points * 3

    coords = arr[:, :raw_dim].reshape(t_frames, num_points, 3).copy()
    m_pts = m[:, :raw_dim].reshape(t_frames, num_points, 3)
    pt_valid = (m_pts > 0.5).any(axis=-1, keepdims=True).astype(np.float32)
    coords = np.where(pt_valid > 0.5, coords, np.float32(0.0))

    # Training-only augmentation
    if augment:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        # 1. Subtle spatial Gaussian coordinate jitter on visible landmarks
        jitter = rng.normal(0.0, 0.006, size=coords.shape).astype(np.float32)
        coords = coords + np.where(pt_valid > 0.5, jitter, np.float32(0.0))

        # 2. Random spatial uniform scaling
        scale_factor = np.float32(rng.uniform(0.94, 1.06))
        coords = coords * scale_factor

        # 3. Random landmark dropout: dropped points become 0.0 AND pt_valid becomes 0.0
        if rng.rand() < 0.25:
            drop_pts = (rng.rand(num_points) > 0.04).reshape(1, num_points, 1).astype(np.float32)
            coords = coords * drop_pts
            pt_valid = pt_valid * drop_pts

    # Spatial centering & scaling
    n_valid = pt_valid.sum(axis=1, keepdims=True)
    has_valid = (n_valid > 0)
    safe_n = np.maximum(n_valid, 1.0)
    center = np.where(has_valid, (coords * pt_valid).sum(axis=1, keepdims=True) / safe_n, np.float32(0.0))
    centered = np.where(pt_valid > 0.5, coords - center, np.float32(0.0))
    sq_sum = np.sum(np.square(centered), axis=(1, 2), keepdims=True)
    scale = np.sqrt(np.where(has_valid, sq_sum / (safe_n * np.float32(3.0)), np.float32(1.0)))
    scale = np.where(scale >= 1e-6, scale, np.float32(1.0))
    norm_coords = np.where(pt_valid > 0.5, centered / scale, np.float32(0.0))

    norm_seq = norm_coords.reshape(t_frames, raw_dim)
    norm_mask = (pt_valid.repeat(3, axis=-1)).reshape(t_frames, raw_dim)

    # Vectorized temporal resampling
    resampled_seq, resampled_mask = resample_sequence(
        norm_seq, target_length=target_length, mask=norm_mask
    )
    resampled_seq = np.where(resampled_mask > 0.5, resampled_seq, np.float32(0.0))

    # Feature representation extraction
    feat, feat_mask = select_feature_representation(
        resampled_seq, name=spec.name, mask=resampled_mask
    )

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat_mask = np.where(np.isfinite(feat_mask) & (feat_mask > 0.5), 1.0, 0.0).astype(np.float32)
    feat = np.where(feat_mask > 0.5, feat, np.float32(0.0))

    return feat, feat_mask


class LandmarkPipeline:
    """Build a deterministic landmark sequence and feature representations."""

    def __init__(
        self,
        feature_dim: int = 1659,
        sequence_length: int = 64,
        extractor: LandmarkExtractor | None = None,
        feature_representation: str | None = None,
    ) -> None:
        from sign_language.features.representations import resolve_feature_representation

        self._extractor = extractor
        self.sequence_length = int(sequence_length)
        if feature_representation:
            self.specification = resolve_feature_representation(feature_representation)
            self.feature_dim = self.specification.feature_dim
        else:
            self.specification = resolve_feature_representation(feature_dim)
            self.feature_dim = self.specification.feature_dim
        self.feature_representation = self.specification.name

    @property
    def extractor(self) -> LandmarkExtractor:
        if self._extractor is None:
            self._extractor = LandmarkExtractor(feature_dim=self.feature_dim)
        return self._extractor

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame into landmark features.

        Warning: Only for single-frame debugging. Dynamic motion models must be
        evaluated on real temporal video sequences, never single-frame repetitions.
        """
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
        """Read real video frames sequentially with monotonic timestamps."""
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        timestamps_ms: list[float] = []

        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            frame_idx = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                ts_ms = frame_idx * (1000.0 / fps)
                obs = self.extractor.extract_observation(frame, timestamp_ms=int(ts_ms))
                frames.append(obs.landmarks)
                masks.append(obs.mask)
                timestamps_ms.append(ts_ms)
                frame_idx += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")

        raw_landmarks = np.stack(frames)
        raw_masks = np.stack(masks)
        raw_timestamps = np.array(timestamps_ms, dtype=np.float32)

        return LandmarkSequence(
            landmarks=raw_landmarks,
            mask=raw_masks,
            sequence_mask=np.ones(len(frames), dtype=np.float32),
            timestamps_ms=raw_timestamps,
        )

    def normalize_sequence_with_mask(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        include_dynamics: bool = False,
        feature_representation: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize geometric landmarks, resample uniformly, and compute feature representations.

        Returns:
            Tuple of (features, feature_mask) with matching shapes [T, D].
        """
        rep_name = feature_representation or (
            "holistic_dynamics" if include_dynamics else self.feature_representation
        )
        return canonical_preprocess_sequence(
            sequence=sequence,
            mask=mask,
            target_length=self.sequence_length,
            feature_representation=rep_name,
            augment=False,
        )

    def normalize_sequence(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        include_dynamics: bool = False,
        feature_representation: str | None = None,
    ) -> np.ndarray:
        """Normalize geometric landmarks, resample uniformly, and return features."""
        features, _ = self.normalize_sequence_with_mask(
            sequence, mask=mask, include_dynamics=include_dynamics, feature_representation=feature_representation
        )
        return features
