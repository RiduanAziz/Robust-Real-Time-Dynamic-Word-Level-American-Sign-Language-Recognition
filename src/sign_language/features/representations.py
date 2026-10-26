"""Feature representation specifications and engineered motion representations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class FeatureSpecification:
    name: str
    modalities: tuple[str, ...]
    feature_dim: int
    schema_version: str = "2.0.0"
    source_landmark_indices: tuple[int, ...] | str = "all"
    coordinate_order: str = "(x, y, z)"
    mask_semantics: str = "1.0=valid landmark, 0.0=missing/occluded"
    normalization_method: str = "per-frame center-scale on visible points"
    temporal_processing_policy: str = "linear temporal interpolation with causal padding"
    description: str = ""


_POINTS_PER_MODALITY = {"hands": 42, "pose": 33, "face": 478}

_FEATURES: dict[str, FeatureSpecification] = {
    "hands": FeatureSpecification(
        name="hands",
        modalities=("hands",),
        feature_dim=126,
        source_landmark_indices=tuple(range(0, 42)),
        coordinate_order="(x, y, z)",
        description="Left (0..20) and Right (21..41) hand 3D landmarks (126 dims)",
    ),
    "hands_pose": FeatureSpecification(
        name="hands_pose",
        modalities=("hands", "pose"),
        feature_dim=225,
        source_landmark_indices=tuple(range(0, 75)),
        coordinate_order="(x, y, z)",
        description="Hands (42) + Pose (33) 3D landmarks (225 dims)",
    ),
    "holistic": FeatureSpecification(
        name="holistic",
        modalities=("hands", "pose", "face"),
        feature_dim=1659,
        source_landmark_indices=tuple(range(0, 553)),
        coordinate_order="(x, y, z)",
        description="Full 553 MediaPipe holistic 3D landmarks (1659 dims)",
    ),
    "holistic_dynamics": FeatureSpecification(
        name="holistic_dynamics",
        modalities=("hands", "pose", "face"),
        feature_dim=4977,
        source_landmark_indices=tuple(range(0, 553)),
        coordinate_order="Position (1659) + Velocity (1659) + Acceleration (1659)",
        description="Full holistic landmarks with first and second temporal derivatives (4977 dims)",
    ),
    "engineered": FeatureSpecification(
        name="engineered",
        modalities=("hands", "pose"),
        feature_dim=365,
        source_landmark_indices=tuple(range(0, 75)),
        coordinate_order="Base hands+pose (225) + Left wrist-rel (63) + Right wrist-rel (63) + Inter-wrist (4) + Fingertip extensions (10)",
        description="Kinematic interaction features from hands and pose (365 dims)",
    ),
}


def get_feature_specification(name: str) -> FeatureSpecification:
    return resolve_feature_representation(name)


def resolve_feature_representation(identifier: str | int | None) -> FeatureSpecification:
    """Resolve a feature specification from representation name, dimension int, or config string."""
    if identifier is None:
        return _FEATURES["holistic"]
    if isinstance(identifier, int):
        dim_map = {
            126: "hands",
            225: "hands_pose",
            365: "engineered",
            1659: "holistic",
            4977: "holistic_dynamics",
        }
        if identifier in dim_map:
            return _FEATURES[dim_map[identifier]]
        pts = max(1, identifier // 3)
        return FeatureSpecification(
            name=f"custom_{identifier}",
            modalities=("custom",),
            feature_dim=identifier,
            source_landmark_indices=tuple(range(pts)),
            coordinate_order="(x, y, z)",
            mask_semantics="1.0=valid landmark, 0.0=missing/occluded",
            normalization_method="per-frame center-scale on visible points",
            temporal_processing_policy="linear temporal interpolation with causal padding",
            schema_version="2.0.0",
            description="Custom/test dimension representation",
        )
    key = str(identifier).strip().lower()
    if key in _FEATURES:
        return _FEATURES[key]
    if key.startswith("custom_"):
        parts = key.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            return resolve_feature_representation(int(parts[1]))
    if key.isdigit():
        return resolve_feature_representation(int(key))
    raise ValueError(
        f"Unsupported feature representation: '{identifier}'. Choose from {list(_FEATURES.keys())}"
    )


def select_feature_representation(
    sequence: np.ndarray,
    name: str,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a documented representation and corresponding validity mask.

    Returns:
        Tuple of (feature_sequence [T, D], feature_mask [T, D])
    """
    specification = get_feature_specification(name)
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Feature sequences must have shape [T, F], got {values.shape}")

    t = values.shape[0]
    m = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(values, dtype=np.float32)

    if name in ("hands", "hands_pose", "holistic") or name.startswith("custom_"):
        dim = specification.feature_dim
        if values.shape[1] < dim:
            pad_width = dim - values.shape[1]
            feat = np.pad(values, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
            feat_m = np.pad(m, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        else:
            feat = values[:, :dim].copy()
            feat_m = m[:, :dim].copy()
        feat = np.where(feat_m > 0.5, feat, 0.0)
        return feat, feat_m

    elif name == "holistic_dynamics":
        raw_dim = 1659
        if values.shape[1] < raw_dim:
            raise ValueError(f"holistic_dynamics requires at least {raw_dim} features, got {values.shape[1]}")
        raw_vals = values[:, :raw_dim].copy()
        raw_m = (m[:, :raw_dim] > 0.5).astype(np.float32)
        raw_vals = np.where(raw_m > 0.5, raw_vals, 0.0)

        # Compute velocity and acceleration with boundary masks
        vel = np.zeros_like(raw_vals)
        acc = np.zeros_like(raw_vals)
        vel_m = np.zeros_like(raw_m)
        acc_m = np.zeros_like(raw_m)
        if t > 1:
            vel[1:] = raw_vals[1:] - raw_vals[:-1]
            vel_m[1:] = ((raw_m[1:] > 0.5) & (raw_m[:-1] > 0.5)).astype(np.float32)
            vel = np.where(vel_m > 0.5, vel, 0.0)
        if t > 2:
            acc[2:] = vel[2:] - vel[1:-1]
            acc_m[2:] = ((vel_m[2:] > 0.5) & (vel_m[1:-1] > 0.5)).astype(np.float32)
            acc = np.where(acc_m > 0.5, acc, 0.0)

        # Ensure no non-finite values reach the model
        raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

        full_feat = np.concatenate([raw_vals, vel, acc], axis=-1)
        full_mask = np.concatenate([raw_m, vel_m, acc_m], axis=-1)
        return full_feat, full_mask

    elif name == "engineered":
        return extract_engineered_spatial_features(values, m)

    raise ValueError(f"Unhandled representation: {name}")


def extract_engineered_spatial_features(
    sequence: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute anatomically grounded spatial and interaction features.

    Features include:
      - Raw Hands + Pose: 225 values (42 hands + 33 pose * 3)
      - Left hand wrist-relative coordinates: 63 values (21 * 3)
      - Right hand wrist-relative coordinates: 63 values (21 * 3)
      - Inter-wrist 3D displacement & distance: 4 values (3 vector + 1 norm)
      - Finger tip-to-wrist extension distances: 10 values (5 left + 5 right)
    Total dimension: 225 + 63 + 63 + 4 + 10 = 365.
    """
    arr = np.asarray(sequence, dtype=np.float32)
    t = arr.shape[0]
    m = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(arr, dtype=np.float32)

    if arr.shape[-1] < 225:
        raise ValueError(
            f"extract_engineered_spatial_features requires at least 225 features (hands+pose), got {arr.shape[-1]}"
        )

    base_coords = arr[:, :225]
    base_mask = (m[:, :225] > 0.5).astype(np.float32)
    base_coords = np.where(base_mask > 0.5, base_coords, 0.0)

    # Parse keypoints: Left hand 0..20, Right hand 21..41, Pose 42..74
    # Left hand [T, 21, 3]
    lh_coords = base_coords[:, 0:63].reshape(t, 21, 3)
    lh_m = base_mask[:, 0:63].reshape(t, 21, 3)
    # Right hand [T, 21, 3]
    rh_coords = base_coords[:, 63:126].reshape(t, 21, 3)
    rh_m = base_mask[:, 63:126].reshape(t, 21, 3)

    # Wrist relative coords
    lh_wrist = lh_coords[:, 0:1, :]  # [T, 1, 3]
    rh_wrist = rh_coords[:, 0:1, :]
    lh_rel = (lh_coords - lh_wrist).reshape(t, 63)
    rh_rel = (rh_coords - rh_wrist).reshape(t, 63)
    lh_rel_m = (lh_m * lh_m[:, 0:1, :]).reshape(t, 63)
    rh_rel_m = (rh_m * rh_m[:, 0:1, :]).reshape(t, 63)
    lh_rel = np.where(lh_rel_m > 0.5, lh_rel, 0.0)
    rh_rel = np.where(rh_rel_m > 0.5, rh_rel, 0.0)

    # Inter-hand distance (left wrist to right wrist)
    hand_disp = (rh_wrist - lh_wrist).squeeze(1)  # [T, 3]
    hand_dist = np.linalg.norm(hand_disp, axis=-1, keepdims=True)  # [T, 1]
    hand_rel = np.concatenate([hand_disp, hand_dist], axis=-1)  # [T, 4]
    hand_rel_m = (lh_m[:, 0:1, 0] * rh_m[:, 0:1, 0]).repeat(4, axis=-1)
    hand_rel = np.where(hand_rel_m > 0.5, hand_rel, 0.0)

    # Finger extensions (tips to wrist: indices 4, 8, 12, 16, 20)
    tip_indices = [4, 8, 12, 16, 20]
    lh_tips_dist = np.linalg.norm(lh_coords[:, tip_indices, :] - lh_wrist, axis=-1)  # [T, 5]
    rh_tips_dist = np.linalg.norm(rh_coords[:, tip_indices, :] - rh_wrist, axis=-1)  # [T, 5]
    tips_dist = np.concatenate([lh_tips_dist, rh_tips_dist], axis=-1)  # [T, 10]
    lh_tips_m = lh_m[:, tip_indices, 0] * lh_m[:, 0:1, 0]  # [T, 5]
    rh_tips_m = rh_m[:, tip_indices, 0] * rh_m[:, 0:1, 0]  # [T, 5]
    tips_m = np.concatenate([lh_tips_m, rh_tips_m], axis=-1)  # [T, 10]
    tips_dist = np.where(tips_m > 0.5, tips_dist, 0.0)

    # Concatenate: base hands+pose (225) + wrist_rel (126) + inter_hand (4) + tips (10) = 365
    engineered = np.concatenate([base_coords, lh_rel, rh_rel, hand_rel, tips_dist], axis=-1)
    engineered_m = np.concatenate([base_mask, lh_rel_m, rh_rel_m, hand_rel_m, tips_m], axis=-1)

    return engineered, engineered_m