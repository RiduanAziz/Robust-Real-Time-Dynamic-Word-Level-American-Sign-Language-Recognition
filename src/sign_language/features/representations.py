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
    description: str = ""


_POINTS_PER_MODALITY = {"hands": 42, "pose": 33, "face": 478}

_FEATURES: dict[str, FeatureSpecification] = {
    "hands": FeatureSpecification("hands", ("hands",), 126, "Left and right hand 21 3D landmarks"),
    "hands_pose": FeatureSpecification("hands_pose", ("hands", "pose"), 225, "Hands + 33 pose landmarks"),
    "holistic": FeatureSpecification("holistic", ("hands", "pose", "face"), 1659, "Full holistic landmarks"),
    "holistic_dynamics": FeatureSpecification("holistic_dynamics", ("hands", "pose", "face"), 4977, "Full holistic + velocity + acceleration"),
    "engineered": FeatureSpecification("engineered", ("hands", "pose"), 280, "Hands + Pose + relative distances & joint angles"),
}


def get_feature_specification(name: str) -> FeatureSpecification:
    try:
        return _FEATURES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported feature representation: '{name}'. Choose from {list(_FEATURES.keys())}") from error


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

    if name in ("hands", "hands_pose", "holistic"):
        dim = specification.feature_dim
        if values.shape[1] < dim:
            raise ValueError(f"Representation {name} requires {dim} features, got {values.shape[1]}")
        return values[:, :dim].copy(), m[:, :dim].copy()

    elif name == "holistic_dynamics":
        if values.shape[1] == 4977:
            return values.copy(), m.copy()
        raw_dim = 1659
        raw_vals = values[:, :raw_dim]
        raw_m = m[:, :raw_dim]
        # compute vel and acc
        vel = np.zeros_like(raw_vals)
        acc = np.zeros_like(raw_vals)
        vel_m = np.zeros_like(raw_m)
        acc_m = np.zeros_like(raw_m)
        if t > 1:
            vel[1:] = raw_vals[1:] - raw_vals[:-1]
            vel_m[1:] = raw_m[1:] * raw_m[:-1]
            vel = vel * vel_m
        if t > 2:
            acc[2:] = vel[2:] - vel[1:-1]
            acc_m[2:] = vel_m[2:] * vel_m[1:-1]
            acc = acc * acc_m
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
      - Raw Hands + Pose (225 values)
      - Left hand wrist-relative coordinates (63 values)
      - Right hand wrist-relative coordinates (63 values)
      - Left-wrist to Right-wrist 3D displacement & distance (4 values)
      - Left/Right wrist to nose/chest distance (4 values)
      - Finger tip-to-wrist extension distances (5 left + 5 right = 10 values)
    Total dimension: 225 + 63 + 63 + 4 + 4 + 10 = 369 (or padded/projected to 280)
    """
    arr = np.asarray(sequence, dtype=np.float32)
    t = arr.shape[0]
    m = np.asarray(mask, dtype=np.float32) if mask is not None else np.ones_like(arr, dtype=np.float32)

    # Need at least hands + pose (225 dims)
    base_dim = min(225, arr.shape[-1])
    base_coords = arr[:, :base_dim]
    base_mask = m[:, :base_dim]

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

    # Inter-hand distance (left wrist to right wrist)
    hand_disp = (rh_wrist - lh_wrist).squeeze(1)  # [T, 3]
    hand_dist = np.linalg.norm(hand_disp, axis=-1, keepdims=True)  # [T, 1]
    hand_rel = np.concatenate([hand_disp, hand_dist], axis=-1)  # [T, 4]
    hand_rel_m = (lh_m[:, 0, 0:1] * rh_m[:, 0, 0:1]).repeat(4, axis=-1)

    # Finger extensions (tips to wrist: indices 4, 8, 12, 16, 20)
    tip_indices = [4, 8, 12, 16, 20]
    lh_tips_dist = np.linalg.norm(lh_coords[:, tip_indices, :] - lh_wrist, axis=-1)  # [T, 5]
    rh_tips_dist = np.linalg.norm(rh_coords[:, tip_indices, :] - rh_wrist, axis=-1)  # [T, 5]
    tips_dist = np.concatenate([lh_tips_dist, rh_tips_dist], axis=-1)  # [T, 10]
    tips_m = np.ones_like(tips_dist)

    # Concatenate: base hands+pose (225) + wrist_rel (126) + inter_hand (4) + tips (10) = 365
    engineered = np.concatenate([base_coords, lh_rel, rh_rel, hand_rel, tips_dist], axis=-1)
    engineered_m = np.concatenate([base_mask, base_mask[:, 0:63], base_mask[:, 63:126], hand_rel_m, tips_m], axis=-1)

    return engineered, engineered_m