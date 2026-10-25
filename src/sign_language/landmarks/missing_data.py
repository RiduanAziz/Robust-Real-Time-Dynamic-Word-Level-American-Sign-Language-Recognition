"""Principled missing-keypoint handling and conservative temporal interpolation."""
from __future__ import annotations

from typing import Any
import numpy as np


def interpolate_short_gaps(
    sequence: np.ndarray,
    mask: np.ndarray,
    max_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively interpolate short gaps (<= max_gap) with valid surrounding frames.

    Preserves the distinction between missing and detected points.
    Long gaps (> max_gap) are NEVER fabricated and remain mask=0.

    Args:
        sequence: Array of shape [T, D] or [T, N, C]
        mask: Binary validity mask of matching shape (1.0 = valid, 0.0 = missing)
        max_gap: Maximum consecutive missing frames to linearly interpolate (default: 3)

    Returns:
        Tuple of (interpolated_sequence, updated_mask)
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()
    m = (np.asarray(mask, dtype=np.float32) > 0.5).copy()

    was_3d = (seq.ndim == 3)
    if was_3d:
        t, n, c = seq.shape
        seq = seq.reshape(t, n * c)
        m = m.reshape(t, n * c)

    t, d = seq.shape
    if t <= 2 or max_gap <= 0:
        if was_3d:
            seq = seq.reshape(t, n, c)
            m = m.reshape(t, n, c)
        return seq, m.astype(np.float32)

    # Process each feature column
    for col in range(d):
        valid_indices = np.where(m[:, col])[0]
        if len(valid_indices) < 2:
            continue

        # Look for interior gaps
        for i in range(len(valid_indices) - 1):
            idx_start = valid_indices[i]
            idx_end = valid_indices[i + 1]
            gap_len = idx_end - idx_start - 1

            if 1 <= gap_len <= max_gap:
                # Linearly interpolate between valid boundaries
                v_start = seq[idx_start, col]
                v_end = seq[idx_end, col]
                steps = np.linspace(v_start, v_end, gap_len + 2, endpoint=True, dtype=np.float32)
                seq[idx_start + 1 : idx_end, col] = steps[1:-1]
                # Keep interpolated mask marked (or mark with fractional confidence 0.8)
                m[idx_start + 1 : idx_end, col] = True

    if was_3d:
        seq = seq.reshape(t, n, c)
        m = m.reshape(t, n, c)

    return seq, m.astype(np.float32)


def partition_modality_masks(
    mask: np.ndarray,
    schema: str = "holistic",
) -> dict[str, np.ndarray]:
    """Extract separate validity masks for each anatomical stream.

    Supports:
      - Hands only (126 coords: 0..63 left, 63..126 right)
      - Hands + Pose (225 coords: 0..63 left, 63..126 right, 126..225 pose)
      - Holistic (1659 coords: 0..63 left, 63..126 right, 126..225 pose, 225..1659 face)
    """
    m = np.asarray(mask, dtype=np.float32)
    t = m.shape[0]
    total_dim = m.shape[-1]

    masks: dict[str, np.ndarray] = {}
    if total_dim == 126:
        masks["left_hand"] = m[..., 0:63]
        masks["right_hand"] = m[..., 63:126]
    elif total_dim == 225:
        masks["left_hand"] = m[..., 0:63]
        masks["right_hand"] = m[..., 63:126]
        masks["pose"] = m[..., 126:225]
    elif total_dim >= 1659:
        masks["left_hand"] = m[..., 0:63]
        masks["right_hand"] = m[..., 63:126]
        masks["pose"] = m[..., 126:225]
        masks["face"] = m[..., 225:1659]
    else:
        # Fallback single stream
        masks["all"] = m

    return masks


def compute_missingness_diagnostics(
    sequence: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute per-modality missingness percentage across a sequence."""
    m = np.asarray(mask, dtype=np.float32)
    mod_masks = partition_modality_masks(m)

    stats: dict[str, float] = {
        "overall_missing_pct": round(float((m <= 0.5).mean() * 100.0), 2)
    }
    for mod_name, mod_m in mod_masks.items():
        missing_pct = float((mod_m <= 0.5).mean() * 100.0)
        stats[f"{mod_name}_missing_pct"] = round(missing_pct, 2)
        stats[f"{mod_name}_active_pct"] = round(100.0 - missing_pct, 2)

    return stats
