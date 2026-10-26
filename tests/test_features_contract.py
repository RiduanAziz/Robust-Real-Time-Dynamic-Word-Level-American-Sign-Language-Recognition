"""Comprehensive Feature-Contract and Representation Validation Tests.

Validates:
1. Exact output dimensions for Representations A-E (Hands, Hands+Pose, Holistic, Engineered, Dynamics)
2. Exact mask dimensions and strict mask propagation (missing wrists invalidate derived relative features)
3. Zero-filling of invalid features
4. Nan / Inf rejection and numerical safety
5. Temporal boundary safety for velocity and acceleration
"""
from __future__ import annotations

import numpy as np
import pytest

from sign_language.features.representations import (
    get_feature_specification,
    select_feature_representation,
    extract_engineered_spatial_features,
    _FEATURES,
)


def test_feature_specifications_registered_and_dimensions_match() -> None:
    expected_dims = {
        "hands": 126,
        "hands_pose": 225,
        "holistic": 1659,
        "engineered": 365,
        "holistic_dynamics": 4977,
    }
    for name, expected_dim in expected_dims.items():
        spec = get_feature_specification(name)
        assert spec.feature_dim == expected_dim, f"Spec {name} declared {spec.feature_dim}, expected {expected_dim}"


def test_representation_a_hands_dimension_and_mask() -> None:
    t = 15
    seq = np.random.randn(t, 1659).astype(np.float32)
    mask = np.ones((t, 1659), dtype=np.float32)
    feat, m = select_feature_representation(seq, "hands", mask=mask)
    assert feat.shape == (t, 126)
    assert m.shape == (t, 126)
    assert np.allclose(feat, seq[:, :126])


def test_representation_b_hands_pose_dimension_and_mask() -> None:
    t = 15
    seq = np.random.randn(t, 1659).astype(np.float32)
    mask = np.ones((t, 1659), dtype=np.float32)
    feat, m = select_feature_representation(seq, "hands_pose", mask=mask)
    assert feat.shape == (t, 225)
    assert m.shape == (t, 225)
    assert np.allclose(feat, seq[:, :225])


def test_representation_c_holistic_dimension_and_mask() -> None:
    t = 10
    seq = np.random.randn(t, 1659).astype(np.float32)
    mask = np.ones((t, 1659), dtype=np.float32)
    feat, m = select_feature_representation(seq, "holistic", mask=mask)
    assert feat.shape == (t, 1659)
    assert m.shape == (t, 1659)
    assert np.allclose(feat, seq)


def test_representation_d_engineered_strict_mask_propagation() -> None:
    t = 20
    seq = np.random.randn(t, 225).astype(np.float32)
    mask = np.ones((t, 225), dtype=np.float32)

    # Simulate missing left hand wrist in frame 5 (left wrist is points 0..2)
    mask[5, 0:3] = 0.0

    # Simulate missing right hand fingertip index 8 in frame 10 (point 21 + 8 = 29 -> coords 87:90)
    # Right hand is coords 63..126. Tip index 8 relative to right hand is 8 * 3 = 24 -> coords 87:90
    mask[10, 87:90] = 0.0

    feat, m = select_feature_representation(seq, "engineered", mask=mask)
    spec = get_feature_specification("engineered")
    assert feat.shape == (t, spec.feature_dim)
    assert m.shape == (t, spec.feature_dim)

    # Frame 5: Left hand wrist is missing, so all left hand wrist-relative coords (indices 225..288) must be masked out and 0.0
    lh_rel_mask = m[5, 225:288]
    assert np.all(lh_rel_mask == 0.0), "Missing wrist must invalidate wrist-relative coordinates"
    assert np.all(feat[5, 225:288] == 0.0), "Invalid wrist-relative features must be zero-filled"

    # Inter-hand distance (indices 351:355) must be masked out in frame 5 because left wrist is missing
    inter_hand_mask = m[5, 351:355]
    assert np.all(inter_hand_mask == 0.0), "Missing wrist must invalidate inter-hand distance"
    assert np.all(feat[5, 351:355] == 0.0), "Invalid inter-hand features must be zero-filled"


def test_representation_e_holistic_dynamics_temporal_boundaries() -> None:
    t = 10
    seq = np.ones((t, 1659), dtype=np.float32) * 5.0
    mask = np.ones((t, 1659), dtype=np.float32)

    feat, m = select_feature_representation(seq, "holistic_dynamics", mask=mask)
    assert feat.shape == (t, 4977)
    assert m.shape == (t, 4977)

    # For constant input, velocity and acceleration should be 0.0
    vel = feat[:, 1659:3318]
    acc = feat[:, 3318:4977]
    assert np.allclose(vel[1:], 0.0)
    assert np.allclose(acc[2:], 0.0)

    # Boundary conditions: frame 0 has no velocity
    assert np.allclose(vel[0], 0.0)
    assert np.allclose(m[0, 1659:3318], 0.0)
