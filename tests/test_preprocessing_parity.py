"""Test suite for Training, Evaluation, and Inference Preprocessing Parity (SignFlow Phase 5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sign_language.landmarks.pipeline import LandmarkPipeline, canonical_preprocess_sequence
from sign_language.features.representations import (
    resolve_feature_representation,
    select_feature_representation,
    extract_engineered_spatial_features,
)


def _create_synthetic_sequence(t: int = 50, num_points: int = 553, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    seq = rng.uniform(-0.5, 0.5, size=(t, num_points * 3)).astype(np.float32)
    mask = (rng.rand(t, num_points * 3) > 0.1).astype(np.float32)
    return seq, mask


def test_preprocessing_parity_training_vs_evaluation() -> None:
    """Verify that training preprocessing (augment=False) produces bit-for-bit identical outputs to evaluation."""
    seq, mask = _create_synthetic_sequence(t=48, num_points=553, seed=123)
    target_length = 64

    for rep in ["hands", "hands_pose", "holistic", "engineered", "holistic_dynamics"]:
        # 1. Direct canonical preprocessing (training collator with augment=False)
        train_feat, train_mask = canonical_preprocess_sequence(
            sequence=seq,
            mask=mask,
            target_length=target_length,
            feature_representation=rep,
            augment=False,
        )

        # 2. Evaluation LandmarkPipeline
        spec = resolve_feature_representation(rep)
        pipeline = LandmarkPipeline(feature_dim=spec.feature_dim, sequence_length=target_length)
        eval_feat, eval_mask = pipeline.normalize_sequence_with_mask(sequence=seq, mask=mask)

        # 3. Assert strict equivalence
        assert train_feat.shape == eval_feat.shape == (target_length, spec.feature_dim), f"Shape mismatch for {rep}"
        assert train_mask.shape == eval_mask.shape == (target_length, spec.feature_dim), f"Mask shape mismatch for {rep}"
        np.testing.assert_allclose(
            train_feat,
            eval_feat,
            atol=1e-6,
            rtol=1e-5,
            err_msg=f"Feature tensor mismatch between training and evaluation for {rep}",
        )
        np.testing.assert_array_equal(
            train_mask,
            eval_mask,
            err_msg=f"Mask tensor mismatch between training and evaluation for {rep}",
        )


def test_engineered_features_segment_dimensions_and_mask_propagation() -> None:
    """Verify the 365-dimensional engineered representation segment breakdown and mask propagation."""
    t = 10
    rng = np.random.RandomState(99)
    seq = rng.uniform(-1.0, 1.0, size=(t, 225)).astype(np.float32)
    mask = np.ones((t, 225), dtype=np.float32)

    # Artificially drop Left Hand wrist at t=2 (landmark 0: coords 0..2)
    mask[2, 0:3] = 0.0
    seq[2, 0:3] = 0.0

    # Artificially drop Right Hand thumb tip at t=3 (tip index 4 in right hand: landmark 21+4=25, coords 75..77)
    mask[3, 75:78] = 0.0
    seq[3, 75:78] = 0.0

    engineered, eng_mask = extract_engineered_spatial_features(seq, mask)

    # 1. Total shape check
    assert engineered.shape == (t, 365)
    assert eng_mask.shape == (t, 365)

    # 2. Segment index boundaries:
    # Base: 0:225 (225)
    # Left wrist-rel: 225:288 (63)
    # Right wrist-rel: 288:351 (63)
    # Inter-wrist: 351:355 (4)
    # Fingertip extensions: 355:365 (10)
    assert 225 + 63 + 63 + 4 + 10 == 365

    # 3. Mask propagation verification:
    # At t=2, Left Wrist was missing -> all 63 left wrist-relative features MUST be invalid and zero
    assert np.all(eng_mask[2, 225:288] == 0.0), "Left wrist missing did not invalidate left wrist-relative mask"
    assert np.all(engineered[2, 225:288] == 0.0), "Left wrist missing did not zero left wrist-relative features"

    # At t=2, Left Wrist was missing -> inter-wrist distance (4 dims: 351:355) MUST be invalid and zero
    assert np.all(eng_mask[2, 351:355] == 0.0), "Missing wrist did not invalidate inter-wrist mask"
    assert np.all(engineered[2, 351:355] == 0.0), "Missing wrist did not zero inter-wrist features"

    # At t=2, Left Wrist was missing -> all 5 left fingertip extensions (355:360) MUST be invalid and zero
    assert np.all(eng_mask[2, 355:360] == 0.0), "Missing wrist did not invalidate left fingertip extension mask"
    assert np.all(engineered[2, 355:360] == 0.0), "Missing wrist did not zero left fingertip extensions"

    # At t=3, Right thumb tip was missing -> thumb tip extension in right hand (index 355+5=360) MUST be invalid
    assert eng_mask[3, 360] == 0.0, "Missing fingertip did not invalidate extension mask"
    assert engineered[3, 360] == 0.0, "Missing fingertip did not zero extension value"


def test_augmentation_coordinate_mask_consistency() -> None:
    """Verify that landmark dropout in training augmentation leaves no coordinate/mask contradictions."""
    seq, mask = _create_synthetic_sequence(t=30, num_points=553, seed=77)

    for seed in [1, 42, 100, 999]:
        feat, feat_mask = canonical_preprocess_sequence(
            sequence=seq,
            mask=mask,
            target_length=32,
            feature_representation="holistic",
            augment=True,
            seed=seed,
        )

        # Coordinate/mask contract: wherever mask == 0.0, coordinates MUST be 0.0
        invalid_locations = (feat_mask == 0.0)
        invalid_values = feat[invalid_locations]
        assert np.all(invalid_values == 0.0), (
            f"Coordinate/mask contradiction under seed {seed}: "
            f"found non-zero values in invalid mask locations!"
        )

        # Check finite
        assert np.all(np.isfinite(feat)), f"Non-finite values found in augmented features under seed {seed}"


def test_dynamics_derivatives_and_boundary_handling() -> None:
    """Verify velocity and acceleration derivatives, boundary masks, and short-sequence handling."""
    t = 8
    seq = np.ones((t, 1659), dtype=np.float32)
    # Linearly increasing coordinates along time: [0, 1, 2, ..., 7]
    for i in range(t):
        seq[i] = float(i)
    mask = np.ones((t, 1659), dtype=np.float32)
    # Invalidate frame t=4
    mask[4] = 0.0

    feat, feat_m = select_feature_representation(seq, "holistic_dynamics", mask=mask)

    assert feat.shape == (t, 4977)
    assert feat_m.shape == (t, 4977)

    pos = feat[:, :1659]
    vel = feat[:, 1659:3318]
    acc = feat[:, 3318:]
    vel_m = feat_m[:, 1659:3318]
    acc_m = feat_m[:, 3318:]

    # At t=0, velocity and acceleration must be 0 and mask 0
    assert np.all(vel[0] == 0.0)
    assert np.all(vel_m[0] == 0.0)
    assert np.all(acc[0] == 0.0)
    assert np.all(acc_m[0] == 0.0)

    # At t=1, velocity is 1.0, mask 1.0; acceleration must be 0 and mask 0
    assert np.all(vel[1] == 1.0)
    assert np.all(vel_m[1] == 1.0)
    assert np.all(acc[1] == 0.0)
    assert np.all(acc_m[1] == 0.0)

    # At t=2, velocity is 1.0, acceleration (vel[2]-vel[1] = 1-1 = 0), mask 1.0
    assert np.all(vel[2] == 1.0)
    assert np.all(acc[2] == 0.0)
    assert np.all(acc_m[2] == 1.0)

    # At t=4, frame is invalid -> velocity and acceleration must be invalid (0 mask and 0 value)
    assert np.all(vel_m[4] == 0.0)
    assert np.all(vel[4] == 0.0)
    assert np.all(acc_m[4] == 0.0)
    assert np.all(acc[4] == 0.0)

    # At t=5, derivative depends on t=4 which was invalid -> vel_m[5] MUST be 0.0
    assert np.all(vel_m[5] == 0.0)
    assert np.all(vel[5] == 0.0)


def test_short_sequence_and_empty_sequence_handling() -> None:
    """Verify that empty, single-frame, and all-missing sequences return well-defined zero tensors without errors."""
    # 1. Empty sequence
    empty_seq = np.zeros((0, 1659), dtype=np.float32)
    feat, m = canonical_preprocess_sequence(empty_seq, target_length=16, feature_representation="holistic")
    assert feat.shape == (16, 1659)
    assert m.shape == (16, 1659)
    assert np.all(feat == 0.0)
    assert np.all(m == 0.0)

    # 2. Single-frame sequence
    single_seq = np.ones((1, 1659), dtype=np.float32)
    feat, m = canonical_preprocess_sequence(single_seq, target_length=16, feature_representation="holistic_dynamics")
    assert feat.shape == (16, 4977)
    assert m.shape == (16, 4977)
    assert np.all(np.isfinite(feat))
