import numpy as np
import pytest

from sign_language.features import get_feature_specification, select_feature_representation


def test_feature_dimensions_are_explicit() -> None:
    assert get_feature_specification("hands").feature_dim == 126
    assert get_feature_specification("hands_pose").feature_dim == 225
    assert get_feature_specification("holistic").feature_dim == 1659


def test_representation_selection_preserves_time_and_selects_prefix() -> None:
    sequence = np.arange(2 * 1659, dtype=np.float32).reshape(2, 1659)
    input_mask = np.ones((2, 1659), dtype=np.float32)
    input_mask[0, 100:] = 0.0

    selected_features, selected_mask = select_feature_representation(
        sequence, "hands_pose", mask=input_mask
    )

    assert selected_features.shape == (2, 225)
    assert selected_mask.shape == (2, 225)
    # Valid portion matches original sequence
    assert np.array_equal(selected_features[:, :100], sequence[:, :100])
    # Masked-out portion is strictly zero-filled
    assert np.all(selected_features[0, 100:] == 0.0)
    # Frame 1 has no masked elements so it fully matches
    assert np.array_equal(selected_features[1], sequence[1, :225])
    assert np.array_equal(selected_mask, input_mask[:, :225])

    # Also verify default mask when mask=None
    feat_default, mask_default = select_feature_representation(sequence, "hands_pose")
    assert feat_default.shape == (2, 225)
    assert mask_default.shape == (2, 225)
    assert np.all(mask_default == 1.0)


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported feature representation"):
        get_feature_specification("unknown")