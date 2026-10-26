"""Automated unit and integration tests for SignFlow accuracy optimization suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sign_language.training.metrics import (
    pure_accuracy_score,
    pure_top_k_accuracy_score,
    pure_confusion_matrix,
    pure_precision_recall_fscore_support,
    pure_balanced_accuracy_score,
)
from sign_language.landmarks.missing_data import interpolate_short_gaps
from sign_language.models.tcn import TCNClassifier
from sign_language.models.reliability_fusion import ReliabilityAwareIsolatedFusionClassifier
from sign_language.data.vocabulary import CanonicalVocabulary


def test_top1_and_top5_metric_correctness():
    """Verify that pure Top-1 and Top-5 accuracy calculations are mathematically exact."""
    y_true = np.array([0, 1, 2, 3, 4])
    # 5 classes: class scores
    y_score = np.array([
        [0.9, 0.05, 0.02, 0.02, 0.01],  # True 0 in top-1, top-5
        [0.4, 0.3, 0.1, 0.1, 0.1],      # True 1: rank 2 (in top-5, not top-1)
        [0.0, 0.0, 0.8, 0.1, 0.1],      # True 2: in top-1, top-5
        [0.2, 0.2, 0.2, 0.2, 0.2],      # True 3: tied, in top-5
        [0.5, 0.3, 0.1, 0.05, 0.05],    # True 4: rank 5 (in top-5, not top-1)
    ])
    
    top1 = pure_accuracy_score(y_true, np.argmax(y_score, axis=1))
    top5 = pure_top_k_accuracy_score(y_true, y_score, k=5)
    top2 = pure_top_k_accuracy_score(y_true, y_score, k=2)
    
    # y_pred = [0, 0, 2, 0, 0] -> correct: index 0 and 2 (2/5 = 0.40)
    assert top1 == pytest.approx(0.40)
    # All 5 true labels are within the top 5
    assert top5 == pytest.approx(1.00)
    # In top 2: sample 0 (yes), sample 1 (yes), sample 2 (yes), sample 3 (yes), sample 4 (no) -> 4/5 = 0.80
    assert top2 == pytest.approx(0.80)


def test_confusion_matrix_and_balanced_accuracy():
    """Verify confusion matrix and balanced accuracy computation."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 1])
    
    cm = pure_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2  # class 0: 2/2 correct
    assert cm[1, 1] == 1  # class 1: 1/2 correct (1 predicted as 0)
    assert cm[2, 2] == 1  # class 2: 1/2 correct (1 predicted as 1)
    
    # Recalls: [1.0, 0.5, 0.5] -> Mean = 2.0 / 3 = 0.6667
    bal_acc = pure_balanced_accuracy_score(y_true, y_pred)
    assert bal_acc == pytest.approx(2.0 / 3.0)


def test_missing_data_short_gap_interpolation():
    """Verify that gaps <= 3 frames are interpolated and gaps > 3 frames remain zeroed."""
    seq = np.zeros((7, 3), dtype=np.float32)
    mask = np.zeros(7, dtype=bool)
    
    # Frame 0 and Frame 3 are observed (gap of 2 frames in between: frame 1 and 2)
    seq[0] = [1.0, 2.0, 3.0]
    mask[0] = True
    seq[3] = [4.0, 5.0, 6.0]
    mask[3] = True
    
    interp_seq, interp_mask = interpolate_short_gaps(seq, mask, max_gap=2)
    
    # Frame 1 and 2 should be linearly interpolated
    assert interp_mask[1].all() and interp_mask[2].all()
    np.testing.assert_allclose(interp_seq[1], [2.0, 3.0, 4.0], rtol=1e-5)
    np.testing.assert_allclose(interp_seq[2], [3.0, 4.0, 5.0], rtol=1e-5)
    
    # Frame 4, 5, 6 should remain unobserved
    assert not interp_mask[4].any() and not interp_mask[5].any() and not interp_mask[6].any()


def test_tcn_and_reliability_fusion_forward_pass():
    """Verify forward pass, output shape, and finite logits for TCN and Reliability-Aware models."""
    batch_size = 2
    seq_len = 64
    input_dim = 225
    num_classes = 100
    
    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32)
    mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    
    # 1. TCN Classifier
    tcn = TCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=[64, 128, 256],
        kernel_size=3,
        dropout=0.2,
    )
    tcn_out = tcn(x, mask=mask)
    assert tcn_out.shape == (batch_size, num_classes)
    assert torch.isfinite(tcn_out).all()
    
    # 2. Reliability-Aware Fusion Classifier
    rel_model = ReliabilityAwareIsolatedFusionClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
    )
    rel_out = rel_model(x, mask=mask)
    assert rel_out.shape == (batch_size, num_classes)
    assert torch.isfinite(rel_out).all()


def test_canonical_vocabulary_immutability():
    """Verify CanonicalVocabulary hash determinism and rejection of unknown labels."""
    classes = ["book", "drink", "computer", "go", "chair"]
    vocab = CanonicalVocabulary(classes)
    
    assert vocab.num_classes == 5
    assert vocab.get_index("book") == 0
    assert vocab.get_label(0) == "book"
    assert vocab.get_index("unknown_sign") is None
    
    # Hash determinism
    vocab2 = CanonicalVocabulary(classes)
    assert vocab.vocabulary_hash == vocab2.vocabulary_hash
