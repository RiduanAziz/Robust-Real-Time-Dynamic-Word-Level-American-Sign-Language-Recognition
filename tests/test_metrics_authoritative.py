"""Authoritative unit tests for classification metrics and edge cases."""
import numpy as np
import pytest

from sign_language.training.metrics import (
    pure_accuracy_score,
    pure_top_k_accuracy_score,
    pure_confusion_matrix,
    pure_precision_recall_fscore_support,
    pure_balanced_accuracy_score,
    pure_expected_calibration_error,
)


def test_top1_accuracy():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    acc = pure_accuracy_score(y_true, y_pred)
    assert acc == pytest.approx(4.0 / 5.0)


def test_top_k_accuracy_ranking():
    # 3 samples, 5 classes
    y_true = np.array([0, 2, 4])
    # scores where true label is ranked 1st, 2nd, 3rd respectively
    y_score = np.array([
        [10.0, 5.0, 2.0, 1.0, 0.0],   # class 0 is top-1
        [1.0, 2.0, 8.0, 9.0, 0.0],    # class 2 is top-2 (9.0, 8.0)
        [1.0, 2.0, 3.0, 4.0, 5.0],    # class 4 is top-1
    ])
    assert pure_top_k_accuracy_score(y_true, y_score, k=1) == pytest.approx(2.0 / 3.0)
    assert pure_top_k_accuracy_score(y_true, y_score, k=2) == pytest.approx(3.0 / 3.0)
    assert pure_top_k_accuracy_score(y_true, y_score, k=5) == pytest.approx(1.0)


def test_top10_with_more_classes():
    # 20 classes, true labels in top 10
    np.random.seed(42)
    N = 20
    C = 20
    y_true = np.arange(N) % C
    y_score = np.zeros((N, C))
    for i in range(N):
        y_score[i, y_true[i]] = 5.0  # make sure it is in top 10
    assert pure_top_k_accuracy_score(y_true, y_score, k=10) == pytest.approx(1.0)


def test_confusion_matrix_and_support():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    labels = [0, 1, 2]
    cm = pure_confusion_matrix(y_true, y_pred, labels=labels)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1

    prec, rec, f1, sup = pure_precision_recall_fscore_support(y_true, y_pred, labels=labels)
    assert list(sup) == [2, 2, 1]
    assert rec[0] == pytest.approx(0.5)
    assert rec[1] == pytest.approx(1.0)
    assert rec[2] == pytest.approx(1.0)


def test_macro_and_weighted_f1():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1])
    labels = [0, 1, 2]
    prec, rec, f1, _ = pure_precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro")
    assert isinstance(prec, float) and isinstance(rec, float) and isinstance(f1, float)

    w_prec, w_rec, w_f1, _ = pure_precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted")
    assert isinstance(w_prec, float) and isinstance(w_rec, float) and isinstance(w_f1, float)


def test_balanced_accuracy():
    # Imbalanced classes: class 0 has 4 samples (all correct), class 1 has 1 sample (incorrect)
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    # Recall for class 0 = 1.0, recall for class 1 = 0.0 -> balanced acc = 0.5
    b_acc = pure_balanced_accuracy_score(y_true, y_pred)
    assert b_acc == pytest.approx(0.5)


def test_expected_calibration_error():
    # Perfect calibration
    probs = np.array([
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
    ])
    # 4 correct out of 5 -> 80% accuracy, conf = 0.8 -> ECE ~ 0
    targets = np.array([0, 0, 0, 0, 1])
    ece = pure_expected_calibration_error(probs, targets, num_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_edge_cases_absent_classes():
    # Vocabulary has classes [0, 1, 2, 3], but only [0, 1] appear in true
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    labels = [0, 1, 2, 3]
    prec, rec, f1, sup = pure_precision_recall_fscore_support(y_true, y_pred, labels=labels)
    assert sup[2] == 0
    assert sup[3] == 0
    assert prec[2] == 0.0
    assert rec[2] == 0.0
    assert f1[2] == 0.0


def test_edge_cases_tied_logits_and_non_finite():
    y_true = np.array([0, 1])
    # Tied logits
    y_score_tied = np.array([[2.0, 2.0], [1.0, 1.0]])
    top1 = pure_top_k_accuracy_score(y_true, y_score_tied, k=1)
    assert 0.0 <= top1 <= 1.0

    # Non-finite logits
    y_score_nan = np.array([[np.nan, 2.0], [np.inf, -np.inf]])
    top1_nan = pure_top_k_accuracy_score(y_true, y_score_nan, k=1)
    assert np.isfinite(top1_nan)


def test_edge_cases_few_samples_and_empty():
    assert pure_accuracy_score(np.array([]), np.array([])) == 0.0
    assert pure_top_k_accuracy_score(np.array([]), np.empty((0, 5))) == 0.0
    assert pure_balanced_accuracy_score(np.array([]), np.array([])) == 0.0
    assert pure_expected_calibration_error(np.empty((0, 2)), np.array([])) == 0.0

    # 2 samples, k=10 when C=3
    y_true = np.array([0, 1])
    y_score = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert pure_top_k_accuracy_score(y_true, y_score, k=10) == 1.0
