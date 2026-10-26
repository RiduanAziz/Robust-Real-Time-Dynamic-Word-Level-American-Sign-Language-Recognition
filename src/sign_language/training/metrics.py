"""Pure-NumPy implementation of classification metrics with zero external C-extension dependencies."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np


def pure_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard categorical accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def pure_top_k_accuracy_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 5,
    labels: Optional[List[int]] = None,
) -> float:
    """Compute Top-K accuracy from scores/logits matrix."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0 or y_score.size == 0:
        return 0.0
    if y_score.ndim != 2:
        return 0.0
    if not np.isfinite(y_score).all():
        y_score = np.nan_to_num(y_score, nan=-1e9, posinf=1e9, neginf=-1e9)

    k = min(k, y_score.shape[1])
    hits = 0
    for idx, true_label in enumerate(y_true):
        # Top-k largest elements
        top_k_indices = np.argsort(y_score[idx])[-k:]
        if labels is not None:
            top_k_classes = [labels[j] for j in top_k_indices]
            if true_label in top_k_classes:
                hits += 1
        else:
            if true_label in top_k_indices:
                hits += 1
    return float(hits / len(y_true))


def pure_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    label_to_idx = {l: idx for idx, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def pure_precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    average: Optional[str] = None,
    zero_division: float = 0.0,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float], Optional[np.ndarray]]:
    """Compute precision, recall, F1, and support per-class and aggregated."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm = pure_confusion_matrix(y_true, y_pred, labels)
    n_classes = len(labels)
    tp = np.diag(cm).astype(np.float64)
    fp = (cm.sum(axis=0) - tp).astype(np.float64)
    fn = (cm.sum(axis=1) - tp).astype(np.float64)
    support = cm.sum(axis=1).astype(np.int64)

    prec = np.zeros(n_classes, dtype=np.float64)
    rec = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)

    for i in range(n_classes):
        denom_p = tp[i] + fp[i]
        prec[i] = tp[i] / denom_p if denom_p > 0 else zero_division
        denom_r = tp[i] + fn[i]
        rec[i] = tp[i] / denom_r if denom_r > 0 else zero_division
        denom_f = prec[i] + rec[i]
        f1[i] = 2.0 * (prec[i] * rec[i]) / denom_f if denom_f > 0 else zero_division

    if average is None:
        return prec, rec, f1, support
    elif average == "macro":
        # Average over all classes in the vocabulary
        return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1)), None
    elif average == "weighted":
        total_sup = float(np.sum(support))
        if total_sup > 0:
            weights = support / total_sup
            return float(np.sum(prec * weights)), float(np.sum(rec * weights)), float(np.sum(f1 * weights)), None
        else:
            return 0.0, 0.0, 0.0, None
    else:
        raise ValueError(f"Unsupported average: {average}")


def pure_balanced_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute balanced accuracy (macro-recall over classes present in ground truth)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(list(set(y_true)))
    if len(labels) == 0:
        return 0.0
    _, rec, _, _ = pure_precision_recall_fscore_support(y_true, y_pred, labels=labels)
    return float(np.mean(rec))


def pure_expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) from probabilities and ground-truth targets."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(probabilities) == 0 or len(targets) == 0:
        return 0.0
    if probabilities.ndim != 2:
        return 0.0
    if not np.isfinite(probabilities).all():
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

