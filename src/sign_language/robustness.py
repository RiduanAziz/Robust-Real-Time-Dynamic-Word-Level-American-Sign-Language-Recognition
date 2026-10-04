from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sign_language.evaluation.metrics import classification_metrics


def _validate_severity(severity: float) -> float:
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError("noise severity must be between 0 and 1")
    return value


def coordinate_jitter(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return array + rng.normal(0.0, value, size=array.shape).astype(np.float32)


def translation_noise(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    rng = np.random.default_rng(seed)
    offset = rng.normal(0.0, value, size=(1, array.shape[-1])).astype(np.float32)
    return array + offset


def scale_noise(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(1.0 - value, 1.0 + value))
    return array * scale


def landmark_dropout(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32).copy()
    if array.ndim != 2:
        raise ValueError("landmark dropout expects a [T, F] sequence")
    rng = np.random.default_rng(seed)
    landmark_count = array.shape[1] // 3 if array.shape[1] % 3 == 0 else array.shape[1]
    dropped = rng.random(landmark_count) < value
    if array.shape[1] % 3 == 0:
        array[:, dropped.repeat(3)] = 0.0
    else:
        array[:, dropped] = 0.0
    return array


def apply_spatial_noise(
    sequence: np.ndarray,
    noise_type: str,
    severity: float,
    seed: int = 42,
) -> np.ndarray:
    operators: dict[str, Callable[..., np.ndarray]] = {
        "coordinate_jitter": coordinate_jitter,
        "translation": translation_noise,
        "scale": scale_noise,
        "landmark_dropout": landmark_dropout,
    }
    try:
        operator = operators[noise_type]
    except KeyError as error:
        raise ValueError(f"Unsupported spatial noise type: {noise_type}") from error
    return operator(sequence, severity, seed)


def frame_drop(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape[0] == 0:
        return array.copy()
    rng = np.random.default_rng(seed)
    keep = rng.random(array.shape[0]) >= value
    keep[0] = True
    return array[keep]


def frame_duplicate(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape[0] == 0 or value == 0.0:
        return array.copy()
    rng = np.random.default_rng(seed)
    duplicate_count = round(array.shape[0] * value)
    indices = rng.integers(0, array.shape[0], size=duplicate_count)
    return np.concatenate([array, array[indices]], axis=0)


def sequence_truncate(sequence: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    value = _validate_severity(severity)
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape[0] == 0:
        return array.copy()
    keep_count = max(1, round(array.shape[0] * (1.0 - value)))
    return array[:keep_count]


def apply_temporal_noise(
    sequence: np.ndarray,
    noise_type: str,
    severity: float,
    seed: int = 42,
) -> np.ndarray:
    operators = {
        "frame_drop": frame_drop,
        "frame_duplicate": frame_duplicate,
        "sequence_truncate": sequence_truncate,
    }
    try:
        return operators[noise_type](sequence, severity, seed)
    except KeyError as error:
        raise ValueError(f"Unsupported temporal noise type: {noise_type}") from error


def apply_combined_noise(
    sequence: np.ndarray,
    spatial_type: str,
    temporal_type: str,
    spatial_severity: float,
    temporal_severity: float,
    seed: int = 42,
) -> np.ndarray:
    spatial = apply_spatial_noise(sequence, spatial_type, spatial_severity, seed)
    return apply_temporal_noise(spatial, temporal_type, temporal_severity, seed + 1)


def recognition_robustness_report(
    targets: list[int],
    clean_predictions: list[int],
    noisy_predictions: list[int],
    class_names: list[str] | None = None,
    noise_type: str = "unknown",
    severity: float = 0.0,
) -> dict[str, Any]:
    """Report recognition degradation; signal error is intentionally secondary."""
    clean = classification_metrics(targets, clean_predictions, class_names)
    noisy = classification_metrics(targets, noisy_predictions, class_names)
    return {
        "noise_type": noise_type,
        "severity": float(severity),
        "clean": clean,
        "noisy": noisy,
        "accuracy_degradation": float(clean["accuracy"] - noisy["accuracy"]),
        "macro_f1_degradation": float(clean["macro_f1"] - noisy["macro_f1"]),
    }


@dataclass(frozen=True)
class NoiseScenario:
    name: str
    severity: float
    clean_signal: np.ndarray
    noisy_signal: np.ndarray


def evaluate_robustness(scenarios: list[NoiseScenario]) -> dict[str, Any]:
    """Evaluate a list of controlled noise scenarios using absolute error summaries."""
    entries: list[dict[str, Any]] = []
    for scenario in scenarios:
        clean = np.asarray(scenario.clean_signal, dtype=np.float32)
        noisy = np.asarray(scenario.noisy_signal, dtype=np.float32)
        if clean.shape != noisy.shape:
            raise ValueError(f"Scenario '{scenario.name}' has mismatched clean/noisy shapes.")

        diff = np.abs(noisy - clean)
        entries.append(
            {
                "name": scenario.name,
                "severity": float(scenario.severity),
                "mean_absolute_error": float(diff.mean()),
                "max_absolute_error": float(diff.max()),
                "signal_length": int(clean.size),
            }
        )

    return {
        "scenario_count": len(entries),
        "scenarios": entries,
        "average_mean_absolute_error": float(np.mean([item["mean_absolute_error"] for item in entries])) if entries else 0.0,
    }
