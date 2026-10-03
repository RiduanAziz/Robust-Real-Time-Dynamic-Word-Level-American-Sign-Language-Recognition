from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
