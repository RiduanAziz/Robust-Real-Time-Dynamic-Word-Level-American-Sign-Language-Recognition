from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


@dataclass
class SignSample:
    sample_id: str
    label: str
    signer_id: str
    landmarks: np.ndarray | str | Path
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        return self.label


class SignLanguageDataset(Dataset):
    """Dataset abstraction for sign samples with landmark sequences."""

    def __init__(
        self,
        samples: list[SignSample],
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        labels = sorted({sample.label for sample in self.samples})
        self.label_to_index = label_to_index or {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        mask = None
        if isinstance(sample.landmarks, (str, Path)):
            with np.load(str(sample.landmarks)) as loaded:
                landmarks = loaded["landmarks"].astype(np.float32)
                if "mask" in loaded:
                    mask = loaded["mask"].astype(np.float32)
        else:
            landmarks = np.asarray(sample.landmarks, dtype=np.float32)
            
        label_index = self.label_to_index[sample.label]
        return {
            "sample_id": sample.sample_id,
            "signer_id": sample.signer_id,
            "label": label_index,
            "label_name": sample.label,
            "landmarks": landmarks,
            "mask": mask,
            "metadata": sample.metadata,
        }

def build_dataset_from_manifest(
    manifest_path: str | Path,
    landmarks_dir: str | Path,
    allowed_classes: list[str] | None = None,
    missing_policy: str = "strict",
    return_diagnostics: bool = False,
) -> list[SignSample] | tuple[list[SignSample], dict[str, Any]]:
    """Build a dataset from a manifest file, preventing silent sample loss.

    Args:
        manifest_path: Path to JSON manifest.
        landmarks_dir: Root directory containing extracted landmark .npz archives.
        allowed_classes: Optional class name whitelist filter.
        missing_policy: "strict" (raise FileNotFoundError on missing file),
                        "diagnostic" (record exclusions and log warning),
                        "ignore" (silently skip, legacy exploratory mode).
        return_diagnostics: If True, return (samples, diagnostics_dict).
    """
    import json
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    landmarks_root = Path(landmarks_dir)
    samples: list[SignSample] = []
    excluded_samples: list[dict[str, Any]] = []
    matching_manifest_count = 0

    allowed_set = set(allowed_classes) if allowed_classes is not None else None

    for entry in manifest:
        if allowed_set is not None and entry["class_name"] not in allowed_set:
            continue

        matching_manifest_count += 1
        npz_path = landmarks_root / f"{entry['sample_id']}.npz"
        if not npz_path.exists():
            reason = f"Missing landmark archive: {npz_path}"
            excluded_samples.append({
                "sample_id": entry["sample_id"],
                "class_name": entry["class_name"],
                "signer_id": entry["signer_id"],
                "reason": reason,
            })
            if missing_policy == "strict":
                raise FileNotFoundError(
                    f"Strict mode violation: Required landmark file does not exist: {npz_path} "
                    f"for sample_id={entry['sample_id']} (class={entry['class_name']}). "
                    "Refusing to silently drop dataset samples."
                )
            elif missing_policy == "diagnostic":
                logger.warning("Sample excluded [%s]: %s", entry["sample_id"], reason)
            continue

        samples.append(
            SignSample(
                sample_id=entry["sample_id"],
                label=entry["class_name"],
                signer_id=entry["signer_id"],
                landmarks=npz_path,
                metadata={
                    "num_frames": entry["num_frames"],
                    "fps": entry["fps"],
                    "split": entry.get("split"),
                },
            )
        )

    diagnostics = {
        "manifest_path": str(manifest_path),
        "landmarks_dir": str(landmarks_dir),
        "total_manifest_samples": len(manifest),
        "matching_class_samples": matching_manifest_count,
        "loaded_samples_count": len(samples),
        "excluded_samples_count": len(excluded_samples),
        "excluded_samples": excluded_samples,
    }

    if return_diagnostics:
        return samples, diagnostics
    return samples


def build_synthetic_dataset(
    labels: list[str] | None = None,
    signers: list[str] | None = None,
    samples_per_class: int = 6,
    sequence_length: int = 16,
    feature_dim: int = 42,
) -> list[SignSample]:
    """Create a small synthetic dataset for testing model training and splitting."""
    labels = labels or ["HELLO", "THANKS", "PLEASE", "YES", "NO"]
    signers = signers or [f"S{i:03d}" for i in range(1, 21)]

    samples: list[SignSample] = []
    for label_idx, label in enumerate(labels):
        for signer_idx, signer in enumerate(signers):
            for session_idx in range(1, samples_per_class + 1):
                base = (label_idx + 1) * 3.0 + signer_idx * 0.7
                sequence = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
                landmarks = np.zeros((sequence_length, feature_dim), dtype=np.float32)
                for t in range(sequence_length):
                    landmarks[t] = base + sequence[t] + np.linspace(0.0, 0.2, feature_dim, dtype=np.float32)
                    landmarks[t] += (signer_idx % 3) * 0.05
                sample = SignSample(
                    sample_id=f"{signer}_{label}_{session_idx:02d}",
                    label=label,
                    signer_id=signer,
                    session_id=f"session_{session_idx:02d}",
                    landmarks=landmarks,
                    metadata={
                        "lighting": "normal",
                        "background": "indoor",
                        "speed": "normal",
                    },
                )
                samples.append(sample)
    return samples
