"""Canonical class vocabulary and checkpoint provenance management."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


def compute_vocabulary_hash(classes: Sequence[str]) -> str:
    """Compute a deterministic SHA256 hash for an ordered class list."""
    payload = json.dumps(list(classes), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CanonicalVocabulary:
    """Class vocabulary ensuring deterministic indexing and checkpoint provenance."""
    classes: list[str]
    label_to_index: dict[str, int] = field(init=False)
    index_to_label: dict[int, str] = field(init=False)
    vocabulary_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("Vocabulary cannot be empty.")
        if len(self.classes) != len(set(self.classes)):
            raise ValueError("Vocabulary contains duplicate class labels.")

        self.label_to_index = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.index_to_label = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
        self.vocabulary_hash = compute_vocabulary_hash(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, item: int | str) -> str | int:
        if isinstance(item, int):
            return self.index_to_label[item]
        if isinstance(item, str):
            if item not in self.label_to_index:
                raise KeyError(f"Label '{item}' is not in vocabulary.")
            return self.label_to_index[item]
        raise TypeError(f"Index must be int or str, got {type(item)}")

    def validate_compatibility(self, checkpoint_vocab_or_hash: Sequence[str] | str) -> None:
        """Verify that a checkpoint vocabulary or hash matches this canonical vocabulary."""
        if isinstance(checkpoint_vocab_or_hash, str):
            if checkpoint_vocab_or_hash != self.vocabulary_hash:
                raise ValueError(
                    f"Vocabulary hash mismatch! Current vocabulary hash: {self.vocabulary_hash[:12]}, "
                    f"checkpoint vocabulary hash: {checkpoint_vocab_or_hash[:12]}. "
                    "Refusing inference or evaluation with misaligned class labels."
                )
        elif isinstance(checkpoint_vocab_or_hash, (list, tuple)):
            ckpt_classes = list(checkpoint_vocab_or_hash)
            if ckpt_classes != self.classes:
                raise ValueError(
                    f"Class vocabulary mismatch! Model checkpoint has {len(ckpt_classes)} classes, "
                    f"expected {len(self.classes)} classes with matching order. "
                    f"First difference: "
                    f"{[(i, a, b) for i, (a, b) in enumerate(zip(self.classes, ckpt_classes)) if a != b][:3]}"
                )

    def to_metadata(self) -> dict[str, Any]:
        """Produce checkpoint metadata payload."""
        return {
            "classes": list(self.classes),
            "class_count": len(self.classes),
            "vocabulary_hash": self.vocabulary_hash,
        }

    @classmethod
    def from_file(cls, path: str | Path) -> CanonicalVocabulary:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {p}")
        if p.suffix in (".yaml", ".yml"):
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                labels = data.get("dataset", {}).get("labels", [])
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    labels = data
                elif isinstance(data, dict):
                    labels = data.get("classes") or data.get("labels") or data.get("vocabulary") or []
        else:
            # Plain text file with one class per line or TSV (e.g. wlasl_class_list.txt)
            labels = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        labels.append(parts[1].strip())
                    elif line.strip():
                        labels.append(line.strip())
        return cls(classes=labels)
