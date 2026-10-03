from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from .manifest import DatasetManifestEntry


def split_manifest_by_signer(
    manifest: Iterable[DatasetManifestEntry],
    train_signers: Sequence[str] | set[str] | None = None,
    val_signers: Sequence[str] | set[str] | None = None,
    test_signers: Sequence[str] | set[str] | None = None,
) -> tuple[list[DatasetManifestEntry], list[DatasetManifestEntry], list[DatasetManifestEntry]]:
    entries = list(manifest)
    if not entries:
        return [], [], []

    all_signers = sorted({item.signer_id for item in entries})
    train_signers = list(train_signers) if train_signers is not None else []
    val_signers = list(val_signers) if val_signers is not None else []
    test_signers = list(test_signers) if test_signers is not None else []

    if not train_signers and not val_signers and not test_signers:
        if len(all_signers) < 3:
            raise ValueError("At least three unique signers are required for a train/val/test split")
        train_signers = all_signers[:-2]
        val_signers = [all_signers[-2]]
        test_signers = [all_signers[-1]]

    train = [replace(item, split="train") for item in entries if item.signer_id in train_signers]
    val = [replace(item, split="validation") for item in entries if item.signer_id in val_signers]
    test = [replace(item, split="test") for item in entries if item.signer_id in test_signers]

    if not train or not val or not test:
        raise ValueError("Train, validation, and test splits must all contain at least one sample")

    return train, val, test
