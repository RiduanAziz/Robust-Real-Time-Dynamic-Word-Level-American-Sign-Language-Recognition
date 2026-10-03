from __future__ import annotations

from collections import defaultdict

from .dataset import SignSample


def signer_aware_split(
    samples: list[SignSample],
    train_signers: set[str] | list[str] | None = None,
    val_signers: set[str] | list[str] | None = None,
    test_signers: set[str] | list[str] | None = None,
) -> tuple[list[SignSample], list[SignSample], list[SignSample]]:
    """Split a dataset by signer so samples from the same signer stay in only one split."""
    all_signers = sorted({sample.signer_id for sample in samples})

    if train_signers is None and val_signers is None and test_signers is None:
        train_count = max(1, int(len(all_signers) * 0.7))
        val_count = max(1, int(len(all_signers) * 0.15))
        test_count = len(all_signers) - train_count - val_count

        train_signers = set(all_signers[:train_count])
        val_signers = set(all_signers[train_count : train_count + val_count])
        test_signers = set(all_signers[train_count + val_count :])

    train_signers = set(train_signers or [])
    val_signers = set(val_signers or [])
    test_signers = set(test_signers or [])

    missing = set(all_signers) - (train_signers | val_signers | test_signers)
    if missing:
        raise ValueError(f"Signers {sorted(missing)} were not assigned to any split.")

    train = [sample for sample in samples if sample.signer_id in train_signers]
    val = [sample for sample in samples if sample.signer_id in val_signers]
    test = [sample for sample in samples if sample.signer_id in test_signers]
    return train, val, test
