"""Unit and integration tests for dataset integrity, leakage prevention, and vocabulary provenance."""
import json
from pathlib import Path
import pytest
import numpy as np

from sign_language.data.dataset import SignSample, build_dataset_from_manifest
from sign_language.data.vocabulary import CanonicalVocabulary, compute_vocabulary_hash
from sign_language.data.splitting import validate_signer_assignments, split_manifest_by_signer


def test_canonical_vocabulary_hashing_and_indexing() -> None:
    classes = ["book", "drink", "computer", "go", "before"]
    vocab = CanonicalVocabulary(classes=classes)

    assert len(vocab) == 5
    assert vocab[0] == "book"
    assert vocab["drink"] == 1
    assert isinstance(vocab.vocabulary_hash, str)
    assert len(vocab.vocabulary_hash) == 64  # SHA256 length

    # Compatibility check with identical classes
    vocab.validate_compatibility(classes)
    vocab.validate_compatibility(vocab.vocabulary_hash)

    # Incompatible class order or missing class
    with pytest.raises(ValueError, match="Class vocabulary mismatch"):
        vocab.validate_compatibility(["drink", "book", "computer", "go", "before"])

    with pytest.raises(ValueError, match="Vocabulary hash mismatch"):
        vocab.validate_compatibility("0" * 64)


def test_build_dataset_from_manifest_strict_mode(tmp_path: Path) -> None:
    # Create fake manifest with 2 entries
    manifest_file = tmp_path / "fake_manifest.json"
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        {"sample_id": "001", "class_name": "book", "signer_id": "S1", "num_frames": 30, "fps": 25.0},
        {"sample_id": "002", "class_name": "drink", "signer_id": "S2", "num_frames": 30, "fps": 25.0},
    ]
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    # Create only landmark 001, missing 002
    np.savez(landmarks_dir / "001.npz", landmarks=np.zeros((10, 42)))

    # Strict mode must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Strict mode violation"):
        build_dataset_from_manifest(
            manifest_path=manifest_file,
            landmarks_dir=landmarks_dir,
            missing_policy="strict",
        )

    # Diagnostic mode must collect exclusion and return diagnostics
    samples, diag = build_dataset_from_manifest(
        manifest_path=manifest_file,
        landmarks_dir=landmarks_dir,
        missing_policy="diagnostic",
        return_diagnostics=True,
    )
    assert len(samples) == 1
    assert diag["loaded_samples_count"] == 1
    assert diag["excluded_samples_count"] == 1
    assert diag["excluded_samples"][0]["sample_id"] == "002"


def test_signer_disjoint_leakage_verification() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    disjoint_dir = repo_root / "data" / "manifests" / "signer_disjoint"
    if not disjoint_dir.is_dir():
        pytest.skip("Signer-disjoint manifests not yet generated.")

    with open(disjoint_dir / "train_manifest.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(disjoint_dir / "validation_manifest.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open(disjoint_dir / "test_manifest.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    train_signers = set(str(s["signer_id"]) for s in train)
    val_signers = set(str(s["signer_id"]) for s in val)
    test_signers = set(str(s["signer_id"]) for s in test)

    # Assert ZERO signer overlap
    assert len(train_signers & val_signers) == 0, "Train and Val share signers!"
    assert len(train_signers & test_signers) == 0, "Train and Test share signers!"
    assert len(val_signers & test_signers) == 0, "Val and Test share signers!"
