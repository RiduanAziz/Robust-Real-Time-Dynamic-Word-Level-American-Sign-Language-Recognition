"""Dataset and Manifest Integrity Validator CLI (SignFlow).

Supports:
  Protocol A: Official / reference split (with documented signer overlap)
  Protocol B: Signer-disjoint split (with 0 signer overlap enforcement)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

# Force unbuffered UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

def validate_dataset(
    repo_root: Path,
    protocol: str = "official",
    strict: bool = False,
    check_features: bool = True,
    max_feature_samples: int = 100,
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "protocol": protocol,
        "strict": strict,
        "passed": True,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    if protocol == "official":
        manifest_dir = repo_root / "data" / "manifests"
    elif protocol in ("signer_disjoint", "disjoint", "protocol_b"):
        manifest_dir = repo_root / "data" / "manifests" / "signer_disjoint"
    else:
        report["passed"] = False
        report["errors"].append(f"Unknown protocol: {protocol}")
        return False, report

    train_path = manifest_dir / "train_manifest.json"
    val_path = manifest_dir / "validation_manifest.json"
    test_path = manifest_dir / "test_manifest.json"

    for p in (train_path, val_path, test_path):
        if not p.is_file():
            report["passed"] = False
            report["errors"].append(f"Missing required manifest: {p}")

    if not report["passed"]:
        return False, report

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    splits = {"train": train_data, "validation": val_data, "test": test_data}
    split_stats = {}
    all_sample_ids = set()

    for name, data in splits.items():
        sample_ids = [d["sample_id"] for d in data]
        signer_ids = [str(d.get("signer_id", "UNKNOWN")) for d in data]
        class_names = [d["class_name"] for d in data]

        unique_samples = set(sample_ids)
        unique_signers = set(signer_ids)
        unique_classes = set(class_names)

        dup_samples = len(data) - len(unique_samples)
        if dup_samples > 0:
            msg = f"{name} split has {dup_samples} duplicate sample IDs!"
            report["errors"].append(msg)
            report["passed"] = False

        split_stats[name] = {
            "count": len(data),
            "unique_samples": len(unique_samples),
            "unique_signers": len(unique_signers),
            "unique_classes": len(unique_classes),
            "signers": sorted(list(unique_signers)),
            "sample_ids": unique_samples,
        }

    report["summary"]["splits"] = {
        name: {
            "samples": s["count"],
            "unique_signers": s["unique_signers"],
            "unique_classes": s["unique_classes"],
        }
        for name, s in split_stats.items()
    }

    # Pairwise sample ID overlap
    train_val_samples = split_stats["train"]["sample_ids"] & split_stats["validation"]["sample_ids"]
    train_test_samples = split_stats["train"]["sample_ids"] & split_stats["test"]["sample_ids"]
    val_test_samples = split_stats["validation"]["sample_ids"] & split_stats["test"]["sample_ids"]

    sample_overlap_count = len(train_val_samples) + len(train_test_samples) + len(val_test_samples)
    if sample_overlap_count > 0:
        report["passed"] = False
        report["errors"].append(
            f"Sample overlap detected between splits: train-val={len(train_val_samples)}, "
            f"train-test={len(train_test_samples)}, val-test={len(val_test_samples)}"
        )
    report["summary"]["sample_id_leakage"] = sample_overlap_count

    # Pairwise signer overlap
    train_signers = set(split_stats["train"]["signers"])
    val_signers = set(split_stats["validation"]["signers"])
    test_signers = set(split_stats["test"]["signers"])

    tv_signers = train_signers & val_signers
    tt_signers = train_signers & test_signers
    vt_signers = val_signers & test_signers

    report["summary"]["signer_overlap"] = {
        "train_validation": len(tv_signers),
        "train_test": len(tt_signers),
        "validation_test": len(vt_signers),
    }

    if protocol in ("signer_disjoint", "disjoint", "protocol_b"):
        if len(tv_signers) > 0 or len(tt_signers) > 0 or len(vt_signers) > 0:
            report["passed"] = False
            report["errors"].append(
                f"Protocol B signer overlap violation! Shared signers: "
                f"train-val={len(tv_signers)}, train-test={len(tt_signers)}, val-test={len(vt_signers)}"
            )
    else:
        # Protocol A has known signer overlap, document as warning
        report["warnings"].append(
            f"Protocol A (Official WLASL) has documented signer overlap: "
            f"train-test={len(tt_signers)} shared signers. Not signer-independent."
        )

    # File existence check on data/landmarks/
    landmarks_dir = repo_root / "data" / "landmarks"
    missing_landmarks: dict[str, list[str]] = {}
    total_missing = 0

    for name, data in splits.items():
        missing_in_split = []
        for d in data:
            sid = d["sample_id"]
            npz_file = landmarks_dir / f"{sid}.npz"
            if not npz_file.is_file():
                missing_in_split.append(sid)
        if missing_in_split:
            missing_landmarks[name] = missing_in_split
            total_missing += len(missing_in_split)

    report["summary"]["missing_landmark_files"] = total_missing
    if total_missing > 0:
        msg = f"Found {total_missing} missing landmark .npz files in data/landmarks/"
        if strict:
            report["passed"] = False
            report["errors"].append(msg)
        else:
            report["warnings"].append(msg)

    # Feature inspection (NaNs, infinities, shapes, masks)
    if check_features and landmarks_dir.is_dir():
        checked_count = 0
        corrupted_files = []
        nan_files = []
        inf_files = []

        # Sample across train, val, test
        sample_subset = []
        for name, data in splits.items():
            sample_subset.extend(data[:max_feature_samples // 3])

        for item in sample_subset:
            sid = item["sample_id"]
            npz_path = landmarks_dir / f"{sid}.npz"
            if not npz_path.is_file():
                continue
            checked_count += 1
            try:
                with np.load(npz_path, allow_pickle=True) as npz:
                    # Look for feature array or landmarks
                    arr = None
                    for key in ("landmarks", "features", "arr_0", "data"):
                        if key in npz:
                            arr = npz[key]
                            break
                    if arr is None:
                        # Inspect first key
                        first_key = list(npz.keys())[0]
                        arr = npz[first_key]

                    if not isinstance(arr, np.ndarray):
                        corrupted_files.append((sid, "Not a numpy array"))
                    elif np.isnan(arr).any():
                        nan_files.append(sid)
                    elif np.isinf(arr).any():
                        inf_files.append(sid)
            except Exception as exc:
                corrupted_files.append((sid, str(exc)))

        report["summary"]["features_checked"] = checked_count
        report["summary"]["nan_count"] = len(nan_files)
        report["summary"]["inf_count"] = len(inf_files)
        report["summary"]["corrupted_count"] = len(corrupted_files)

        if nan_files:
            msg = f"{len(nan_files)} feature arrays contain NaN values."
            if strict:
                report["passed"] = False
                report["errors"].append(msg)
            else:
                report["warnings"].append(msg)

        if inf_files:
            msg = f"{len(inf_files)} feature arrays contain Inf values."
            if strict:
                report["passed"] = False
                report["errors"].append(msg)
            else:
                report["warnings"].append(msg)

        if corrupted_files:
            msg = f"{len(corrupted_files)} feature files are unreadable/corrupted."
            report["passed"] = False
            report["errors"].append(msg)

    return report["passed"], report


def main() -> None:
    parser = argparse.ArgumentParser(description="SignFlow Dataset & Manifest Integrity Validator")
    parser.add_argument("--protocol", choices=["official", "signer_disjoint"], default="official", help="Split protocol to validate")
    parser.add_argument("--strict", action="store_true", help="Fail with non-zero exit code on any integrity warning")
    parser.add_argument("--no-features", action="store_true", help="Skip deep feature array inspection")
    parser.add_argument("--max-feature-samples", type=int, default=150, help="Max feature files to deeply inspect")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON validation report")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    passed, report = validate_dataset(
        repo_root=repo_root,
        protocol=args.protocol,
        strict=args.strict,
        check_features=not args.no_features,
        max_feature_samples=args.max_feature_samples,
    )

    out_path = Path(args.output) if args.output else repo_root / "results" / f"dataset_validation_{args.protocol}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Terminal summary
    print("=" * 60)
    print(f"SIGNFLOW DATASET INTEGRITY VALIDATOR - Protocol: {args.protocol.upper()}")
    print("=" * 60)
    status_str = "PASSED" if passed else "FAILED"
    print(f"Status: {status_str} (strict={args.strict})")
    print(f"Report written to: {out_path}")
    print("\nSplit Statistics:")
    for split_name, stats in report["summary"].get("splits", {}).items():
        print(f"  - {split_name.capitalize():<12}: {stats['samples']} samples, {stats['unique_signers']} signers, {stats['unique_classes']} classes")

    overlap = report["summary"].get("signer_overlap", {})
    print("\nSigner Overlap:")
    print(f"  - Train & Val : {overlap.get('train_validation', 0)}")
    print(f"  - Train & Test: {overlap.get('train_test', 0)}")
    print(f"  - Val & Test  : {overlap.get('validation_test', 0)}")

    if report["errors"]:
        print("\nErrors:")
        for err in report["errors"]:
            print(f"  [ERROR] {err}")

    if report["warnings"]:
        print("\nWarnings:")
        for warn in report["warnings"]:
            print(f"  [WARNING] {warn}")

    print("=" * 60)
    if not passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
