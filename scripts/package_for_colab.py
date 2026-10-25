#!/usr/bin/env python3
"""Package Project and Landmark Subsets for Google Colab.

Creates clean, compressed archives (.tar.gz) ready for instant upload to Google Drive:
1. asl_code.tar.gz (~3 MB): Source code, configs, scripts, and manifests.
2. landmarks_wlasl100.tar.gz (~350 MB): Landmark files for the 100 classes (2,038 samples).
3. landmarks_wlasl20.tar.gz (~70 MB): Landmark files for the 20 classes (456 samples).
4. landmarks_all.tar.gz: Optional full 21,083 landmarks archive.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path


def package_code(repo_root: Path, out_dir: Path) -> Path:
    out_file = out_dir / "asl_code.tar.gz"
    print(f"Creating code archive: {out_file}...")

    include_paths = [
        "src",
        "configs",
        "scripts",
        "data/manifests",
        "data/wlasl_class_list.txt",
        "data/word_embeddings.npz",
        "requirements.txt",
        "pyproject.toml",
    ]

    with tarfile.open(out_file, "w:gz") as tar:
        for rel_path in include_paths:
            full_path = repo_root / rel_path
            if full_path.exists():
                tar.add(full_path, arcname=rel_path)
                print(f"  Added {rel_path}")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Code archive created: {size_mb:.2f} MB")
    return out_file


def package_landmarks_subset(
    repo_root: Path,
    out_dir: Path,
    subset_name: str,
    class_names: set[str] | None = None,
) -> Path:
    out_file = out_dir / f"landmarks_{subset_name}.tar.gz"
    landmarks_dir = repo_root / "data" / "landmarks"

    with open(repo_root / "data" / "manifests" / "full_manifest.json", "r", encoding="utf-8") as f:
        full_manifest = json.load(f)

    if class_names is not None:
        selected_entries = [e for e in full_manifest if e["class_name"] in class_names]
    else:
        selected_entries = full_manifest

    sample_ids = {e["sample_id"] for e in selected_entries}
    print(f"\nCreating {subset_name} archive with {len(sample_ids)} landmark files...")

    with tarfile.open(out_file, "w:gz") as tar:
        for i, sid in enumerate(sample_ids):
            npz_path = landmarks_dir / f"{sid}.npz"
            if npz_path.exists():
                tar.add(npz_path, arcname=f"data/landmarks/{sid}.npz")
            if (i + 1) % 500 == 0 or (i + 1) == len(sample_ids):
                print(f"  Archived {i + 1}/{len(sample_ids)} files...")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Archived {subset_name} successfully: {size_mb:.2f} MB -> {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Package code and landmarks for Google Colab.")
    parser.add_argument("--subset", choices=["wlasl20", "wlasl100", "all", "code_only"], default="wlasl100", help="Subset to package")
    parser.add_argument("--out-dir", default="dist", help="Output directory for archives")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always package code
    package_code(repo_root, out_dir)

    if args.subset == "code_only":
        print("\nPackaging complete!")
        return

    # 2. Package chosen landmark subset
    if args.subset == "wlasl20":
        with open(repo_root / "configs" / "experiments" / "wlasl20_proposed.yaml", "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)
            w20_classes = set(cfg["dataset"]["labels"])
        package_landmarks_subset(repo_root, out_dir, "wlasl20", w20_classes)

    elif args.subset == "wlasl100":
        with open(repo_root / "configs" / "experiments" / "wlasl100_v1.json", "r", encoding="utf-8") as f:
            w100_classes = set(json.load(f)["vocabulary"])
        package_landmarks_subset(repo_root, out_dir, "wlasl100", w100_classes)

    elif args.subset == "all":
        package_landmarks_subset(repo_root, out_dir, "all", None)

    print(f"\nAll packages are ready in '{out_dir}/'!")
    print("Upload them to your Google Drive in a folder named 'ASL_Colab'.")


if __name__ == "__main__":
    main()
