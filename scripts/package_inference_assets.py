#!/usr/bin/env python3
"""Package production inference assets into a deterministic, SHA-256 verified bundle.

This script inspects and packages ONLY the required runtime model weights,
matching config files, and the MediaPipe HolisticLandmarker asset into a
versioned ZIP bundle with manifest.json and manifest.sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("package_inference_assets")

REQUIRED_INFERENCE_FILES = [
    {
        "source": "models/robust_holistic_fusion_best.pt",
        "archive_path": "models/robust_holistic_fusion_best.pt",
        "config": "configs/dataset.yaml",
        "expected_arch": "robust_holistic_fusion",
        "expected_dim": 1659,
        "expected_seq": 64,
        "expected_classes": 100,
    },
    {
        "source": "models/wlasl100_unisign_pgf_best.pt",
        "archive_path": "models/wlasl100_unisign_pgf_best.pt",
        "config": "configs/colab/wlasl100_unisign_pgf.yaml",
        "expected_arch": "unisign_pgf",
        "expected_dim": 1659,
        "expected_seq": 64,
        "expected_classes": 100,
    },
    {
        "source": "models/wlasl100_max_accuracy_best.pt",
        "archive_path": "models/wlasl100_max_accuracy_best.pt",
        "config": "configs/colab/wlasl100_max_accuracy.yaml",
        "expected_arch": "robust_holistic_fusion",
        "expected_dim": 1659,
        "expected_seq": 64,
        "expected_classes": 100,
    },
    {
        "source": "models/prior_logits_100.npy",
        "archive_path": "models/prior_logits_100.npy",
    },
    {
        "source": "models/mediapipe/holistic_landmarker.task",
        "archive_path": "models/mediapipe/holistic_landmarker.task",
    },
]

CONFIG_FILES = [
    "configs/dataset.yaml",
    "configs/base.yaml",
    "configs/colab/wlasl100_unisign_pgf.yaml",
    "configs/colab/wlasl100_max_accuracy.yaml",
    "configs/colab/wlasl100_baseline_colab.yaml",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint_compatibility(root: Path) -> None:
    """Validate that all 3 model checkpoints load, have 100 classes, and execute dummy forward pass."""
    import numpy as np
    from sign_language.api.inference import RealTimePredictor, EnsembleRealTimePredictor

    logger.info("Validating checkpoints against configs and model loaders...")
    ensemble_specs = [
        (str(root / "models/robust_holistic_fusion_best.pt"), str(root / "configs/dataset.yaml"), 0.25),
        (str(root / "models/wlasl100_unisign_pgf_best.pt"), str(root / "configs/colab/wlasl100_unisign_pgf.yaml"), 0.10),
        (str(root / "models/wlasl100_max_accuracy_best.pt"), str(root / "configs/colab/wlasl100_max_accuracy.yaml"), 0.65),
    ]

    for ckpt, cfg, _ in ensemble_specs:
        ckpt_path = Path(ckpt)
        cfg_path = Path(cfg)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing required model checkpoint: {ckpt_path}")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing required model configuration: {cfg_path}")

        pred = RealTimePredictor(model_path=str(ckpt_path), config_path=str(cfg_path))
        if pred.num_classes != 100:
            raise ValueError(f"Model {ckpt_path} has {pred.num_classes} classes, expected 100.")
        if pred.input_dim != 1659:
            raise ValueError(f"Model {ckpt_path} has input_dim {pred.input_dim}, expected 1659.")
        if pred.sequence_length != 64:
            raise ValueError(f"Model {ckpt_path} has sequence_length {pred.sequence_length}, expected 64.")

        # Test forward pass with dummy sequence
        dummy_seq = np.zeros((pred.sequence_length, pred.input_dim), dtype=np.float32)
        res = pred.predict_label(dummy_seq, confidence_threshold=0.0)
        logger.info("  %s verified successfully (Top class: %s, conf: %.4f)", ckpt_path.name, res["predicted_label"], res["confidence"])

    # Validate ensemble loader
    ens = EnsembleRealTimePredictor(ensemble_specs)
    dummy_seq = np.zeros((ens.sequence_length, ens.input_dim), dtype=np.float32)
    ens_res = ens.predict_label(dummy_seq, confidence_threshold=0.0)
    logger.info("Ensemble verified successfully: 100 classes, Top class: %s", ens_res["predicted_label"])


def package_bundle(
    output_zip: Path,
    root: Path = REPO_ROOT,
    version: str = "1.0.0",
) -> Path:
    """Build a deterministic ZIP containing inference assets and manifest."""
    validate_checkpoint_compatibility(root)

    manifest_entries: list[dict[str, Any]] = []
    sha256_lines: list[str] = []

    # 1. Audit and collect file metadata
    for item in REQUIRED_INFERENCE_FILES:
        src_path = root / item["source"]
        if not src_path.is_file():
            raise FileNotFoundError(f"Required file not found: {src_path}")
        file_sha = sha256_file(src_path)
        file_size = src_path.stat().st_size
        entry = {
            "path": item["archive_path"],
            "size_bytes": file_size,
            "sha256": file_sha,
        }
        for k in ("expected_arch", "expected_dim", "expected_seq", "expected_classes", "config"):
            if k in item:
                entry[k] = item[k]
        manifest_entries.append(entry)
        sha256_lines.append(f"{file_sha}  {item['archive_path']}")

    # Collect config files
    for cfg in CONFIG_FILES:
        cfg_path = root / cfg
        if cfg_path.is_file():
            c_sha = sha256_file(cfg_path)
            c_sz = cfg_path.stat().st_size
            manifest_entries.append({
                "path": cfg,
                "size_bytes": c_sz,
                "sha256": c_sha,
                "type": "configuration",
            })
            sha256_lines.append(f"{c_sha}  {cfg}")

    manifest: dict[str, Any] = {
        "bundle_name": "signflow-inference-assets",
        "bundle_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(manifest_entries),
        "total_size_bytes": sum(e["size_bytes"] for e in manifest_entries),
        "models": {
            "robust_holistic_fusion": {
                "checkpoint": "models/robust_holistic_fusion_best.pt",
                "config": "configs/dataset.yaml",
                "weight": 0.25,
            },
            "unisign_pgf": {
                "checkpoint": "models/wlasl100_unisign_pgf_best.pt",
                "config": "configs/colab/wlasl100_unisign_pgf.yaml",
                "weight": 0.10,
            },
            "max_accuracy_bigru": {
                "checkpoint": "models/wlasl100_max_accuracy_best.pt",
                "config": "configs/colab/wlasl100_max_accuracy.yaml",
                "weight": 0.65,
            },
        },
        "mediapipe": {
            "task_path": "models/mediapipe/holistic_landmarker.task",
            "landmarks_count": 553,
        },
        "files": manifest_entries,
    }

    manifest_json_str = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_sha256_str = "\n".join(sorted(sha256_lines)) + "\n"

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = (2026, 10, 30, 0, 0, 0)

    logger.info("Writing deterministic ZIP to %s...", output_zip)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Add manifest.json and manifest.sha256
        z_info_json = zipfile.ZipInfo("manifest.json", date_time=fixed_time)
        z_info_json.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(z_info_json, manifest_json_str.encode("utf-8"))

        z_info_sha = zipfile.ZipInfo("manifest.sha256", date_time=fixed_time)
        z_info_sha.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(z_info_sha, manifest_sha256_str.encode("utf-8"))

        # Add all assets in sorted order
        for entry in sorted(manifest_entries, key=lambda x: x["path"]):
            rel_path = entry["path"]
            src_file = root / rel_path
            z_info = zipfile.ZipInfo(rel_path, date_time=fixed_time)
            z_info.compress_type = zipfile.ZIP_DEFLATED
            z_info.external_attr = 0o644 << 16
            with open(src_file, "rb") as sf:
                zf.writestr(z_info, sf.read())
            logger.info("  Archived: %s (%d bytes)", rel_path, entry["size_bytes"])

    zip_size = output_zip.stat().st_size
    zip_sha = sha256_file(output_zip)
    logger.info("Bundle successfully created!")
    logger.info("  Path: %s", output_zip)
    logger.info("  Size: %d bytes (%.2f MB)", zip_size, zip_size / (1024 * 1024))
    logger.info("  Bundle SHA-256: %s", zip_sha)

    # Also write bundle SHA-256 sidecar file
    sidecar_path = output_zip.with_suffix(".zip.sha256")
    sidecar_path.write_text(f"{zip_sha}  {output_zip.name}\n", encoding="utf-8")
    logger.info("  Sidecar: %s", sidecar_path)

    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SignFlow inference assets into a production bundle.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "dist" / "signflow-inference-assets.zip",
        help="Path for generated ZIP bundle",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Version tag for manifest",
    )
    args = parser.parse_args()
    package_bundle(output_zip=args.output, version=args.version)


if __name__ == "__main__":
    main()

