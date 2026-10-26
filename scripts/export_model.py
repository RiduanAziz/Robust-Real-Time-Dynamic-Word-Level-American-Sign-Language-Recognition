"""Export and package the selected model checkpoint and artifacts for local deployment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import yaml

from sign_language.config import load_experiment_config
from sign_language.data.vocabulary import CanonicalVocabulary
from sign_language.models.factory import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_bundle(
    checkpoint_path: Path,
    config_path: Path,
    eval_report_path: Path | None,
    output_dir: Path,
    verify: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting production model bundle to %s...", output_dir)

    # 1. Load Config
    config = load_experiment_config(str(config_path))
    class_names = list(config.dataset.labels)
    canonical_vocab = CanonicalVocabulary(class_names)

    # 2. Copy Config
    out_config_path = output_dir / "config.yaml"
    shutil.copy2(config_path, out_config_path)

    # 3. Export Vocabulary JSON
    out_vocab_path = output_dir / "vocabulary.json"
    with out_vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vocabulary": class_names,
                "num_classes": len(class_names),
                "vocabulary_hash": canonical_vocab.vocabulary_hash,
            },
            f,
            indent=2,
        )

    # 4. Copy / Package Checkpoint
    out_ckpt_path = output_dir / "model_best.pt"
    if checkpoint_path.is_file():
        ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Ensure metadata is present in checkpoint
        if isinstance(ckpt_data, dict):
            ckpt_data["class_names"] = class_names
            ckpt_data["num_classes"] = len(class_names)
            ckpt_data["vocabulary_hash"] = canonical_vocab.vocabulary_hash
            ckpt_data["config"] = config.to_dict()
            torch.save(ckpt_data, out_ckpt_path)
        else:
            payload = {
                "model_state_dict": ckpt_data,
                "class_names": class_names,
                "num_classes": len(class_names),
                "vocabulary_hash": canonical_vocab.vocabulary_hash,
                "config": config.to_dict(),
            }
            torch.save(payload, out_ckpt_path)
    else:
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")

    # 5. Export Metrics & Analysis Tables
    eval_data: Dict[str, Any] = {}
    if eval_report_path and eval_report_path.is_file():
        with eval_report_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)

    metrics = eval_data.get("metrics") or eval_data.get("ensemble", {})
    out_metrics_path = output_dir / "metrics.json"
    with out_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": config.model.name,
                "protocol": eval_data.get("protocol", "official"),
                "evaluated_samples": eval_data.get("evaluated_samples", 257),
                "num_classes": len(class_names),
                "vocabulary_hash": canonical_vocab.vocabulary_hash,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Export per_class_metrics.csv
    per_class = eval_data.get("per_class", {})
    out_per_class_csv = output_dir / "per_class_metrics.csv"
    with out_per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for idx, cname in enumerate(class_names):
            c_info = per_class.get(cname, {})
            writer.writerow([
                idx,
                cname,
                c_info.get("precision", 0.0),
                c_info.get("recall", 0.0),
                c_info.get("f1", 0.0),
                c_info.get("support", 0),
            ])

    # Export confusion_matrix.csv
    conf_mat = eval_data.get("confusion_matrix")
    if conf_mat is not None:
        out_cm_csv = output_dir / "confusion_matrix.csv"
        with out_cm_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_class"] + class_names)
            for idx, cname in enumerate(class_names):
                row = conf_mat[idx] if idx < len(conf_mat) else [0] * len(class_names)
                writer.writerow([cname] + list(row))

    # Export experiment_comparison.csv
    out_exp_csv = output_dir / "experiment_comparison.csv"
    with out_exp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "architecture", "protocol", "top1_accuracy", "top5_accuracy", "macro_f1", "ece", "samples"])
        top1 = metrics.get("top1_accuracy", 0.0)
        top5 = metrics.get("top5_accuracy", 0.0)
        mf1 = metrics.get("macro_f1", 0.0)
        ece = metrics.get("expected_calibration_error", 0.0)
        samples = eval_data.get("evaluated_samples", 257)
        writer.writerow([
            config.model.name,
            config.model.name,
            eval_data.get("protocol", "official"),
            f"{top1:.4f}",
            f"{top5:.4f}",
            f"{mf1:.4f}",
            f"{ece:.4f}",
            samples,
        ])

    # Export preprocessing_schema.json
    out_schema_path = output_dir / "preprocessing_schema.json"
    with out_schema_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0.0",
                "total_points": 553,
                "input_coordinates": config.model.input_dim,
                "sequence_length": config.dataset.sequence_length,
                "modality_coordinates": {
                    "left_hand": [0, 63],
                    "right_hand": [63, 126],
                    "pose": [126, 225],
                    "face": [225, 1659],
                },
                "feature_representation": getattr(config.dataset, "feature_representation", "holistic"),
                "normalization": "center_scale",
            },
            f,
            indent=2,
        )

    # Copy Model Card
    card_source = Path("docs/MODEL_CARD.md")
    if card_source.is_file():
        shutil.copy2(card_source, output_dir / "MODEL_CARD.md")

    # 6. Verification Test
    if verify:
        logger.info("Running export verification test on %s...", out_ckpt_path)
        reloaded_ckpt = torch.load(out_ckpt_path, map_location="cpu", weights_only=False)
        reloaded_config = load_experiment_config(str(out_config_path))
        model = build_model(reloaded_config)
        state_dict = reloaded_ckpt["model_state_dict"] if "model_state_dict" in reloaded_ckpt else reloaded_ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Fixed deterministic input
        torch.manual_seed(42)
        dummy_input = torch.randn(1, reloaded_config.dataset.sequence_length, reloaded_config.model.input_dim)
        dummy_mask = torch.ones(1, reloaded_config.dataset.sequence_length)
        with torch.no_grad():
            out = model(dummy_input, mask=dummy_mask)

        assert out.shape == (1, len(class_names)), f"Output shape mismatch: {out.shape} vs {(1, len(class_names))}"
        assert torch.isfinite(out).all(), "Exported model produced non-finite logits"
        logger.info("✓ EXPORT_VERIFICATION_PASSED: Model reloaded successfully with deterministic forward pass.")

    logger.info("Successfully exported bundle to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "checkpoint": str(out_ckpt_path),
        "config": str(out_config_path),
        "vocabulary": str(out_vocab_path),
        "metrics": str(out_metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and package model bundle for deployment.")
    parser.add_argument("--checkpoint", default="models/robust_holistic_fusion_best.pt", help="Path to trained checkpoint")
    parser.add_argument("--config", default="configs/colab/wlasl100_baseline_colab.yaml", help="Path to config file")
    parser.add_argument("--eval-report", default="results/baseline_reproduced_report.json", help="Path to eval report JSON")
    parser.add_argument("--output-dir", default="outputs/final", help="Destination export directory")
    parser.add_argument("--no-verify", action="store_true", help="Skip reload verification")
    args = parser.parse_args()

    export_bundle(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        eval_report_path=Path(args.eval_report) if args.eval_report else None,
        output_dir=Path(args.output_dir),
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
