#!/usr/bin/env python3
"""Probability-Level Ensemble Evaluator (SignFlow Track D).

Performs validation-tuned weighted probability fusion across candidate models:
1. Computes prediction probability matrices on the validation set for both models.
2. Grid-searches the optimal fusion weight alpha in [0.0, 1.0] strictly on validation data.
3. Evaluates the fused ensemble on the held-out test split using the locked alpha.
4. Reports Top-1, Top-5, Top-10, Macro-F1, Weighted-F1, and Expected Calibration Error (ECE).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from sign_language.config.loader import load_experiment_config
from sign_language.data.dataset import SignLanguageDataset, build_dataset_from_manifest
from sign_language.data.vocabulary import CanonicalVocabulary
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models.factory import build_model
from sign_language.training.metrics import (
    pure_accuracy_score,
    pure_top_k_accuracy_score,
    pure_precision_recall_fscore_support,
    pure_balanced_accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signflow.ensemble")


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def get_model_probabilities(
    model: torch.nn.Module,
    samples: list,
    pipeline: LandmarkPipeline,
    label_to_index: dict[str, int],
    input_dim: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Infers softmax probability distributions and target labels for a dataset."""
    dataset = SignLanguageDataset(samples, label_to_index=label_to_index)
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for item in dataset:
            target_idx = item["label"]
            seq = item["landmarks"]
            mask = item.get("mask")

            norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                seq, mask=mask, include_dynamics=(input_dim >= 4977)
            )
            t_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
            m_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(t_in, mask=m_in)
            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

            all_probs.append(probs)
            all_targets.append(target_idx)

    return np.array(all_probs, dtype=np.float32), np.array(all_targets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignFlow Probability-Level Ensemble Evaluator")
    parser.add_argument("--model1-ckpt", required=True, help="Path to Model 1 checkpoint (.pt)")
    parser.add_argument("--model1-config", required=True, help="Path to Model 1 config (.yaml)")
    parser.add_argument("--model2-ckpt", required=True, help="Path to Model 2 checkpoint (.pt)")
    parser.add_argument("--model2-config", required=True, help="Path to Model 2 config (.yaml)")
    parser.add_argument("--val-manifest", default="data/manifests/validation_manifest.json")
    parser.add_argument("--test-manifest", default="data/manifests/test_manifest.json")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument("--output", default="results/ensemble_evaluation_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load configs
    cfg1 = load_experiment_config(args.model1_config)
    cfg2 = load_experiment_config(args.model2_config)

    class_names = list(cfg1.dataset.labels)
    canonical_vocab = CanonicalVocabulary(classes=class_names)
    label_to_index = canonical_vocab.label_to_index

    # Build Model 1
    cfg1.model.num_classes = len(class_names)
    model1 = build_model(cfg1)
    ckpt1 = torch.load(args.model1_ckpt, map_location=device)
    sd1 = ckpt1.get("model_state_dict", ckpt1.get("state_dict", ckpt1))
    model1.load_state_dict(sd1)
    model1.to(device)
    raw_dim1 = cfg1.model.input_dim // 3 if cfg1.model.input_dim >= 4977 else cfg1.model.input_dim
    pipe1 = LandmarkPipeline(feature_dim=raw_dim1, sequence_length=cfg1.dataset.sequence_length)

    # Build Model 2
    cfg2.model.num_classes = len(class_names)
    model2 = build_model(cfg2)
    ckpt2 = torch.load(args.model2_ckpt, map_location=device)
    sd2 = ckpt2.get("model_state_dict", ckpt2.get("state_dict", ckpt2))
    model2.load_state_dict(sd2)
    model2.to(device)
    raw_dim2 = cfg2.model.input_dim // 3 if cfg2.model.input_dim >= 4977 else cfg2.model.input_dim
    pipe2 = LandmarkPipeline(feature_dim=raw_dim2, sequence_length=cfg2.dataset.sequence_length)

    # Load Validation Data
    val_samples = build_dataset_from_manifest(args.val_manifest, args.landmarks_dir, allowed_classes=class_names)
    test_samples = build_dataset_from_manifest(args.test_manifest, args.landmarks_dir, allowed_classes=class_names)

    if args.max_samples:
        val_samples = val_samples[: args.max_samples]
        test_samples = test_samples[: args.max_samples]

    logger.info("Computing validation probabilities (N=%d)...", len(val_samples))
    val_p1, val_targets = get_model_probabilities(model1, val_samples, pipe1, label_to_index, cfg1.model.input_dim, device)
    val_p2, _ = get_model_probabilities(model2, val_samples, pipe2, label_to_index, cfg2.model.input_dim, device)

    # Validation Grid Search for optimal alpha
    best_alpha = 0.5
    best_val_top1 = -1.0
    val_results = {}

    for alpha in np.linspace(0.0, 1.0, 21):
        alpha = float(round(alpha, 2))
        fused_val = alpha * val_p1 + (1.0 - alpha) * val_p2
        top1 = pure_accuracy_score(val_targets, np.argmax(fused_val, axis=-1))
        val_results[str(alpha)] = float(top1)
        if top1 > best_val_top1:
            best_val_top1 = top1
            best_alpha = alpha

    logger.info("Validation grid search optimal alpha: %.2f (Val Top-1: %.4f)", best_alpha, best_val_top1)

    # Evaluate Test Split
    logger.info("Computing test probabilities (N=%d)...", len(test_samples))
    test_p1, test_targets = get_model_probabilities(model1, test_samples, pipe1, label_to_index, cfg1.model.input_dim, device)
    test_p2, _ = get_model_probabilities(model2, test_samples, pipe2, label_to_index, cfg2.model.input_dim, device)

    # Individual metrics
    m1_top1 = pure_accuracy_score(test_targets, np.argmax(test_p1, axis=-1))
    m1_top5 = pure_top_k_accuracy_score(test_targets, test_p1, k=5)
    m2_top1 = pure_accuracy_score(test_targets, np.argmax(test_p2, axis=-1))
    m2_top5 = pure_top_k_accuracy_score(test_targets, test_p2, k=5)

    # Fused metrics with validation-tuned alpha
    fused_test = best_alpha * test_p1 + (1.0 - best_alpha) * test_p2
    fused_preds = np.argmax(fused_test, axis=-1)

    ens_top1 = pure_accuracy_score(test_targets, fused_preds)
    ens_top5 = pure_top_k_accuracy_score(test_targets, fused_test, k=5)
    ens_top10 = pure_top_k_accuracy_score(test_targets, fused_test, k=10)
    ens_bal_acc = pure_balanced_accuracy_score(test_targets, fused_preds)
    ens_p_mac, ens_r_mac, ens_f1_mac, _ = pure_precision_recall_fscore_support(test_targets, fused_preds, average="macro")
    ens_p_wt, ens_r_wt, ens_f1_wt, _ = pure_precision_recall_fscore_support(test_targets, fused_preds, average="weighted")
    ens_ece = compute_expected_calibration_error(fused_test, test_targets)

    report = {
        "model1": {"name": cfg1.model.name, "checkpoint": args.model1_ckpt, "top1": float(m1_top1), "top5": float(m1_top5)},
        "model2": {"name": cfg2.model.name, "checkpoint": args.model2_ckpt, "top1": float(m2_top1), "top5": float(m2_top5)},
        "ensemble": {
            "optimal_alpha": float(best_alpha),
            "validation_top1": float(best_val_top1),
            "test_top1": float(ens_top1),
            "test_top5": float(ens_top5),
            "test_top10": float(ens_top10),
            "balanced_accuracy": float(ens_bal_acc),
            "macro_f1": float(ens_f1_mac),
            "weighted_f1": float(ens_f1_wt),
            "ece": float(ens_ece),
            "evaluated_samples": len(test_targets),
        },
        "alpha_grid_search": val_results,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("SIGNFLOW PROBABILITY ENSEMBLE EVALUATION REPORT")
    print("=" * 60)
    print(f"Model 1 ({cfg1.model.name}):       Top-1: {m1_top1:.4f} ({m1_top1*100:.2f}%), Top-5: {m1_top5:.4f}")
    print(f"Model 2 ({cfg2.model.name}):       Top-1: {m2_top1:.4f} ({m2_top1*100:.2f}%), Top-5: {m2_top5:.4f}")
    print(f"Optimal Alpha (Model 1 Weight):    {best_alpha:.2f} (tuned on validation split)")
    print("-" * 60)
    print(f"Ensemble Top-1 Accuracy:           {ens_top1:.4f} ({ens_top1*100:.2f}%)")
    print(f"Ensemble Top-5 Accuracy:           {ens_top5:.4f} ({ens_top5*100:.2f}%)")
    print(f"Ensemble Top-10 Accuracy:          {ens_top10:.4f} ({ens_top10*100:.2f}%)")
    print(f"Ensemble Macro-F1:                 {ens_f1_mac:.4f}")
    print(f"Ensemble Weighted-F1:              {ens_f1_wt:.4f}")
    print(f"Ensemble ECE:                      {ens_ece:.4f}")
    print("=" * 60)
    print(f"Report written to: {out_p}")


if __name__ == "__main__":
    main()
