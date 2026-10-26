import argparse
import json
import logging
import sys
from pathlib import Path

# Automatically add 'src' directory to Python path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Force UTF-8 unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        top_k_accuracy_score,
    )
except Exception:
    from sign_language.training.metrics import (
        pure_accuracy_score as accuracy_score,
        pure_balanced_accuracy_score as balanced_accuracy_score,
        pure_confusion_matrix as confusion_matrix,
        pure_precision_recall_fscore_support as precision_recall_fscore_support,
        pure_top_k_accuracy_score as top_k_accuracy_score,
    )

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest, CanonicalVocabulary
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
from sign_language.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sign-language model on held-out test data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", "--model-path", type=str, required=True, dest="checkpoint", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--manifest", type=str, default="data/manifests/test_manifest.json", help="Path to test manifest")
    parser.add_argument("--landmarks-dir", type=str, default="data/landmarks", help="Path to extracted landmarks directory")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json", help="Path to write evaluation report")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation")
    parser.add_argument("--protocol", type=str, default="official", help="Name of evaluation protocol (e.g. official, signer_disjoint)")
    parser.add_argument("--missing-policy", type=str, default="strict", choices=["strict", "diagnostic", "ignore"], help="Policy for missing feature files")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max test samples to evaluate")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)

    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        class_names = ckpt.get("class_names")
        saved_config = ckpt.get("config", {})
        saved_vocab_hash = ckpt.get("vocabulary_hash")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        class_names = None
        saved_config = {}
        saved_vocab_hash = None
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    config = load_experiment_config(args.config)
    if saved_config and isinstance(saved_config, dict):
        if "model" in saved_config and isinstance(saved_config["model"], dict):
            for k, v in saved_config["model"].items():
                setattr(config.model, k, v)
        if "dataset" in saved_config and isinstance(saved_config["dataset"], dict):
            for k, v in saved_config["dataset"].items():
                setattr(config.dataset, k, v)

    if class_names is None:
        class_names = list(config.dataset.labels)

    canonical_vocab = CanonicalVocabulary(classes=class_names)
    if saved_vocab_hash:
        canonical_vocab.validate_compatibility(saved_vocab_hash)

    input_dim = ckpt.get("input_dim", config.model.input_dim) if isinstance(ckpt, dict) else config.model.input_dim
    seq_len = ckpt.get("sequence_length", config.dataset.sequence_length) if isinstance(ckpt, dict) else config.dataset.sequence_length

    label_to_index = canonical_vocab.label_to_index
    config.model.num_classes = len(class_names)
    config.model.input_dim = input_dim

    # 2. Build Model & Load Weights
    model = build_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Load Test Samples with diagnostics
    test_samples, diagnostics = build_dataset_from_manifest(
        manifest_path=args.manifest,
        landmarks_dir=args.landmarks_dir,
        allowed_classes=class_names,
        missing_policy=args.missing_policy,
        return_diagnostics=True,
    )
    if args.limit:
        test_samples = test_samples[: args.limit]

    if not test_samples:
        raise ValueError(f"No valid test samples found in manifest {args.manifest} and directory {args.landmarks_dir}")

    logger.info("Evaluating %d held-out test samples under protocol: %s...", len(test_samples), args.protocol)

    pipeline = LandmarkPipeline(feature_dim=input_dim, sequence_length=seq_len)
    test_dataset = SignLanguageDataset(test_samples, label_to_index=label_to_index)

    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    all_logits_list: list[np.ndarray] = []

    with torch.no_grad():
        for item in test_dataset:
            target_idx = item["label"]
            seq = item["landmarks"]
            mask = item.get("mask")
            norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                seq, mask=mask
            )
            tensor_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
            mask_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(tensor_in, mask=mask_in)
            logits_np = logits.cpu().numpy().squeeze(0)
            pred_idx = int(np.argmax(logits_np))

            targets.append(target_idx)
            predictions.append(pred_idx)
            sample_ids.append(item["sample_id"])
            all_logits_list.append(logits_np)

    targets_arr = np.array(targets, dtype=np.int64)
    preds_arr = np.array(predictions, dtype=np.int64)
    logits_matrix = np.stack(all_logits_list, axis=0)
    probabilities = F.softmax(torch.from_numpy(logits_matrix), dim=-1).numpy()

    # 4. Compute Comprehensive Evaluation Metrics
    accuracy = float(accuracy_score(targets_arr, preds_arr))
    balanced_acc = float(balanced_accuracy_score(targets_arr, preds_arr))

    top5_acc = None
    if len(class_names) >= 5 and len(targets) >= 5:
        top5_acc = float(top_k_accuracy_score(targets_arr, logits_matrix, k=5, labels=list(range(len(class_names)))))

    top10_acc = None
    if len(class_names) >= 10 and len(targets) >= 10:
        top10_acc = float(top_k_accuracy_score(targets_arr, logits_matrix, k=10, labels=list(range(len(class_names)))))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_arr, preds_arr, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets_arr, preds_arr, average="weighted", zero_division=0
    )

    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets_arr, preds_arr, labels=list(range(len(class_names))), zero_division=0
    )

    conf_mat = confusion_matrix(targets_arr, preds_arr, labels=list(range(len(class_names))))
    ece = compute_expected_calibration_error(probabilities, targets_arr)

    report = {
        "protocol": args.protocol,
        "checkpoint": str(checkpoint_path),
        "manifest": str(args.manifest),
        "total_manifest_samples": diagnostics["total_manifest_samples"],
        "matching_manifest_samples": diagnostics["matching_class_samples"],
        "evaluated_samples": len(targets),
        "excluded_samples_count": diagnostics["excluded_samples_count"],
        "num_classes": len(class_names),
        "vocabulary_hash": canonical_vocab.vocabulary_hash,
        "class_names": class_names,
        "metrics": {
            "top1_accuracy": accuracy,
            "top5_accuracy": top5_acc,
            "top10_accuracy": top10_acc,
            "balanced_accuracy": balanced_acc,
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
            "expected_calibration_error": ece,
        },
        "per_class": {
            class_names[i]: {
                "precision": float(p_per_class[i]),
                "recall": float(r_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
            for i in range(len(class_names))
        },
        "evaluated_sample_ids": sample_ids,
        "confusion_matrix": conf_mat.tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation Complete:")
    logger.info("  Protocol:       %s", args.protocol)
    logger.info("  Evaluated:      %d / %d samples", len(targets), diagnostics["total_manifest_samples"])
    logger.info("  Top-1 Accuracy: %.4f", accuracy)
    if top5_acc is not None:
        logger.info("  Top-5 Accuracy: %.4f", top5_acc)
    logger.info("  Macro-F1:       %.4f", float(f1_macro))
    logger.info("  Weighted-F1:    %.4f", float(f1_weighted))
    logger.info("  ECE:            %.4f", ece)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
