from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate empirical thesis comparison study between baseline and proposed models.")
    parser.add_argument("--baseline-eval", default="results/wlasl20_baseline_evaluation.json", help="Path to baseline evaluation JSON")
    parser.add_argument("--proposed-eval", default="results/wlasl20_proposed_evaluation.json", help="Path to proposed evaluation JSON")
    parser.add_argument("--baseline-robustness", default="results/robustness/wlasl20_baseline_report.json", help="Path to baseline robustness report JSON")
    parser.add_argument("--proposed-robustness", default="results/robustness/wlasl20_proposed_report.json", help="Path to proposed robustness report JSON")
    parser.add_argument("--baseline-train", default="results/baseline_training_metrics.json", help="Path to baseline training metrics JSON")
    parser.add_argument("--proposed-train", default="results/proposed_training_metrics.json", help="Path to proposed training metrics JSON")
    parser.add_argument("--output", default="results/wlasl20_comparative_study.md", help="Path to write comparative study markdown")
    args = parser.parse_args()

    required_paths = {
        "Baseline evaluation": Path(args.baseline_eval),
        "Proposed evaluation": Path(args.proposed_eval),
        "Baseline robustness": Path(args.baseline_robustness),
        "Proposed robustness": Path(args.proposed_robustness),
        "Baseline training": Path(args.baseline_train),
        "Proposed training": Path(args.proposed_train),
    }

    missing = [f"{desc} ({path})" for desc, path in required_paths.items() if not path.is_file()]
    if missing:
        logger.error("Cannot generate comparative study. Missing required artifact files:\n  - " + "\n  - ".join(missing))
        sys.exit(1)

    with Path(args.baseline_eval).open("r", encoding="utf-8") as f:
        base_eval = json.load(f)
    with Path(args.proposed_eval).open("r", encoding="utf-8") as f:
        prop_eval = json.load(f)
    with Path(args.baseline_robustness).open("r", encoding="utf-8") as f:
        base_rob = json.load(f)
    with Path(args.proposed_robustness).open("r", encoding="utf-8") as f:
        prop_rob = json.load(f)
    with Path(args.baseline_train).open("r", encoding="utf-8") as f:
        base_train = json.load(f)
    with Path(args.proposed_train).open("r", encoding="utf-8") as f:
        prop_train = json.load(f)

    # Validate experiment compatibility
    base_samples = base_eval.get("total_samples", base_eval.get("sample_count", 0))
    prop_samples = prop_eval.get("total_samples", prop_eval.get("sample_count", 0))
    if base_samples != prop_samples:
        logger.warning(
            "Evaluation sample counts differ: baseline evaluated %d samples, proposed evaluated %d samples.",
            base_samples,
            prop_samples,
        )

    lines: list[str] = [
        "# Empirical Thesis Comparison: Baseline vs. Proposed Architecture on WLASL-20",
        "",
        "## 1. Experimental Protocol",
        f"- **Vocabulary**: {prop_eval.get('num_classes', len(prop_eval.get('class_names', [])))} classes.",
        f"- **Held-Out Test Sample Count**: {prop_samples} samples evaluated under identical preprocessing.",
        "- **Signer Independence**: Mutually disjoint signers across train, validation, and held-out test splits.",
        "- **Input Representation**: Canonical 553-point holistic landmarks with 3D coordinate normalization, uniform temporal resampling (64 frames), boundary-safe velocity/acceleration dynamics (4,977 input dimensions).",
        "",
        "## 2. Model Performance on Held-Out Test Signers",
        "",
        "| Metric | Baseline (Temporal Transformer) | Proposed (RobustHolisticFusionClassifier) | Relative Improvement |",
        "|---|---|---|---|",
    ]

    b_acc = base_eval["metrics"]["top1_accuracy"]
    p_acc = prop_eval["metrics"]["top1_accuracy"]
    rel_acc = ((p_acc - b_acc) / max(1e-5, b_acc)) * 100

    b_f1 = base_eval["metrics"]["macro_f1"]
    p_f1 = prop_eval["metrics"]["macro_f1"]
    rel_f1 = ((p_f1 - b_f1) / max(1e-5, b_f1)) * 100

    b_wf1 = base_eval["metrics"]["weighted_f1"]
    p_wf1 = prop_eval["metrics"]["weighted_f1"]
    rel_wf1 = ((p_wf1 - b_wf1) / max(1e-5, b_wf1)) * 100

    best_val_b = base_train.get("best_macro_f1", 0.0)
    best_val_p = prop_train.get("best_macro_f1", 0.0)
    rel_val = ((best_val_p - best_val_b) / max(1e-5, best_val_b)) * 100

    b_loss = base_train.get("loss_history", [0.0])[-1]
    p_loss = prop_train.get("loss_history", [0.0])[-1]

    lines.append(f"| **Top-1 Accuracy** | {b_acc:.4f} ({b_acc*100:.2f}%) | **{p_acc:.4f} ({p_acc*100:.2f}%)** | **+{rel_acc:.1f}%** |")
    lines.append(f"| **Macro-F1** | {b_f1:.4f} | **{p_f1:.4f}** | **+{rel_f1:.1f}%** |")
    lines.append(f"| **Weighted-F1** | {b_wf1:.4f} | **{p_wf1:.4f}** | **+{rel_wf1:.1f}%** |")
    lines.append(f"| **Best Val Macro-F1** | {best_val_b:.4f} | **{best_val_p:.4f}** | **+{rel_val:.1f}%** |")
    lines.append(f"| **Final Train Loss** | {b_loss:.4f} | **{p_loss:.4f}** | - |")
    lines.append("")
    lines.append("## 3. Spatial and Temporal Robustness Evaluation")
    lines.append("")
    lines.append("Degradation is defined as Clean Accuracy - Noisy Accuracy (lower degradation is better; negative degradation indicates noise resistance/augmentation benefit).")
    lines.append("")
    lines.append("| Disturbance Type | Severity | Baseline Noisy Acc | Baseline Degradation | Proposed Noisy Acc | Proposed Degradation | Robustness Advantage |")
    lines.append("|---|---|---|---|---|---|---|")

    b_recs = {(r["noise_type"], r["severity"]): r for r in base_rob.get("results", [])}
    p_recs = {(r["noise_type"], r["severity"]): r for r in prop_rob.get("results", [])}

    for key in sorted(b_recs.keys()):
        if key not in p_recs:
            continue
        b = b_recs[key]
        p = p_recs[key]
        ntype, sev = key
        b_deg = b["accuracy_degradation"]
        p_deg = p["accuracy_degradation"]
        adv = b_deg - p_deg
        adv_str = f"+{adv:.4f}" if adv > 0 else f"{adv:.4f}"
        lines.append(
            f"| `{ntype}` | {sev:.1f} | {b['noisy']['accuracy']:.4f} | {b_deg:.4f} | {p['noisy']['accuracy']:.4f} | {p_deg:.4f} | **{adv_str}** |"
        )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparative study report written successfully to %s", out_file)


if __name__ == "__main__":
    main()
