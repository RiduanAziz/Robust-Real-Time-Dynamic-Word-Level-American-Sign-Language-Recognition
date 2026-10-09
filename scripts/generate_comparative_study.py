import json
from pathlib import Path


def main():
    with open("results/wlasl20_baseline_evaluation.json", "r", encoding="utf-8") as f:
        base_eval = json.load(f)

    with open("results/wlasl20_proposed_evaluation.json", "r", encoding="utf-8") as f:
        prop_eval = json.load(f)

    with open("results/robustness/wlasl20_baseline_report.json", "r", encoding="utf-8") as f:
        base_rob = json.load(f)

    with open("results/robustness/wlasl20_proposed_report.json", "r", encoding="utf-8") as f:
        prop_rob = json.load(f)

    with open("results/baseline_training_metrics.json", "r", encoding="utf-8") as f:
        base_train = json.load(f)

    with open("results/proposed_training_metrics.json", "r", encoding="utf-8") as f:
        prop_train = json.load(f)

    lines = []
    lines.append("# Empirical Thesis Comparison: Baseline vs. Proposed Architecture on WLASL-20")
    lines.append("")
    lines.append("## 1. Experimental Protocol")
    lines.append("- **Vocabulary (WLASL-20)**: 20 high-frequency ASL words (`before`, `thin`, `cool`, `drink`, `go`, `computer`, `who`, `cousin`, `help`, `candy`, `thanksgiving`, `bed`, `bowling`, `tall`, `accident`, `short`, `yes`, `what`, `later`, `man`).")
    lines.append("- **Signer Independence**: Mutually disjoint signers across train (122 samples), validation (74 samples), and held-out test (78 samples). Zero signer leakage.")
    lines.append("- **Input Representation**: Canonical 553-point holistic landmarks with 3D coordinate normalization, uniform temporal resampling (64 frames), boundary-safe velocity/acceleration dynamics (4,977 input dimensions).")
    lines.append("- **Epochs**: 10 epochs with identical optimizer (AdamW, lr=0.001) and batch size 16.")
    lines.append("")
    lines.append("## 2. Model Performance on Held-Out Test Signers")
    lines.append("")
    lines.append("| Metric | Baseline (Temporal Transformer) | Proposed (RobustHolisticFusionClassifier) | Relative Improvement |")
    lines.append("|---|---|---|---|")

    b_acc = base_eval["metrics"]["top1_accuracy"]
    p_acc = prop_eval["metrics"]["top1_accuracy"]
    rel_acc = ((p_acc - b_acc) / max(1e-5, b_acc)) * 100

    b_f1 = base_eval["metrics"]["macro_f1"]
    p_f1 = prop_eval["metrics"]["macro_f1"]
    rel_f1 = ((p_f1 - b_f1) / max(1e-5, b_f1)) * 100

    b_wf1 = base_eval["metrics"]["weighted_f1"]
    p_wf1 = prop_eval["metrics"]["weighted_f1"]
    rel_wf1 = ((p_wf1 - b_wf1) / max(1e-5, b_wf1)) * 100

    best_val_b = base_train["best_macro_f1"]
    best_val_p = prop_train["best_macro_f1"]
    rel_val = ((best_val_p - best_val_b) / max(1e-5, best_val_b)) * 100

    lines.append(f"| **Top-1 Accuracy** | {b_acc:.4f} ({b_acc*100:.2f}%) | **{p_acc:.4f} ({p_acc*100:.2f}%)** | **+{rel_acc:.1f}%** |")
    lines.append(f"| **Macro-F1** | {b_f1:.4f} | **{p_f1:.4f}** | **+{rel_f1:.1f}%** |")
    lines.append(f"| **Weighted-F1** | {b_wf1:.4f} | **{p_wf1:.4f}** | **+{rel_wf1:.1f}%** |")
    lines.append(f"| **Best Val Macro-F1** | {best_val_b:.4f} | **{best_val_p:.4f}** | **+{rel_val:.1f}%** |")
    lines.append(f"| **Final Train Loss** | {base_train['loss_history'][-1]:.4f} | **{prop_train['loss_history'][-1]:.4f}** | - |")
    lines.append("")
    lines.append("## 3. Spatial and Temporal Robustness Evaluation")
    lines.append("")
    lines.append("Degradation is defined as Clean Accuracy - Noisy Accuracy (lower degradation is better; negative degradation indicates noise resistance/augmentation benefit).")
    lines.append("")
    lines.append("| Disturbance Type | Severity | Baseline Noisy Acc | Baseline Degradation | Proposed Noisy Acc | Proposed Degradation | Robustness Advantage |")
    lines.append("|---|---|---|---|---|---|---|")

    b_recs = {(r["noise_type"], r["severity"]): r for r in base_rob["results"]}
    p_recs = {(r["noise_type"], r["severity"]): r for r in prop_rob["results"]}

    for key in sorted(b_recs.keys()):
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

    out_file = Path("results/wlasl20_comparative_study.md")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()
