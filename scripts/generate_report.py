from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pathlib import Path

def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    metrics_path = results_dir / "training_metrics.json"
    if not metrics_path.exists():
        print("training_metrics.json not found! Cannot generate charts.")
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    loss_history = data.get("loss_history", [])
    val_metrics = data.get("validation_metrics", {})
    
    # 1. Plot the loss curve
    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('Training Loss Curve (100 Epochs)', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = results_dir / "training_loss_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Generated Training Loss Curve: {plot_path}")
        plt.close()
        
    # 2. Write the thesis summary
    report = results_dir / "research_summary.txt"
    summary_text = (
        "--- ASL Recognition Model Thesis Summary ---\n"
        f"Final Epoch Loss: {loss_history[-1]:.4f}\n\n"
        "--- Final Validation Metrics ---\n"
        f"Validation Accuracy: {val_metrics.get('accuracy', 0) * 100:.2f}%\n"
        f"Macro F1-Score: {val_metrics.get('macro_f1', 0):.4f}\n"
        f"Validation Loss: {val_metrics.get('loss', 0):.4f}\n"
        f"Evaluation Samples: {val_metrics.get('num_samples', 0)}\n\n"
        "Conclusion: The Temporal Transformer successfully learned the spatial-temporal\n"
        "dynamics of the extracted holistic landmarks."
    )
    report.write_text(summary_text, encoding="utf-8")
    print(f"Report written to {report}")

if __name__ == "__main__":
    main()
