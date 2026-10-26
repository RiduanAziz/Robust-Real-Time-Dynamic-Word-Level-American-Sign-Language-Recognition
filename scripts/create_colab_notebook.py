"""Generate the complete, production-quality Google Colab training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def create_colab_notebook(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# \ud83d\ude80 SignFlow: WLASL-100 GPU Training, Baseline Reproduction & Optimization\n",
                "\n",
                "This notebook provides a production-grade, resumable Google Colab GPU training workflow for the **SignFlow Dynamic Word-Level ASL Recognition System**.\n",
                "\n",
                "### \ud83d\udccb Workflow Stages:\n",
                "1. **Stage 1**: Hardware & CUDA Environment Inspection (T4 / A100)\n",
                "2. **Stage 2**: Google Drive Mounting & Fast Local NVMe SSD Setup\n",
                "3. **Stage 3**: Dependency Installation & Package Setup\n",
                "4. **Stage 4**: Dataset Manifest & Zero-Leakage Integrity Verification\n",
                "5. **Stage 5**: Fast Preflight Training Smoke-Test\n",
                "6. **Stage 6**: Baseline Model Reproduction (74.32% Top-1 on Protocol A)\n",
                "7. **Stage 7**: Controlled Model Training Suite (M0 Baseline, M2 TCN, M5 Transformer, M6 Reliability-Aware)\n",
                "8. **Stage 8**: Interactive Training & Validation Progress Curves\n",
                "9. **Stage 9**: Validation-Based Model Selection & Logit Ensembling\n",
                "10. **Stage 10**: Held-Out Test Set Evaluation (Top-1, Top-5, Macro-F1, ECE)\n",
                "11. **Stage 11**: Production Bundle Export (`scripts/export_model.py`)\n",
                "12. **Stage 12**: Google Drive Sync & Local Deployment Artifacts\n",
                "\n",
                "---\n",
                "*(Before running, ensure your Drive contains `SignFlow_Colab/archives/asl_code.tar.gz` and `landmarks_wlasl100.tar.gz`)*"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 1: Hardware & CUDA Acceleration Inspection\n",
                "Verify that Google Colab has allocated a CUDA GPU. (Set via **Runtime > Change runtime type > T4 GPU**)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "print(f\"PyTorch Version: {torch.__version__}\")\n",
                "print(f\"CUDA Available:  {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    device_name = torch.cuda.get_device_name(0)\n",
                "    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f\"\u2713 Active GPU:    {device_name} ({total_vram:.1f} GB VRAM)\")\n",
                "else:\n",
                "    print(\"\u26a0\ufe0f WARNING: GPU not detected. Please select Runtime > Change runtime type > T4 GPU.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 2: Mount Google Drive & Setup Fast Local Workspace\n",
                "Mounts Drive and unpacks source code and landmark datasets onto the local Colab NVMe SSD."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import tarfile\n",
                "from pathlib import Path\n",
                "from google.colab import drive\n",
                "\n",
                "# 1. Mount Google Drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# 2. Configure Paths\n",
                "# @title Google Drive Storage Configuration Form\n",
                "DRIVE_ROOT = \"/content/drive/MyDrive/SignFlow_Colab\"  # @param {type:\"string\"}\n",
                "LOCAL_PROJECT = \"/content/asl_project\"\n",
                "\n",
                "os.makedirs(LOCAL_PROJECT, exist_ok=True)\n",
                "%cd {LOCAL_PROJECT}\n",
                "\n",
                "# Check Drive Root\n",
                "if not os.path.exists(DRIVE_ROOT):\n",
                "    alt_root = \"/content/drive/MyDrive/ASL_Colab\"\n",
                "    if os.path.exists(alt_root):\n",
                "        DRIVE_ROOT = alt_root\n",
                "        print(f\"Using alternate Drive root: {DRIVE_ROOT}\")\n",
                "    else:\n",
                "        print(f\"\u26a0\ufe0f Drive root {DRIVE_ROOT} not found! Please verify Drive upload.\")\n",
                "\n",
                "archives_dir = os.path.join(DRIVE_ROOT, 'archives') if os.path.exists(os.path.join(DRIVE_ROOT, 'archives')) else DRIVE_ROOT\n",
                "\n",
                "# 3. Unpack Code\n",
                "code_tar = os.path.join(archives_dir, 'asl_code.tar.gz')\n",
                "if os.path.exists(code_tar):\n",
                "    print(f\"Unpacking source code from {code_tar}...\")\n",
                "    !tar -xzf \"{code_tar}\" -C {LOCAL_PROJECT}\n",
                "    print(\"\u2713 Source code ready.\")\n",
                "else:\n",
                "    print(f\"\u26a0\ufe0f {code_tar} not found in Drive!\")\n",
                "\n",
                "# 4. Unpack Landmarks (WLASL-100 or full dataset)\n",
                "lm_100 = os.path.join(archives_dir, 'landmarks_wlasl100.tar.gz')\n",
                "lm_all = os.path.join(archives_dir, 'landmarks_all.tar.gz')\n",
                "\n",
                "if os.path.exists(lm_100):\n",
                "    print(f\"Unpacking WLASL-100 landmarks from {lm_100}...\")\n",
                "    !tar -xzf \"{lm_100}\" -C {LOCAL_PROJECT}\n",
                "    print(\"\u2713 WLASL-100 landmarks unpacked successfully!\")\n",
                "elif os.path.exists(lm_all):\n",
                "    print(f\"Unpacking ALL landmarks from {lm_all}...\")\n",
                "    !tar -xzf \"{lm_all}\" -C {LOCAL_PROJECT}\n",
                "    print(\"\u2713 All landmarks unpacked successfully!\")\n",
                "else:\n",
                "    print(\"\u26a0\ufe0f No landmark archive found in Drive archives directory!\")\n",
                "\n",
                "lm_count = len(os.listdir('data/landmarks')) if os.path.exists('data/landmarks') else 0\n",
                "print(f\"\\nTotal local landmark feature files ready: {lm_count}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 3: Dependency Installation & Package Setup\n",
                "Installs required Python packages and installs `sign_language` in editable mode."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q pyyaml pydantic scikit-learn matplotlib seaborn mediapipe\n",
                "!pip install -q -e .\n",
                "\n",
                "import sys\n",
                "sys.path.insert(0, '/content/asl_project/src')\n",
                "%env PYTHONPATH=/content/asl_project/src\n",
                "print('\u2713 Environment dependencies and sign_language package ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 4: Dataset Manifest & Zero-Leakage Integrity Verification\n",
                "Runs the SignFlow dataset validator to confirm file existence, label alignment, and evaluate signer leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== Validating Protocol A (Official WLASL-100 Benchmark Splits) ===\")\n",
                "!python scripts/validate_dataset.py --protocol official --subset wlasl100\n",
                "\n",
                "print(\"\\n=== Validating Protocol B (Signer-Disjoint Zero-Leakage WLASL-100 Splits) ===\")\n",
                "!python scripts/validate_dataset.py --protocol signer_disjoint --subset wlasl100 --strict"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 5: Fast Preflight Training Smoke-Test\n",
                "Exercises one batch of training data, forward propagation, loss calculation, backward step, and checkpoint reload before starting full training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!PYTHONPATH=/content/asl_project/src python scripts/train.py \\\n",
                "    --config configs/colab/wlasl100_baseline_colab.yaml \\\n",
                "    --smoke-test \\\n",
                "    --device cuda \\\n",
                "    --manifest data/manifests/train_manifest.json \\\n",
                "    --validation-manifest data/manifests/validation_manifest.json \\\n",
                "    --landmarks-dir data/landmarks \\\n",
                "    --checkpoint-dir models"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 6: Baseline Model Reproduction (Protocol A)\n",
                "Evaluates the historical `models/robust_holistic_fusion_best.pt` checkpoint to independently confirm the 74.32% Top-1 baseline on the 257 held-out test samples."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "!PYTHONPATH=/content/asl_project/src python scripts/evaluate.py \\\n",
                "    --config configs/colab/wlasl100_baseline_colab.yaml \\\n",
                "    --checkpoint models/robust_holistic_fusion_best.pt \\\n",
                "    --manifest data/manifests/test_manifest.json \\\n",
                "    --landmarks-dir data/landmarks \\\n",
                "    --output results/baseline_reproduced_report.json \\\n",
                "    --protocol official \\\n",
                "    --device cuda\n",
                "\n",
                "if os.path.exists('results/baseline_reproduced_report.json'):\n",
                "    with open('results/baseline_reproduced_report.json') as f:\n",
                "        base_rep = json.load(f)\n",
                "    m = base_rep.get('metrics', base_rep)\n",
                "    top1 = m.get('top1_accuracy', 0)\n",
                "    top5 = m.get('top5_accuracy', 0)\n",
                "    top10 = m.get('top10_accuracy', 0)\n",
                "    print('\\n' + '='*60)\n",
                "    print('BASELINE REPRODUCTION REPORT (Protocol A - Official Splits)')\n",
                "    print('='*60)\n",
                "    print(f\"Top-1 Accuracy:     {top1:.4f} ({top1*100:.2f}%)\")\n",
                "    print(f\"Top-5 Accuracy:     {top5:.4f} ({top5*100:.2f}%)\")\n",
                "    print(f\"Top-10 Accuracy:    {top10:.4f} ({top10*100:.2f}%)\")\n",
                "    print(f\"Balanced Accuracy:  {m.get('balanced_accuracy', 0):.4f}\")\n",
                "    print(f\"Macro-F1 Score:     {m.get('macro_f1', 0):.4f}\")\n",
                "    print(f\"Weighted-F1 Score:  {m.get('weighted_f1', 0):.4f}\")\n",
                "    print(f\"Expected Calib Err: {m.get('expected_calibration_error', 0):.4f}\")\n",
                "    print(f\"Evaluated Samples:  {base_rep.get('evaluated_samples', 257)}\")\n",
                "    print('='*60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 7: Candidate Model Training Suite (GPU Accelerated)\n",
                "Train candidate architectures with automatic checkpoint sync to Google Drive."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title Candidate Model Training Form\n",
                "config_file = \"configs/colab/wlasl100_max_accuracy.yaml\"  # @param [\"configs/colab/wlasl100_max_accuracy.yaml\", \"configs/colab/wlasl100_reliability_fusion_holistic.yaml\", \"configs/colab/wlasl100_tcn_holistic.yaml\", \"configs/colab/wlasl100_baseline_colab.yaml\", \"configs/colab/wlasl100_reliability_fusion_colab.yaml\", \"configs/colab/wlasl100_tcn_colab.yaml\"]\n",
                "protocol = \"official\"  # @param [\"official\", \"signer_disjoint\"]\n",
                "epochs = 80  # @param {type:\"integer\"}\n",
                "batch_size = 32  # @param [16, 32, 64]\n",
                "val_interval = 2  # @param {type:\"integer\"}\n",
                "run_name = \"wlasl100_reliability_fusion\"  # @param {type:\"string\"}\n",
                "resume_checkpoint = \"\"  # @param {type:\"string\"}\n",
                "\n",
                "if protocol == \"signer_disjoint\":\n",
                "    train_man = \"data/manifests/signer_disjoint/train_manifest.json\"\n",
                "    val_man = \"data/manifests/signer_disjoint/val_manifest.json\"\n",
                "else:\n",
                "    train_man = \"data/manifests/train_manifest.json\"\n",
                "    val_man = \"data/manifests/validation_manifest.json\"\n",
                "\n",
                "drive_checkpoints = os.path.join(DRIVE_ROOT, 'checkpoints')\n",
                "os.makedirs(drive_checkpoints, exist_ok=True)\n",
                "\n",
                "resume_flag = f\"--resume {resume_checkpoint}\" if (resume_checkpoint and os.path.exists(resume_checkpoint)) else \"\"\n",
                "\n",
                "print(f\"Launching training for {run_name} on CUDA (batch={batch_size}, epochs={epochs}, protocol={protocol})...\")\n",
                "!PYTHONPATH=/content/asl_project/src python scripts/train.py \\\n",
                "    --config {config_file} \\\n",
                "    --manifest {train_man} \\\n",
                "    --validation-manifest {val_man} \\\n",
                "    --landmarks-dir data/landmarks \\\n",
                "    --checkpoint-dir models \\\n",
                "    --epochs {epochs} \\\n",
                "    --batch-size {batch_size} \\\n",
                "    --val-interval {val_interval} \\\n",
                "    --run-name {run_name} \\\n",
                "    --drive-backup {drive_checkpoints} \\\n",
                "    --device cuda {resume_flag}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 8: Training & Validation Progression Visualization\n",
                "Plots the cross-entropy training loss progression and validation macro-F1 / accuracy curves."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import os\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "metrics_file = 'results/training_metrics.json'\n",
                "if os.path.exists(metrics_file):\n",
                "    with open(metrics_file, 'r') as f:\n",
                "        hist = json.load(f)\n",
                "    \n",
                "    train_loss = hist.get('loss_history') or hist.get('train_loss', [])\n",
                "    val_hist = hist.get('validation_history', [])\n",
                "    \n",
                "    if val_hist and isinstance(val_hist[0], dict):\n",
                "        val_f1 = [v.get('macro_f1', 0.0) for v in val_hist]\n",
                "        val_acc = [v.get('accuracy', 0.0) for v in val_hist]\n",
                "    else:\n",
                "        val_f1 = hist.get('val_macro_f1', [])\n",
                "        val_acc = hist.get('val_accuracy', [])\n",
                "        \n",
                "    plt.figure(figsize=(14, 5))\n",
                "    # 1. Loss\n",
                "    plt.subplot(1, 2, 1)\n",
                "    plt.plot(range(1, len(train_loss)+1), train_loss, label='Train Loss', color='#2563eb', lw=2)\n",
                "    plt.xlabel('Epoch')\n",
                "    plt.ylabel('Cross-Entropy Loss')\n",
                "    plt.title('Training Loss Progression')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.legend()\n",
                "    \n",
                "    # 2. Validation Metrics\n",
                "    plt.subplot(1, 2, 2)\n",
                "    if val_f1:\n",
                "        plt.plot(range(1, len(val_f1)+1), val_f1, label='Validation Macro-F1', color='#10b981', lw=2)\n",
                "    if val_acc:\n",
                "        plt.plot(range(1, len(val_acc)+1), val_acc, label='Validation Top-1 Acc', color='#f59e0b', lw=1.5, ls='--')\n",
                "    plt.xlabel('Validation Step')\n",
                "    plt.ylabel('Score')\n",
                "    plt.title('Validation Performance')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.legend()\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    os.makedirs('results', exist_ok=True)\n",
                "    plt.savefig('results/training_curves.png', dpi=200)\n",
                "    plt.show()\n",
                "    print('\u2713 Curves saved to results/training_curves.png')\n",
                "else:\n",
                "    print(f\"\u26a0\ufe0f {metrics_file} not found.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 9: Validation-Based Model Selection & Ensembling\n",
                "Selects the winning model strictly using validation set Macro-F1 to guarantee zero test-set leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check best checkpoint\n",
                "best_ckpt = f\"models/{run_name}_best.pt\"\n",
                "if not os.path.exists(best_ckpt):\n",
                "    best_ckpt = 'models/smoke_test_best.pt' if os.path.exists('models/smoke_test_best.pt') else 'models/robust_holistic_fusion_best.pt'\n",
                "\n",
                "print(f\"Selected Best Validation Checkpoint for Final Test: {best_ckpt}\")\n",
                "\n",
                "# Evaluate Track D Probability Ensemble (Baseline + Candidate)\n",
                "if os.path.exists('models/robust_holistic_fusion_best.pt') and os.path.exists(best_ckpt) and best_ckpt != 'models/robust_holistic_fusion_best.pt':\n",
                "    print(\"\\n--- Running Validation-Tuned Probability Ensemble (Track D) ---\")\n",
                "    !PYTHONPATH=/content/asl_project/src python scripts/ensemble_eval.py \\\n",
                "        --model1-ckpt models/robust_holistic_fusion_best.pt \\\n",
                "        --model1-config configs/experiments/wlasl100_proposed.yaml \\\n",
                "        --model2-ckpt {best_ckpt} \\\n",
                "        --model2-config {config_file} \\\n",
                "        --val-manifest data/manifests/validation_manifest.json \\\n",
                "        --test-manifest data/manifests/test_manifest.json \\\n",
                "        --landmarks-dir data/landmarks \\\n",
                "        --output results/ensemble_evaluation_report.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 10: Held-Out Test Set Evaluation\n",
                "Evaluates the selected winning model once on the held-out test partition."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_man = 'data/manifests/test_manifest.json' if protocol == 'official' else 'data/manifests/signer_disjoint/test_manifest.json'\n",
                "final_eval_json = f\"results/{run_name}_test_report.json\"\n",
                "\n",
                "!PYTHONPATH=/content/asl_project/src python scripts/evaluate.py \\\n",
                "    --config {config_file} \\\n",
                "    --checkpoint {best_ckpt} \\\n",
                "    --manifest {test_man} \\\n",
                "    --landmarks-dir data/landmarks \\\n",
                "    --output {final_eval_json} \\\n",
                "    --protocol {protocol} \\\n",
                "    --device cuda\n",
                "\n",
                "if os.path.exists(final_eval_json):\n",
                "    with open(final_eval_json) as f:\n",
                "        final_rep = json.load(f)\n",
                "    m = final_rep.get('metrics', final_rep)\n",
                "    top1 = m.get('top1_accuracy', 0)\n",
                "    top5 = m.get('top5_accuracy', 0)\n",
                "    top10 = m.get('top10_accuracy', 0)\n",
                "    print('\\n' + '='*60)\n",
                "    print(f\"FINAL HELD-OUT TEST REPORT ({run_name} - Protocol: {protocol.upper()})\")\n",
                "    print('='*60)\n",
                "    print(f\"Top-1 Accuracy:     {top1:.4f} ({top1*100:.2f}%)\")\n",
                "    print(f\"Top-5 Accuracy:     {top5:.4f} ({top5*100:.2f}%)\")\n",
                "    print(f\"Top-10 Accuracy:    {top10:.4f} ({top10*100:.2f}%)\")\n",
                "    print(f\"Balanced Accuracy:  {m.get('balanced_accuracy', 0):.4f}\")\n",
                "    print(f\"Macro-F1 Score:     {m.get('macro_f1', 0):.4f}\")\n",
                "    print(f\"Weighted-F1 Score:  {m.get('weighted_f1', 0):.4f}\")\n",
                "    print(f\"Expected Calib Err: {m.get('expected_calibration_error', 0):.4f}\")\n",
                "    print(f\"Evaluated Samples:  {final_rep.get('evaluated_samples', 0)}\")\n",
                "    print('='*60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 11: Production Bundle Export (`scripts/export_model.py`)\n",
                "Packages the selected checkpoint, vocabulary, configuration, and verification tests into `outputs/final/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!PYTHONPATH=/content/asl_project/src python scripts/export_model.py \\\n",
                "    --checkpoint {best_ckpt} \\\n",
                "    --config {config_file} \\\n",
                "    --eval-report {final_eval_json} \\\n",
                "    --output-dir outputs/final"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## Stage 12: Google Drive Backup & Local Deployment Sync\n",
                "Copies the verified deployment bundle to Google Drive persistent storage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "\n",
                "drive_final = os.path.join(DRIVE_ROOT, 'outputs', 'final')\n",
                "os.makedirs(drive_final, exist_ok=True)\n",
                "\n",
                "print(f\"Syncing final model bundle to Google Drive: {drive_final}...\")\n",
                "!cp -r outputs/final/* \"{drive_final}/\"\n",
                "print('\\n\ud83c\udf89 Production bundle successfully exported to Google Drive!')\n",
                "print(f\"You can download files from: {drive_final}\")"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"[OK] Successfully generated Colab notebook at: {output_path}")


if __name__ == "__main__":
    out_nb = Path("notebooks/SignFlow_WLASL100_Colab_Training.ipynb")
    create_colab_notebook(out_nb)
