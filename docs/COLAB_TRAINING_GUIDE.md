# SignFlow: Google Colab GPU Training User Guide

This guide details the step-by-step procedure to train, optimize, evaluate, and export research-grade WLASL-100 sign language recognition models using Google Colab's free or paid GPU runtimes (Tesla T4, V100, or A100).

---

## 1. Prerequisites & Google Drive Setup

Before opening Google Colab, verify that your Google Drive contains the packaged assets as specified in [`docs/COLAB_DATA_UPLOAD_CHECKLIST.md`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/docs/COLAB_DATA_UPLOAD_CHECKLIST.md):

```
Google Drive > MyDrive > SignFlow_Colab > archives/
├── asl_code.tar.gz            (~3.5 MB: code, configs, manifests, baseline weights)
└── landmarks_wlasl100.tar.gz  (~350 MB: 2,038 landmark .npz files for 100 classes)
```

*(To generate these archives from your local machine, run `python scripts/package_for_colab.py --subset wlasl100`)*.

---

## 2. Launching Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Select **Upload** and upload `notebooks/SignFlow_WLASL100_Colab_Training.ipynb` from your local repository.
3. Switch runtime to GPU:
   - Click **Runtime > Change runtime type**.
   - Select **T4 GPU** (or **A100 GPU** if using Colab Pro).
   - Click **Save**.

---

## 3. Notebook Execution Stages

The notebook is divided into 12 logically self-contained, resumable stages:

### Stage 1: Hardware & CUDA Inspection
- Executes `!nvidia-smi` and checks PyTorch CUDA availability.
- Prints active GPU device name and allocated VRAM (e.g., Tesla T4 15.0 GB).

### Stage 2: Google Drive Mounting & Local Workspace Setup
- Mounts `/content/drive`.
- Extracts `asl_code.tar.gz` and `landmarks_wlasl100.tar.gz` onto fast local NVMe SSD storage at `/content/asl_project`.
- Verifies that all 2,038 landmark files are extracted without corruption.

### Stage 3: Dependency Installation & Environment
- Installs pinned requirements (`pyyaml`, `scikit-learn`, `matplotlib`, `seaborn`, `mediapipe`).
- Installs the `sign_language` package in editable mode (`pip install -e .`).
- Sets `PYTHONPATH=/content/asl_project/src`.

### Stage 4: Dataset Protocol & Integrity Verification
- Runs `python scripts/validate_dataset.py --protocol official` (Protocol A).
- Runs `python scripts/validate_dataset.py --protocol signer_disjoint --strict` (Protocol B).
- Confirms 0 missing files and exact class-to-gloss alignments.

### Stage 5: Baseline Reproduction (Protocol A)
- Evaluates `models/robust_holistic_fusion_best.pt` on the 257 held-out test samples.
- Confirms exact reproduction of:
  - **Top-1 Accuracy**: 74.32%
  - **Top-5 Accuracy**: 88.72%
  - **Macro-F1**: 0.7213

### Stage 6: Fast Smoke Test
- Executes a 1-epoch, 8-sample preflight test on the configured architecture.
- Validates forward pass, backward pass, optimizer step, checkpoint save, and reload before starting a long training run.

### Stage 7: Candidate Model Training (Validation Optimization)
- Uses an interactive form to select:
  - **Architecture**: `M6: Reliability-Aware Fusion`, `M2: TCN`, `M5: Transformer`, or `M0: Baseline`.
  - **Protocol**: `official` (Protocol A) or `signer_disjoint` (Protocol B).
  - **Hyperparameters**: Epochs (e.g., 80), Batch Size (e.g., 32), Learning Rate (e.g., 0.001).
- Training logs progress epoch-by-epoch. Checkpoints are automatically synced to Drive at each epoch (`--drive-backup`).

### Stage 8: Training & Validation Progression Visualization
- Generates dual plots for Cross-Entropy Loss and Validation Macro-F1 / Top-1 Accuracy.
- Saves high-resolution figure to `results/training_curves.png`.

### Stage 9: Model Selection & Ensembling
- Measures validation macro-F1 across trained candidates.
- Evaluates validation logit blending ($\alpha \mathbf{z}_{\text{M6}} + (1 - \alpha) \mathbf{z}_{\text{M2}}$).
- Locks the winning model without consulting the test set.

### Stage 10: Final Held-Out Test Evaluation
- Evaluates the winning model **once** on the held-out test partition.
- Reports Top-1, Top-5, Top-10, Balanced Accuracy, Macro-F1, and ECE.

### Stage 11: Production Bundle Export
- Runs `scripts/export_model.py` to create the complete export bundle in `outputs/final/`.
- Verifies model reload and deterministic forward pass.

### Stage 12: Google Drive Backup & Local Sync
- Automatically archives the final bundle to `/content/drive/MyDrive/SignFlow_Colab/outputs/final/`.
- Download instructions for local integration into the React frontend and FastAPI backend.

---

## 4. Resuming Interrupted Training Runs

If Colab disconnects or terminates your session:
1. Re-run Stages 1 through 3 to mount Drive and extract the code.
2. In Stage 7 form, set `resume_checkpoint = "/content/drive/MyDrive/SignFlow_Colab/checkpoints/wlasl100_reliability_fusion.pt"`.
3. Re-run Stage 7. Training will automatically resume from the exact last saved epoch with restored optimizer and scheduler states.
