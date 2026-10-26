# SignFlow: Google Colab GPU Training & Optimization Plan

**Document Version:** 1.0-research  
**Date:** October 26, 2026  
**Target:** Maximizing Defensible WLASL-100 Top-1 and Top-5 Accuracy via Controlled GPU Training  
**Author:** Senior Machine Learning Engineer & PyTorch Specialist  

---

## 1. System Architecture & Verified Repository Entry Points

An audit of the repository checkout confirms the following active components and interfaces:

### 1.1 Core Modules & Entry Points
- **Training CLI**: [`scripts/train.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/scripts/train.py)
  - Supports `--config`, `--manifest`, `--validation-manifest`, `--landmarks-dir`, `--checkpoint-dir`, `--device (cuda/cpu)`, `--epochs`, `--batch-size`, `--amp`, `--val-interval`, `--resume`, `--smoke-test`, and `--drive-backup`.
  - Integrates automatic mixed precision (AMP) via `torch.cuda.amp.autocast`.
  - Checkpoint metadata contains `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_macro_f1`, `config`, `vocabulary_hash`, `schema_version`, and `seed`.
- **Evaluation CLI**: [`scripts/evaluate.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/scripts/evaluate.py)
  - Computes Top-1, Top-5, Top-10 categorical accuracy, balanced accuracy, macro/weighted precision, recall, F1, Expected Calibration Error (ECE), and confusion matrix.
  - Features pure-NumPy fallback via [`src/sign_language/training/metrics.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/training/metrics.py) to guarantee platform portability across Linux and Windows without C-extension DLL crashes.
- **Model Factory**: [`src/sign_language/models/factory.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/models/factory.py)
  - Constructs `MLPClassifier`, `LSTMClassifier`, `GRUClassifier`, `TCNClassifier`, `TemporalTransformerClassifier`, `RobustHolisticFusionClassifier`, and `ReliabilityAwareIsolatedFusionClassifier`.
- **Dataset Preprocessing & Dataloaders**:
  - [`src/sign_language/data/dataset.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/data/dataset.py): `build_dataset_from_manifest` with `missing_policy="strict"` and non-silent sample exclusion disclosure.
  - [`src/sign_language/landmarks/pipeline.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/landmarks/pipeline.py): Vectorized spatial normalization and linear temporal interpolation to $T=64$ frames.
  - [`src/sign_language/landmarks/missing_data.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/landmarks/missing_data.py): Short-gap linear interpolation ($\le 3$ frames) for transient occlusion recovery.
  - [`src/sign_language/data/vocabulary.py`](file:///f:/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/src/sign_language/data/vocabulary.py): Canonical SHA256 vocabulary locking preventing vocabulary drift.

---

## 2. Google Drive Storage Architecture & Directory Contract

To guarantee seamless resumption and persistence across Colab session disconnects, the workflow defines a strict Google Drive layout:

```
/content/drive/MyDrive/SignFlow_Colab/
├── archives/
│   ├── asl_code.tar.gz                   # Source code, configs, manifests (~3.5 MB)
│   ├── landmarks_wlasl100.tar.gz         # Extracted landmarks for 100 classes (~350 MB, 2,038 samples)
│   └── landmarks_all.tar.gz              # (Optional) Full 21,083 landmark archives (~3.2 GB)
├── configs/
│   ├── wlasl100_baseline_colab.yaml      # M0 Baseline configuration
│   ├── wlasl100_tcn_colab.yaml           # M2 Temporal Convolutional Network
│   ├── wlasl100_transformer_colab.yaml   # M5 Temporal Transformer
│   └── wlasl100_reliability_fusion_colab.yaml # M6 Proposed Reliability-Aware Model
├── checkpoints/                          # Auto-synced checkpoints per epoch
│   ├── wlasl100_baseline_best.pt
│   ├── wlasl100_tcn_best.pt
│   ├── wlasl100_transformer_best.pt
│   └── wlasl100_reliability_fusion_best.pt
└── outputs/                              # Experiment evaluation reports and metrics
    ├── baseline_eval_report.json
    ├── candidate_comparison_table.csv
    └── final/                            # Selected export bundle for local deployment
        ├── model_best.pt
        ├── config.yaml
        ├── vocabulary.json
        ├── metrics.json
        └── model_card.md
```

---

## 3. Training & Optimization Methodology (Phased Colab Workflow)

### Stage 1: Runtime & Hardware Verification
- Inspect Colab runtime: verify GPU device name, compute capability, VRAM size ($15.0$ GB on T4, $40.0$ GB on A100).
- Pin dependencies: `torch>=2.0.0`, `pyyaml`, `scikit-learn`, `matplotlib`, `seaborn`, `mediapipe>=0.10.0`.
- Verify editable install of `sign_language` package and set `PYTHONPATH`.

### Stage 2: Data Extraction & Preflight Integrity
- Mount Drive and extract `asl_code.tar.gz` and `landmarks_wlasl100.tar.gz` directly to `/content/asl_project` on fast local NVMe SSD storage.
- Execute `scripts/validate_dataset.py --protocol official` and `scripts/validate_dataset.py --protocol signer_disjoint --strict`.
- Fail early if landmark files are missing or vocabulary hashes are misaligned.

### Stage 3: Baseline Reproduction (Protocol A)
- Evaluate historical baseline checkpoint (`models/robust_holistic_fusion_best.pt`).
- Verify exact reproduction of 74.32% Top-1, 88.72% Top-5, and 0.7213 Macro-F1 on the 257 held-out test samples.

### Stage 4: Controlled Candidate Model Training (Validation Set Optimization)
- Train candidate architectures with identical seeds, data splits, and batch budgets:
  1. **M0**: `RobustHolisticFusionClassifier` (Recurrent GRU + Modality Gating)
  2. **M2**: `TCNClassifier` (Dilated 1D Causal Convolutions)
  3. **M5**: `TemporalTransformerClassifier` (Multi-Head Temporal Self-Attention)
  4. **M6**: `ReliabilityAwareIsolatedFusionClassifier` (Proposed Dynamic Modality Reliability Gating)
- Apply training-only optimizations:
  - Cosine annealing learning rate scheduler with 5-epoch warm-up.
  - Label smoothing ($\alpha = 0.05$).
  - Mild coordinate jitter ($\sigma = 0.01$) and temporal sequence warping ($\pm 10\%$).
  - Model selection strictly governed by **Validation Macro-F1 / Validation Accuracy**, avoiding any test-set leakage.

### Stage 5: Ensembling & Model Selection
- Evaluate validation logit averaging:
  $$\mathbf{z}_{\text{ens}} = \alpha \mathbf{z}_{\text{M6}} + (1 - \alpha) \mathbf{z}_{\text{M2}}$$
- Select the single best model or ensemble based solely on validation performance.

### Stage 6: Final Held-Out Test Evaluation & Export
- Lock the selected architecture and evaluate on the held-out test partition once.
- Export all artifacts to `/content/drive/MyDrive/SignFlow_Colab/outputs/final/` for download and integration into the local application.
