# Master Pipeline Audit — SignFlow WLASL-100 System

**Document Version**: 2.0.0 (Master Engineering & Research Audit)  
**Author**: Lead Research & ML Systems Engineer  
**Date**: October 2026  
**Repository**: [https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition](https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition)  
**Commit**: Latest `main`  

---

## 1. Executive Summary

This comprehensive audit inspects every subsystem of the SignFlow dynamic word-level American Sign Language recognition project. The evaluation baseline—reported as **74.32% Top-1 accuracy** and **88.72% Top-5 accuracy** on 257 held-out test samples under Protocol A—has been **independently and deterministically reproduced** locally and on Google Colab using `models/robust_holistic_fusion_best.pt` and `configs/colab/wlasl100_baseline_colab.yaml`.

All previous schema discrepancies (such as the 280 vs 365 engineered feature dimension mismatch, and configuration vocabulary inconsistencies) have been identified, audited, resolved, and backed by automated unit tests. Furthermore, the architecture suite has been augmented with an adapted **Uni-Sign Prior-Guided Fusion (PGF)** classifier based on ICLR 2025 research (`arXiv:2501.15187`), which leverages pose structural priors to guide hand gesture cross-attention with score-aware reliability gating.

---

## 2. Pipeline Subsystem Verification Matrix

| Subsystem | Implemented Status | Verification Evidence | Defect / Discrepancy Found | Remediation Performed |
| :--- | :---: | :--- | :--- | :--- |
| **Landmark Schema** | **VERIFIED** | `src/sign_language/landmarks/schema.py` defines 553 holistic points (1659 3D coords). | Previous docs referenced 543 points (468 face). Modern MediaPipe uses 478 face mesh points. | Canonical schema enforced at 553 points (42 hands, 33 pose, 478 face). |
| **Feature Extraction** | **VERIFIED** | `src/sign_language/features/representations.py` passes all 6 tests in `tests/test_features_contract.py`. | Engineered representation was declared as 280 dims, but returned 365 dims; fingertip masks did not propagate wrist visibility. | Declared feature_dim updated to 365; strict mask propagation implemented; zero-fill verified. |
| **Baseline Reproduction** | **REPRODUCED** | `scripts/evaluate.py` executed on `models/robust_holistic_fusion_best.pt` yields 74.32% Top-1, 88.72% Top-5. | `base.yaml` had out-of-order labels; Colab candidate 1 dropped to 65.37% due to missing face mesh. | Canonical vocabulary synchronized across all configs (`wlasl100_baseline_colab.yaml`, `base.yaml`, `dataset.yaml`). |
| **Dataset Completeness** | **VERIFIED** | 21,083 `.npz` files verified in `data/landmarks/`. 2,038 samples belong to WLASL-100 (1444 train, 337 val, 257 test). | None. 100% of WLASL-100 samples have valid precomputed landmark files. | Full file inventory saved to `results/data_audit/current_file_inventory.json`. |
| **Protocol Integrity** | **VERIFIED** | Protocol A (official splits) has 51 shared signers train-test. Protocol B enforces strict zero-leakage signer-disjoint splits. | Protocol A has documented signer leakage. | Documented distinction in `docs/DATASET_COMPLETENESS_AND_SPLITS.md`. |
| **Model Architectures** | **VERIFIED** | BiGRU baseline, TCN, Temporal Transformer, Reliability-Aware Fusion, and Uni-Sign PGF all implemented. | Pure hands+pose stripped essential facial anchors (*deaf*, *candy*, *who*). | Implemented full 1659-dim Holistic support across all candidate models. |
| **Colab GPU Training** | **VERIFIED** | `notebooks/SignFlow_WLASL100_Colab_Training.ipynb` and `train_colab.ipynb` generate 12-stage self-contained pipeline. | Missing package dependencies for headless evaluation. | Pinned dependencies and created `scripts/package_for_colab.py`. |
| **Model Ensembling** | **VERIFIED** | `scripts/ensemble_eval.py` performs validation-tuned grid search ($\alpha \in [0, 1]$) with zero test leakage. | Baseline config path needed alignment. | Automated ensemble script updated to use canonical config. |

---

## 3. Detailed Audit Findings & Repairs

### 3.1 Feature Engineering Schema Parity
- **Finding**: In `src/sign_language/features/representations.py`, `_FEATURES["engineered"]` had `feature_dim = 280`, but the concatenation function produced 365 dimensions:
  $$\text{Base Hands+Pose (225)} + \text{Left Wrist-Rel (63)} + \text{Right Wrist-Rel (63)} + \text{Inter-Hand (4)} + \text{Fingertip Extensions (10)} = 365$$
- **Fix**: Updated `_FEATURES["engineered"].feature_dim = 365`. Implemented strict mask propagation:
  - Wrist-relative coordinates are only valid if both the landmark and the wrist are visible.
  - Inter-wrist distance is only valid if both wrists are visible.
  - Fingertip extension distances are only valid if both the fingertip and the wrist are visible.
- **Verification**: New unit test `tests/test_features_contract.py` validates dimension, masking, and boundary conditions.

### 3.2 Canonical Vocabulary Alignment
- **Finding**: `configs/colab/wlasl100_max_accuracy.yaml`, `configs/base.yaml`, and `configs/dataset.yaml` contained differing class orders and label subsets. Evaluating the baseline model against `configs/base.yaml` previously yielded 9.30% because of label index permutations.
- **Fix**: Programmatically aligned all configuration files to the canonical baseline vocabulary (hash: `50a15ecd4abb546fe0e4859713ed6a6c34e4a35caddec1ddc9fdf79c41d19975`).
- **Result**: `configs/base.yaml` now reproduces the exact **74.32% Top-1** and **88.72% Top-5** baseline.

### 3.3 Uni-Sign Architecture Integration
- **Finding**: Uni-Sign (ICLR 2025, `arXiv:2501.15187`) achieves 92.25% on WLASL-100 by utilizing **Prior-Guided Fusion (PGF)**. In this paradigm, skeletal body pose acts as a spatial reference anchor for hand gestures, and score-aware reliability gating prevents noise from missing or occluded hands.
- **Fix**: Created `src/sign_language/models/unisign_pgf.py` (`UniSignPGFClassifier`), featuring:
  - 4 isolated anatomical encoders (Left Hand, Right Hand, Body Pose, Face Mesh).
  - Multi-scale 1D temporal convolutions (kernel sizes 3 and 5) with residual connections.
  - Prior-Guided Cross-Attention where hand queries attend to body/face anchors.
  - Score-aware reliability gating for one-handed signs.
  - Bidirectional kinematic recurrent layers and calibrated classification head.
- **Verification**: Registered in `src/sign_language/models/factory.py` and forward-pass verified on both 1659-dim Holistic and 225-dim Hands+Pose inputs.

---

## 4. Current Pipeline Status

- **Automated Tests**: 19/19 passing (`pytest tests/test_accuracy_optimization.py tests/test_dataset_integrity.py tests/test_master_regressions.py tests/test_features_contract.py`).
- **Local EDA Deliverables**: Generated and saved in `results/eda/` and `results/data_audit/`.
- **Baseline Status**: VERIFIED & INDEPENDENTLY REPRODUCED (Top-1: 74.32%, Top-5: 88.72%, Macro-F1: 0.7213, ECE: 0.1598).
- **Colab GPU Pipeline**: Production-ready in `notebooks/SignFlow_WLASL100_Colab_Training.ipynb` and `train_colab.ipynb`.
