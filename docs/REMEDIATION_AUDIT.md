# SignFlow Master Remediation Audit & Technical Defensibility Report

This document records the comprehensive audit, source-code evidence, reproduction commands, severities, remediation changes, regression tests, and verification statuses for the SignFlow WLASL-100 sign-language recognition pipeline.

## Audit Matrix

| ID | Defect / Risk | Current Source-Code Evidence | Reproduction Command | Severity | Files Changed | Fix Implemented | Regression Test | Verification Status |
|:---|:---|:---|:---|:---|:---|:---|:---|:---:|
| **D1** | **Failing CI Test: Representation Selection Contract** | `tests/test_features_phase8.py:44`: called `.shape` directly on tuple return `(features, mask)` from `select_feature_representation`. | `python -m pytest tests/test_features_phase8.py -k test_representation_selection_preserves_time_and_selects_prefix` | **Critical** (Blocked CI) | `tests/test_features_phase8.py` | Unpacked `selected_features, selected_mask = select_feature_representation(...)`; asserted on feature shape, mask shape, and values. | `tests/test_features_phase8.py` (3 passed) | **VERIFIED** |
| **D2** | **Feature Contract Violation: Collator Sliced Dimensions** | `scripts/train.py:PipelineCollator`: sliced `coords[:, :feature_dim]`, yielding truncated holistic coordinates instead of genuine representations for 365 (engineered) or 4977 (dynamics). | Running training with `feature_dim=365` received raw holistic coordinate prefix instead of inter-hand/wrist-relative features. | **Critical** (Silent feature corruption) | `src/sign_language/landmarks/pipeline.py`, `scripts/train.py`, `src/sign_language/features/representations.py` | Implemented `canonical_preprocess_sequence` and `resolve_feature_representation`; collator builds genuine representations and verifies input dimension. | `tests/test_preprocessing_parity.py` | **VERIFIED** |
| **D3** | **Training Augmentation Mask Contradiction** | `scripts/train.py:PipelineCollator`: dropped points modified `pt_mask`, but the un-augmented `mask` was interpolated and returned to model, marking dropped points valid. | Inspecting output masks after landmark dropout showed `mask == 1.0` on dropped coordinates. | **High** (False confidence on synthetic noise) | `src/sign_language/landmarks/pipeline.py`, `scripts/train.py` | In `canonical_preprocess_sequence`, `pt_valid = pt_valid * drop_pts` propagates dropout to feature masks before interpolation. | `tests/test_preprocessing_parity.py::test_training_augmentation_mask_consistency` | **VERIFIED** |
| **D4** | **Preprocessing Discrepancy Between Training & Inference** | `scripts/train.py:PipelineCollator` vs `LandmarkPipeline.normalize_sequence_with_mask` duplicated centering, scaling, and padding. | Numerical mismatch between offline evaluation tensor and collator tensor on identical input. | **High** (Train/eval skew) | `src/sign_language/landmarks/pipeline.py`, `scripts/evaluate.py`, `scripts/ensemble_eval.py`, `src/sign_language/api/inference.py` | Unified all preprocessing onto `canonical_preprocess_sequence`. Training-only augmentation is the sole divergence. | `tests/test_preprocessing_parity.py::test_training_eval_preprocessing_parity` | **VERIFIED** |
| **D5** | **Colab Notebook Manifest Filename Discrepancy** | `scripts/create_colab_notebook.py:Stage 3`: referenced `signer_disjoint/val_manifest.json`, but the actual manifest file is `validation_manifest.json`. | Running Stage 3 in Colab raised `FileNotFoundError: val_manifest.json`. | **High** (Execution blocker) | `scripts/create_colab_notebook.py`, `notebooks/SignFlow_WLASL100_Colab_Training.ipynb`, `train_colab.ipynb` | Corrected path to `validation_manifest.json` and added preflight existence check. | `python scripts/create_colab_notebook.py` | **VERIFIED** |
| **D6** | **Colab Silent Fallback to Smoke Test Checkpoints** | `scripts/create_colab_notebook.py:Stage 9`: fell back to `models/smoke_test_best.pt` if target checkpoint was absent. | Evaluating non-existent checkpoint evaluated dummy smoke-test weights without raising an error. | **High** (Scientific validity hazard) | `scripts/create_colab_notebook.py`, `notebooks/SignFlow_WLASL100_Colab_Training.ipynb`, `train_colab.ipynb` | Removed silent fallback; raised explicit `FileNotFoundError` with remediation instructions. | Code inspection & notebook regeneration | **VERIFIED** |
| **D7** | **EDA Analysis Coverage & File Naming Conflation** | `scripts/run_master_eda_and_audit.py`: checked landmarks but named output `missing_videos.csv`; sampled only 500 training samples while claiming full dataset audit. | `python scripts/run_master_eda_and_audit.py` produced misleading filenames and incomplete coverage. | **Medium** (Reporting defect) | `scripts/run_master_eda_and_audit.py` | Expanded to 100% census (all 2,038 samples); separated `missing_landmarks.csv` (0 missing) from `missing_videos.csv` (2,038 raw MP4s in archive); added `corrupted_landmarks.csv` (0). | `python scripts/run_master_eda_and_audit.py` (exit 0) | **VERIFIED** |
| **D8** | **Signer-Disjoint Protocol Incomplete Class Coverage Policy** | `scripts/validate_dataset.py`: passed Protocol B despite validation missing 2 classes (`['help', 'short']`) and test missing 6 classes (`['cousin', 'drink', 'fine', 'like', 'medicine', 'play']`). | `python scripts/validate_dataset.py --protocol signer_disjoint --subset wlasl100` passed without class coverage warnings. | **Medium** (Scientific reporting flaw) | `scripts/validate_dataset.py` | Added explicit class support counting, missing classes breakdown, coverage percentage reporting, and `--full-scan` mode for 100% feature verification. | `python scripts/validate_dataset.py --protocol signer_disjoint --subset wlasl100 --full-scan` | **VERIFIED** |
| **D9** | **Metric & Vocabulary Hash Contradiction in Results Documentation** | `docs/COLAB_FINAL_RESULTS.md`: listed baseline Top-10 as 94.16% (the ensemble score) and hash prefix `5b4e7a89277f...`, contradicting `results/baseline_reproduced_report.json` (Top-10: 93.39%, hash: `50a15ecd4abb...`). | Comparing `docs/COLAB_FINAL_RESULTS.md` against canonical `results/baseline_reproduced_report.json`. | **Medium** (Documentation discrepancy) | `docs/COLAB_FINAL_RESULTS.md` | Corrected baseline Top-10 to 93.39% (240 / 257) and canonical vocabulary hash to `50a15ecd4abb546fe0e4859713ed6a6c34e4a35caddec1ddc9fdf79c41d19975`. | Verified against JSON prediction artifacts | **VERIFIED** |
| **D10** | **Metric Implementation Standardization & Edge Cases** | Duplicated ECE across scripts; missing edge-case handling for non-finite logits, tied logits, absent classes, and empty samples. | Feeding `NaN` or `Inf` to `pure_top_k_accuracy_score` or `compute_expected_calibration_error`. | **Medium** (Robustness hazard) | `src/sign_language/training/metrics.py`, `scripts/evaluate.py`, `scripts/ensemble_eval.py`, `tests/test_metrics_authoritative.py` | Consolidated `pure_expected_calibration_error` into `src/sign_language/training/metrics.py`; added non-finite sanitization; added 10 unit tests for all edge cases. | `tests/test_metrics_authoritative.py` (10 passed) | **VERIFIED** |
| **D11** | **Ensemble Validation vs Test Data Leakage Hazard** | `scripts/ensemble_eval.py`: read test manifest on every candidate run; lacked verification of sample ID alignment. | Running candidate search could repeatedly consult test accuracy to pick candidate weights. | **Medium** (Overfitting hazard) | `scripts/ensemble_eval.py` | Added `--validation-only` mode; added explicit assertion that `sample_ids` match across all models; saved comprehensive `predictions.npz`. | `scripts/ensemble_eval.py` | **VERIFIED** |
| **D12** | **Export Artifact Provenance & Clean-Load Verification** | `scripts/export_model.py`: exported static schema 1.0.0 without provenance metadata or clean-load test. | Missing SHA256 and git commit in exported bundle. | **Low** (MLOps best practice) | `scripts/export_model.py`, `tests/test_export_clean_load.py` | Generated `provenance.json` with checkpoint SHA256, git commit, schema version 2.0.0; automated clean-load test. | `tests/test_export_clean_load.py` (passed) | **VERIFIED** |

---

## Technical Defensibility & Viva Examiner Summary

### 1. Protocol Distinction
- **Protocol A (Official WLASL Benchmark):**
  - Train: 1,444 samples (91 signers, 100 classes).
  - Validation: 337 samples (68 signers, 100 classes).
  - Test: 257 samples (55 signers, 100 classes).
  - **Signer Overlap:** Documented 51 shared signers between Train and Test, 63 between Train and Val, 49 between Val and Test. **Protocol A is NOT signer-independent**; it is the official literature benchmark for direct comparison with published SOTA.
- **Protocol B (Signer-Disjoint Generalization):**
  - Train: 1,433 samples (71 signers, 100 classes).
  - Validation: 370 samples (14 signers, 98 classes, missing: `['help', 'short']`).
  - Test: 235 samples (12 signers, 94 classes, missing: `['cousin', 'drink', 'fine', 'like', 'medicine', 'play']`).
  - **Signer Overlap:** Strictly 0 (zero sample leakage, zero signer overlap).
  - **Thesis Defense Stance:** 100% test class coverage is mathematically impossible under strict signer-disjoint partitioning because certain signs in WLASL were only performed by 1 or 2 signers whose entire sample set is placed in Train or Val. Protocol B test must be reported as 94-class evaluation, not 100-class evaluation.

### 2. Verified Authoritative Baseline Metrics (Protocol A)
- **Top-1 Accuracy:** 74.32% (191 / 257)
- **Top-5 Accuracy:** 88.72% (228 / 257)
- **Top-10 Accuracy:** 93.39% (240 / 257)
- **Balanced Accuracy:** 73.92%
- **Macro Precision:** 0.7628
- **Macro Recall:** 0.7392
- **Macro-F1 Score:** 0.7213
- **Weighted-F1 Score:** 0.7257
- **Expected Calibration Error (ECE):** 0.1598
- **Vocabulary SHA256:** `50a15ecd4abb546fe0e4859713ed6a6c34e4a35caddec1ddc9fdf79c41d19975`

### 3. Verified Authoritative 3-Way Ensemble Metrics (Protocol A)
- **Optimal Weights:** $(w_1=0.25, w_2=0.10, w_3=0.65)$ tuned strictly on validation set.
- **Validation Top-1:** 78.34% (264 / 337)
- **Test Top-1 Accuracy:** 76.65% (197 / 257)
- **Test Top-5 Accuracy:** 91.44% (235 / 257)
- **Test Top-10 Accuracy:** 94.16% (242 / 257)
- **Ensemble Macro-F1:** 0.7558
- **Ensemble Weighted-F1:** 0.7542
- **Ensemble ECE:** 0.0838
