# SignFlow: Accuracy Optimization & Comprehensive System Audit

**Audit Date:** October 26, 2026  
**Auditor:** Senior Computer Vision & ML Engineering Specialist  
**Repository:** `https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`  
**Target Vocabulary:** WLASL-100 (100 Word-Level Dynamic ASL Signs)  

---

## 1. Executive Summary & Baseline Status

The reported repository baseline of **74.32% Top-1 accuracy** on WLASL-100 was audited, tested, and **independently reproduced** from local artifacts.

| Metric | Repository Reported | Independently Reproduced (Local CPU) | Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Categorical Accuracy** | 74.32% (0.7432) | **74.32% (191 / 257)** | **VERIFIED & REPRODUCED** |
| **Top-5 Categorical Accuracy** | *Unreported in summary* | **88.72% (228 / 257)** | **EMPIRICALLY VERIFIED** |
| **Macro Precision** | 0.7628 | **0.7628** | **VERIFIED & REPRODUCED** |
| **Macro Recall** | 0.7392 | **0.7392** | **VERIFIED & REPRODUCED** |
| **Macro-F1 Score** | 0.7213 | **0.7213** | **VERIFIED & REPRODUCED** |
| **Weighted-F1 Score** | 0.7257 | **0.7257** | **VERIFIED & REPRODUCED** |
| **Expected Calibration Error (ECE)** | *Unreported* | **0.1598** | **EMPIRICALLY VERIFIED** |
| **Evaluated Samples** | 257 | **257** | **VERIFIED & REPRODUCED** |
| **Total Test Manifest Samples** | 2,878 | **2,878** | **VERIFIED & REPRODUCED** |
| **Checkpoint Path** | `models/robust_holistic_fusion_best.pt` | `models/robust_holistic_fusion_best.pt` | **VERIFIED** |
| **Active Epoch** | Epoch 53 | Epoch 53 | **VERIFIED** |

---

## 2. End-to-End Architecture & Data Flow

```
Raw Camera Frame (RGB) / Video File
               │
               ▼
   [ MediaPipe HolisticLandmarker ]
   - Left Hand:   21 keypoints × 3 =  63 dims
   - Right Hand:  21 keypoints × 3 =  63 dims
   - Pose Stream: 33 keypoints × 3 =  99 dims
   - Face Stream: 478 keypoints × 3 = 1,434 dims
               │
               ▼
   [ Feature Normalization & Velocity Derivation ]
   - Spatial coordinates: D = 1,659 dims
   - First-order velocity: D = 1,659 dims
   - Second-order acceleration: D = 1,659 dims
   - Holistic concatenation: Total Input Dimension = 4,977 (or 1,659 static)
               │
               ▼
   [ Stream-Isolated Spatial Projection Heads ]
   - Left Hand Linear:  63 → 64
   - Right Hand Linear: 63 → 64
   - Pose Linear:       99 → 128
   - Face Linear:       1,434 → 256
               │
               ▼
   [ Temporal Modeling ]
   - Bidirectional GRU (hidden_dim=128, num_layers=1, dropout=0.2)
   - Dynamic Modality Gating Layer (attention weights per stream)
               │
               ▼
   [ Classifier Head ]
   - Linear: 256 → 100 logits
   - Softmax with Top-1 & Top-5 Ranking
```

---

## 3. Dataset & Split Audit: Leakage Analysis

### 3.1 Sample Counts & Class Representation
- **Total Samples in `data/manifests/full_manifest.json`**: 21,083
- **Total Local Feature Files (`data/landmarks/*.npz`)**: 21,083 (0 missing files)
- **Official WLASL Test Manifest (`data/manifests/test_manifest.json`)**:
  - Contains 2,878 samples across all 2,000 WLASL classes.
  - Exactly **257 samples** belong to the 100 classes specified in `configs/experiments/wlasl100_proposed.yaml`.
  - All 257 sample landmark files exist and are valid.

### 3.2 Protocol A Signer Leakage
An audit of signer identities in the official WLASL splits reveals:
- **Train Signers**: 117
- **Validation Signers**: 109
- **Test Signers**: 105
- **Signer Overlap between Train and Test**: **104 signers (99.0%)**!
- *Conclusion*: Protocol A accuracy measures performance on signers that the model also saw during training. It does not measure genuine out-of-distribution signer-independent generalization.

### 3.3 Protocol B Signer-Disjoint Architecture
- Generated in `data/manifests/signer_disjoint/`:
  - Train: 14,757 samples (87 signers)
  - Validation: 3,163 samples (16 signers)
  - Test: 3,163 samples (16 signers)
  - **Signer Overlap**: Strictly **0** across all pairwise splits.
- *Scientific Policy*: Protocol A and Protocol B must be evaluated and reported separately.

---

## 4. Identified Pipeline Defects & Remediation Status

| Defect / Risk | Impact | Status | Remediation |
| :--- | :--- | :---: | :--- |
| **Windows Application Control Scipy DLL Failure** | Calling `scripts/evaluate.py` crashed on Windows due to OS security policy blocking `levyst` C DLL in `scipy.stats`. | **FIXED** | Built pure-NumPy classification metrics module (`src/sign_language/training/metrics.py`) providing zero-dependency mathematical equivalence. |
| **Silent Sample Drop Risk** | Incomplete manifests or missing files were previously ignored. | **FIXED** | Added `missing_policy="strict"` in dataset loaders, aborting with actionable diagnostics. |
| **Vocabulary Permutation Drift** | No cryptographic check ensuring label order between training and serving. | **FIXED** | Implemented `CanonicalVocabulary` with deterministic SHA256 vocabulary hash verification. |
| **MediaPipe Pipeline Lazy Loading** | `LandmarkPipeline` previously instantiated MediaPipe eagerly, adding 10-15s overhead to test runs. | **FIXED** | Extractor converted to lazy `@property`, eliminating unnecessary initialization. |
| **WebSocket Frame Desynchronization** | 55ms arbitrary unlock timer caused frame drops and stale predictions. | **FIXED** | Implemented monotonic `frame_id`, bounding boxes, and terminal `frame_processed` acknowledgement. |
