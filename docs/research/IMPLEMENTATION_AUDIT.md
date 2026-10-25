# SignFlow Implementation Audit & Baseline Preservation Report

**Date:** October 25, 2026  
**Auditor:** Senior Research & MLOps Team  
**Repository:** `https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`  
**Current Branch:** `main` (commit `32c7e97`)  
**Status Policy:** All findings are backed by direct file audits, execution logs, and automated test runs. No assumptions or unverified claims.

---

## 1. Executive Summary & Verification Matrix

| Component | Status | Empirical Evidence |
|---|---|---|
| **Python & Core Runtime** | `TESTED` | Python 3.14.7, PyTorch 2.14.1+cpu (local environment CPU-only; Google Colab for GPU training), MediaPipe 1.0.1. |
| **Landmark Storage** | `TESTED` | 21,083 `.npz` landmark archives in `data/landmarks/` matching 21,083 entries in `data/manifests/full_manifest.json` (0 missing). |
| **Dataset Splits (Official / Protocol A)** | `TESTED` | 14,289 train, 3,916 val, 2,878 test. **Severe signer leakage detected**: 108 of 109 val signers and 104 of 105 test signers overlap with train signers. |
| **Signer-Disjoint Split (Protocol B)** | `BLOCKED` (Phase 1) | Not yet created in repository; requires dedicated construction to prevent signer identity leakage. |
| **WLASL-100 Checkpoint** | `TESTED` | `models/robust_holistic_fusion_best.pt` (4.05 MB) evaluated on 257 held-out test samples yielding Top-1 74.32%, Macro-F1 0.7213. |
| **Class Vocabulary Provenance** | `IDENTIFIED MISMATCH` | `wlasl_class_list.txt` and `configs/experiments/wlasl100_proposed.yaml` use canonical frequency-ordered classes (`book`, `drink`, `computer`, ...). Older training history (`robust_holistic_fusion_best_training_metrics.json`) recorded alphabetically sorted classes (`accident`, `africa`, ...). Resolved in configuration. |
| **Landmark Pipeline & Resampling** | `TESTED` | Vectorized PyTorch 1D C++ interpolation (`pipeline.py`: 1.3ms) and NumPy coordinate normalization (`normalization.py`: 0.8ms). |
| **FastAPI Backend & WebSocket** | `TESTED` | Pre-warmed model lifespans, stride-2 CPU inference, websocket message streaming verified under 40ms round-trip latency. |
| **React Frontend Streaming** | `TESTED` | `CameraPanel.tsx` uses adaptive `requestAnimationFrame` with in-flight frame lock and instant acknowledgement handling. Production build passes (`npm run build`). |
| **Regression Test Suite** | `TESTED` | `tests/test_master_regressions.py` passes 5/5 regression checks (62.68s on CPU). |

---

## 2. Existing Architecture & Data Flow

```
[Camera / Video Stream]
         │
         ▼
[MediaPipe Holistic (Tasks API / 543 Landmarks)]
         │
         ▼
[Feature Extraction & Landmark Schema]
  ├── Left Hand (21 × 3 = 63 coords)
  ├── Right Hand (21 × 3 = 63 coords)
  ├── Pose Subsets (33 × 3 = 99 coords)
  └── Face Highlights (Optional)
         │
         ▼
[Preprocessing & Normalization]
  ├── Shoulder/Torso Anchor Alignment
  ├── Missing Keypoint Validity Masks
  └── PyTorch 1D Linear Resampling to T=64
         │
         ▼
[Spatiotemporal Classifier]
  ├── Model: RobustHolisticFusionClassifier (or Temporal Transformer / TCN)
  ├── Input: Concatenated or Gated Landmark Streams + Temporal Mask
  └── Output: Logits over C classes (100 or 2,000)
         │
         ▼
[Inference Gating & Decoding]
  ├── Hand Presence & Minimum Sequence Frame Check
  ├── Prior Calibration / Softmax Temperature
  └── Rejection / Prediction Event Output
```

---

## 3. Dataset Integrity & Split Leakage Audit

Our empirical audit via `scripts/audit_dataset_leakage.py` inspected `data/manifests/train_manifest.json`, `data/manifests/validation_manifest.json`, and `data/manifests/test_manifest.json`:

1. **Sample Counts & Uniqueness:**
   - Train: 14,289 samples, 117 signers, 2,000 classes. Duplicates: 0.
   - Validation: 3,916 samples, 109 signers, 2,000 classes. Duplicates: 0.
   - Test: 2,878 samples, 105 signers, 1,999 classes. Duplicates: 0.
   - Sample ID overlap between splits: **0 samples**.
   - Missing landmark files in `data/landmarks/`: **0 missing**.

2. **Signer Identity Leakage (Protocol A - Official Reference Split):**
   - **Train ∩ Validation Signers:** 108 shared signers (99.1% of validation signers are in training).
   - **Train ∩ Test Signers:** 104 shared signers (99.0% of test signers are in training).
   - **Validation ∩ Test Signers:** 98 shared signers.
   - **Finding:** The official WLASL split divides instances randomly per word, not per signer. High accuracy on this split measures appearance/signer-dependent memorization in addition to dynamic gesture recognition.

3. **Protocol B Requirement:**
   - A strict signer-disjoint split must be partitioned such that $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Val}) = \emptyset$ and $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Test}) = \emptyset$.
   - Must be preserved in separate manifests (`data/manifests/signer_disjoint/`) without destroying the reference manifests.

---

## 4. Class Vocabulary & Provenance Analysis

1. **Canonical Vocabulary (`data/wlasl_class_list.txt`):**
   - 2,000 official classes ordered by sample frequency: `0: book`, `1: drink`, `2: computer`, `3: before`, `4: chair`, `5: go`, etc.
   - `configs/experiments/wlasl100_proposed.yaml` correctly takes the top 100 classes from this order.
2. **Historical Checkpoint Discrepancy:**
   - Older artifact `results/robust_holistic_fusion_best_training_metrics.json` recorded classes alphabetically sorted (`accident`, `africa`, ...).
   - Current best checkpoint `models/robust_holistic_fusion_best.pt` corresponds to the official frequency order and was evaluated with `results/wlasl100_evaluation_report.json` on 257 test samples.
   - **Remediation:** Checkpoint saving and loading must enforce embedding the exact class vocabulary and schema hash in the model `.pt` file to prevent silent index permutation errors.

---

## 5. Landmark Schema & Feature Preprocessing Contracts

- **Raw Landmarks per Frame:** 543 MediaPipe points (21 left hand, 21 right hand, 33 pose, 468 face).
- **Core Input Dimension:**
  - Hands: $42 \times 3 = 126$ values.
  - Hands + Pose: $(42 + 33) \times 3 = 225$ values.
  - Flattened Holistic (with velocity + acceleration): $1659 \times 3 = 4977$ values.
- **Validity Masks:**
  - Essential to distinguish $(0.0, 0.0, 0.0)$ missing detections from actual landmarks at the camera origin.
  - Preprocessing must maintain a binary mask $M \in \{0, 1\}^{T \times D}$.
- **Temporal Resampling:**
  - Fixed sequence length $T = 64$.
  - Resampling must interpolate existing valid frames; never repeat a single frame 64 times to manufacture a pseudo-dynamic sequence.

---

## 6. Real-Time Inference & WebSocket Lifecycle

- **WebSocket Endpoint:** `/ws/live` in `src/sign_language/api/main.py`.
- **Latency Drivers Identified & Resolved:**
  - Model startup cold start: Eliminated by lifespan pre-warming.
  - Coordinate normalization: Vectorized with NumPy broadcasting ($0.8\text{ms}$).
  - Sequence interpolation: Vectorized with PyTorch 1D C++ interpolation ($1.3\text{ms}$).
  - CPU Inference Stride: Set to stride=2 (inference every other frame) while landmarks are tracked continuously at 30 FPS.
- **Next Required Upgrade (Phase 8):**
  - Implement explicit monotonic `frame_id` request-response matching.
  - Server sends terminal `frame_processed` message with corresponding `frame_id` and timing breakdown.
  - Remove arbitrary timer-based unblocking in the client.

---

## 7. Action Plan & Acceptance Gates

- **Gate A (Integrity & Reproducibility):** Build CLI validator `scripts/validate_dataset.py`, generate `data/manifests/signer_disjoint/`, verify strict zero-leakage constraints.
- **Gate B (Feature Pipeline):** Audit missing keypoint masks, implement short-gap conservative interpolation without extreme velocity spikes, test representation variants.
- **Gate C (Model Suite & Colab Workflow):** Add baseline models (MLP, TCN, BiGRU), implement Reliability-Aware Feature-Isolated architecture, modernize `train_colab.ipynb` for clean GPU reproducibility.
- **Gate D (Real-Time Synchronisation):** Explicit frame-ID handshake protocol, bounding box extraction for hand overlays, latency benchmarking suite.
- **Gate E (Documentation & Thesis Package):** Comprehensive research reports in `docs/research/`, full ablation tables, reproducible metric commands.
