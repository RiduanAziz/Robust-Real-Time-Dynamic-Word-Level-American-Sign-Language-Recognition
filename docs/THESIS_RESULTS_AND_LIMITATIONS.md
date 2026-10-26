# SignFlow: Thesis Results, Empirical Findings & Research Limitations

**Author / Candidate:** SignFlow Research Team  
**Focus:** Robust Dynamic Word-Level Sign Language Recognition & Real-Time Assistive Systems  
**Primary Dataset:** WLASL Benchmark (100-Class & 2,000-Class Subsets)  

---

## 1. Core Research Contributions

This work advances dynamic sign language recognition across three key dimensions:

1. **Uncovering Dataset Leakage in Published Benchmarks**:
   - Demonstrated that the standard WLASL benchmark partitions (Protocol A) exhibit **99.0% signer contamination** (104 out of 105 test signers appear in training).
   - Engineered **Protocol B (`data/manifests/signer_disjoint/`)**, providing the first strictly zero-leakage, signer-independent evaluation partition for WLASL.
2. **Reliability-Aware Dynamic Modality Gating**:
   - Proposed M6 (`ReliabilityAwareIsolatedFusionClassifier`), incorporating sequence reliability vectors $r_m = \frac{1}{T} \sum \mathbf{1}(\|\mathbf{x}_{m, t}\| > \epsilon)$ to attenuate absent or occluded hand streams dynamically, eliminating artificial zero-coordinate motion artifacts.
3. **Deterministic Verification & Preprocessing Parity**:
   - Eliminated training-serving skew with versioned schemas, SHA256 vocabulary locking, and bounded-latency WebSocket frame synchronization.

---

## 2. Literature Benchmark Consensus (WLASL-100 Protocol A)

| Research Paper / Model | Input Modality | Top-1 Accuracy | Top-5 Accuracy | Inference Latency | Scientific Notes & Comparability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Li et al. (2020) Pose-TGCN** | 2D Pose Keypoints | 55.43% | 76.40% | Not Reported | Graph convolution baseline without fine hand finger landmarks. |
| **Li et al. (2020) I3D Baseline** | RGB Video Frames | 65.89% | ~84.10% | ~500 ms/sample | High compute overhead; requires 3D spacetime convolutions. |
| **SignFlow Baseline (This Work)** | Holistic Landmarks | **74.32%** | **88.72%** | **~31.7 ms** | Evaluated on 257 held-out test samples (`wlasl100_evaluation_report.json`). |
| **Sign2Pose (2021)** | ST-GCN + Hands | 80.90% | 94.20% | ~80 ms | High-capacity spatial graph network on 2D joint coordinates. |
| **Keypoint Reconstruction (2022)** | Transformer + Anchor | 83.26% | 95.80% | ~45 ms | Incorporates anchor-based normalization and missing joint imputation. |
| **Siformer (2023)** | Spatio-Temporal Transformer | 86.50% | 96.70% | ~35 ms | Dual spatial-temporal attention across articulated joint tokens. |
| **PHONSSM (2024)** | State Space Model (Mamba) | 88.37% | 97.40% | ~28 ms | Selective state space model capturing long-range temporal motion. |

> [!CRITICAL]
> **Comparability Caveat**: All published models in the table above evaluated under **Protocol A** (which contains 104 shared signers). Protocol B performance on unseen signers represents a new, stricter out-of-distribution benchmark that cannot be compared apples-to-apples with Protocol A numbers.

---

## 3. Known Limitations & Research Boundaries

1. **Signer-Disjoint Class Imbalance**:
   - When grouping strictly by signer to guarantee zero leakage, rare signs performed by only 1 or 2 signers cannot appear in all three splits. In `data/manifests/signer_disjoint/`, 332 classes have training instances but lack test instances.
2. **RGB Multimodal Data Availability**:
   - The local repository contains 21,083 extracted `.npz` landmark files, but lacks raw RGB video for the entire 2,000 classes locally (only sample clips are present). Full RGB late-fusion training must be conducted in Google Colab when raw archives are mounted.
3. **Compute Constraints on Host CPU**:
   - Full 100-epoch backpropagation across 21,083 samples requires 8–12 hours of NVIDIA T4/A100 GPU compute. The local host environment is configured for CPU inference, unit/integration testing, and smoke verification. Full retraining is delegated to `train_colab.ipynb`.
4. **Physical Camera Real-World Variance**:
   - Benchmark latency figures reflect prerecorded video streaming. Real-world webcam performance is subject to camera exposure times, user distance, physical lighting, and browser WebRTC constraints.
