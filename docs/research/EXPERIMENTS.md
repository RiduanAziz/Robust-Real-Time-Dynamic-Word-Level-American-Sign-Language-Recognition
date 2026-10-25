# SignFlow Experiments & Literature Benchmarks

This document records the experimental results, controlled benchmark comparisons, parameter counts, ablation studies, and literature consensus for the SignFlow ASL recognition system.

---

## 1. Controlled Model Benchmark Suite

All models evaluated under uniform conditions: WLASL-100 vocabulary ($C = 100$), sequence length $T = 64$, batch size 32, AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$).

| Model Architecture | Model Identifier | Input Modality | Parameters | Checkpoint Size | Protocol A Top-1 (%) | Protocol A Macro-F1 | Protocol B Top-1 (%) | Status | Provenance / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Robust Holistic Fusion (Baseline)** | `robust_holistic_fusion` | Holistic (Hands+Pose+Face) | ~1.06 M | 4.05 MB | **74.32%** | **0.7213** | *Pending GPU Run* | **Evaluated** | Baseline preserved in `models/robust_holistic_fusion_best.pt`; 257 test samples. |
| **Lightweight Pooled Baseline (M1)** | `pooled_baseline` | Hands + Pose | ~0.24 M | ~0.95 MB | *Smoke Tested* | *Smoke Tested* | *Pending GPU Run* | **Smoke Tested** | Fast convergence, lower bound on temporal modeling. |
| **Temporal Convolutional Network (M2)** | `tcn` | Hands + Pose (Dilated 1D) | ~0.58 M | ~2.31 MB | *Smoke Tested* | *Smoke Tested* | *Pending GPU Run* | **Smoke Tested** | Fully implemented in `src/sign_language/models/tcn.py`, config in `configs/experiments/tcn.yaml`. |
| **Feature-Isolated Temporal Transformer (M5)** | `temporal_transformer` | Hands + Pose (Attention) | ~1.85 M | ~7.40 MB | *Smoke Tested* | *Smoke Tested* | *Pending GPU Run* | **Smoke Tested** | Multi-head self-attention with causal sequence masks. |
| **Reliability-Aware Isolated Fusion (M6, Proposed)** | `reliability_aware_fusion` | Hands + Pose (Dynamic Gate) | ~1.42 M | ~5.68 MB | *Smoke Tested* | *Smoke Tested* | *Pending GPU Run* | **Smoke Tested** | Fully implemented in `src/sign_language/models/reliability_fusion.py`; masks prevent missing-hand corruption. |

---

## 2. Literature Comparison (WLASL-100 Benchmark)

Published research reference points on the WLASL-100 benchmark. Note: All published works listed below evaluated under **Protocol A (Official WLASL splits)**. None of the published literature below evaluated on a strict signer-disjoint protocol.

| Study / Citation | Architecture | Input Modality | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Notes & Caveats |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Li et al. (2020)**<br>*WLASL Dataset & Benchmark* | I3D Baseline | RGB Video Frames | 65.89% | ~84.1% | Original WLASL benchmark; heavy computational cost (~0.5s/clip). |
| **Li et al. (2020)**<br>*WLASL Dataset & Benchmark* | Pose-TGCN | Pose / Keypoints only | 55.43% | 76.40% | Graph convolution baseline without holistic hand-finger articulation. |
| **Sign2Pose (2021)** | ST-GCN + Hand Encoders | Pose + Hands | 80.90% | 94.20% | High-capacity graph convolutional network on 2D keypoints. |
| **Keypoint Reconstruction (2022)** | Transformer + Anchor Norm | Pose + Hands | 83.26% | 95.80% | Uses anchor-based normalization and missing keypoint reconstruction. |
| **Siformer (2023)** | Spatio-Temporal Transformer | Holistic Keypoints | 86.50% | 96.70% | Spatial and temporal self-attention across joints and frames. |
| **PHONSSM (2024)** | State Space Model (SSM) | Graph Keypoints | 88.37% | 97.40% | Mamba / SSM-based long-range temporal sign language sequence modeling. |
| **SignFlow Baseline (This Work)** | `RobustHolisticFusionClassifier` | Holistic Keypoints (1659-dim) | **74.32%** | **91.83%** | Evaluated on 257 held-out test samples (`wlasl100_evaluation_report.json`). |

---

## 3. Ablation Framework

The repository is equipped to run systematic ablations across the following axes:

### Ablation 1: Input Representations
- **Rep A (Hands Only, 126-dim)** vs **Rep B (Hands + Pose, 225-dim)** vs **Rep C (Holistic, 1659-dim)** vs **Rep D (Engineered Features, 257-dim)**.
- *Hypothesis*: Rep D provides wrist-relative invariance, reducing sensitivity to camera distance while retaining fine finger flexion.

### Ablation 2: Missing Keypoint Handling
- **Strategy 1**: Zero-filling (baseline, treats origin as active joint).
- **Strategy 2**: Linear short-gap interpolation ($\le 3$ frames) + zero-padding without interpolation for long occlusions.
- **Strategy 3**: Mask-aware feature isolation (Proposed M6, dynamically weights observed streams by reliability).

### Ablation 3: Modality Fusion Mechanism
- **Concatenation Fusion** vs **Static Weighted Sum** vs **Reliability-Aware Dynamic Gating**.
