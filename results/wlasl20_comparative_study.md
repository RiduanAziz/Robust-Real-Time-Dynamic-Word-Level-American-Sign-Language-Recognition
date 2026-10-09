# Empirical Thesis Comparison: Baseline vs. Proposed Architecture on WLASL-20

## 1. Experimental Protocol
- **Vocabulary (WLASL-20)**: 20 high-frequency ASL words (`before`, `thin`, `cool`, `drink`, `go`, `computer`, `who`, `cousin`, `help`, `candy`, `thanksgiving`, `bed`, `bowling`, `tall`, `accident`, `short`, `yes`, `what`, `later`, `man`).
- **Signer Independence**: Mutually disjoint signers across train (122 samples), validation (74 samples), and held-out test (78 samples). Zero signer leakage.
- **Input Representation**: Canonical 553-point holistic landmarks with 3D coordinate normalization, uniform temporal resampling (64 frames), boundary-safe velocity/acceleration dynamics (4,977 input dimensions).
- **Epochs**: 10 epochs with identical optimizer (AdamW, lr=0.001) and batch size 16.

## 2. Model Performance on Held-Out Test Signers

| Metric | Baseline (Temporal Transformer) | Proposed (RobustHolisticFusionClassifier) | Relative Improvement |
|---|---|---|---|
| **Top-1 Accuracy** | 0.1026 (10.26%) | **0.1923 (19.23%)** | **+87.5%** |
| **Macro-F1** | 0.0768 | **0.2015** | **+162.3%** |
| **Weighted-F1** | 0.0755 | **0.1752** | **+132.1%** |
| **Best Val Macro-F1** | 0.1108 | **0.2159** | **+94.9%** |
| **Final Train Loss** | 2.2648 | **1.6061** | - |

## 3. Spatial and Temporal Robustness Evaluation

Degradation is defined as Clean Accuracy - Noisy Accuracy (lower degradation is better; negative degradation indicates noise resistance/augmentation benefit).

| Disturbance Type | Severity | Baseline Noisy Acc | Baseline Degradation | Proposed Noisy Acc | Proposed Degradation | Robustness Advantage |
|---|---|---|---|---|---|---|
| `spatial_coordinate_jitter` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_coordinate_jitter` | 0.1 | 0.0897 | 0.0128 | 0.1923 | 0.0000 | **+0.0128** |
| `spatial_coordinate_jitter` | 0.3 | 0.0769 | 0.0256 | 0.1923 | 0.0000 | **+0.0256** |
| `spatial_coordinate_jitter` | 0.5 | 0.0769 | 0.0256 | 0.1923 | 0.0000 | **+0.0256** |
| `spatial_landmark_dropout` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_landmark_dropout` | 0.1 | 0.0769 | 0.0256 | 0.1410 | 0.0513 | **-0.0256** |
| `spatial_landmark_dropout` | 0.3 | 0.1154 | -0.0128 | 0.0897 | 0.1026 | **-0.1154** |
| `spatial_landmark_dropout` | 0.5 | 0.0769 | 0.0256 | 0.0897 | 0.1026 | **-0.0769** |
| `spatial_scale` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_scale` | 0.1 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_scale` | 0.3 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_scale` | 0.5 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_translation` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `spatial_translation` | 0.1 | 0.1026 | 0.0000 | 0.1795 | 0.0128 | **-0.0128** |
| `spatial_translation` | 0.3 | 0.0897 | 0.0128 | 0.1795 | 0.0128 | **-0.0000** |
| `spatial_translation` | 0.5 | 0.1026 | 0.0000 | 0.1795 | 0.0128 | **-0.0128** |
| `temporal_frame_drop` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `temporal_frame_drop` | 0.1 | 0.0513 | 0.0513 | 0.1923 | 0.0000 | **+0.0513** |
| `temporal_frame_drop` | 0.3 | 0.0641 | 0.0385 | 0.1923 | 0.0000 | **+0.0385** |
| `temporal_frame_drop` | 0.5 | 0.1026 | 0.0000 | 0.2051 | -0.0128 | **+0.0128** |
| `temporal_frame_duplicate` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `temporal_frame_duplicate` | 0.1 | 0.0769 | 0.0256 | 0.1923 | 0.0000 | **+0.0256** |
| `temporal_frame_duplicate` | 0.3 | 0.0769 | 0.0256 | 0.1795 | 0.0128 | **+0.0128** |
| `temporal_frame_duplicate` | 0.5 | 0.0897 | 0.0128 | 0.1923 | 0.0000 | **+0.0128** |
| `temporal_sequence_truncate` | 0.0 | 0.1026 | 0.0000 | 0.1923 | 0.0000 | **0.0000** |
| `temporal_sequence_truncate` | 0.1 | 0.0641 | 0.0385 | 0.2051 | -0.0128 | **+0.0513** |
| `temporal_sequence_truncate` | 0.3 | 0.0513 | 0.0513 | 0.1667 | 0.0256 | **+0.0256** |
| `temporal_sequence_truncate` | 0.5 | 0.0769 | 0.0256 | 0.2051 | -0.0128 | **+0.0385** |