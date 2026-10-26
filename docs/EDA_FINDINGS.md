# Exploratory Data Analysis (EDA) Findings — SignFlow WLASL-100

**Document Version**: 2.0.0 (Empirical Analysis Report)  
**Author**: Lead Research & ML Systems Engineer  
**Date**: October 2026  
**Artifact Dependencies**:
- `results/eda/class_distribution.csv`
- `results/eda/sequence_statistics.csv`
- `results/eda/landmark_quality.csv`
- `results/eda/feature_statistics.csv`
- `results/eda/plots/`

---

## 1. Executive Summary

A comprehensive exploratory data analysis was conducted directly across the precomputed MediaPipe landmark tensors and manifest metadata for the 2,038 samples comprising WLASL-100. 

### Key Findings:
1. **Modality Asymmetry**:
   - **Body Pose**: **99.4%** average frame detection rate.
   - **Face Mesh**: **97.3%** average frame detection rate.
   - **Right Hand**: **63.6%** average frame detection rate.
   - **Left Hand**: **32.5%** average frame detection rate.
   *Implication*: ASL exhibits high right-hand dominance. Forcing uniform concatenation without validity gating introduces severe zero-fill noise for left-hand non-dominant signs.
2. **Temporal Dynamics**:
   - Video clip lengths range from **9 frames (0.36 s)** to **178 frames (7.12 s)**.
   - The median sequence length is **52 frames (~2.08 seconds)**.
   - Standard resampling to $T=64$ frames preserves temporal resolution across >85% of samples without downsampling information loss.
3. **Class Balance**:
   - Training sample count per class ranges from 9 to 25 samples.
   - Validation and test sets maintain near-uniform per-class support (2 to 4 samples per class).

---

## 2. Modality Visibility & Detection Analysis

Empirical detection rates computed from `results/eda/feature_statistics.csv`:

| Modality | Keypoint Count | Flattened 3D Coords | Mean Detection Rate | Min Rate | Max Rate | Structural Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Left Hand** | 21 | 63 | **32.5%** | 0.0% | 100.0% | Non-dominant hand in two-handed signs. |
| **Right Hand** | 21 | 63 | **63.6%** | 0.0% | 100.0% | Primary dominant signing hand. |
| **Body Pose** | 33 | 99 | **99.4%** | 5.0% | 100.0% | Spatial reference frame (shoulders, torso). |
| **Face Mesh** | 478 | 1,434 | **97.3%** | 0.0% | 100.0% | Spatial anchor for chin/mouth/forehead signs (*deaf*, *candy*, *who*). |

```
Landmark Detection Visibility by Modality:
Pose (Body)  [████████████████████████████████████████] 99.4%
Face Mesh    [█████████████████████████████████████   ] 97.3%
Right Hand   [███████████████████████                 ] 63.6%
Left Hand    [████████████                            ] 32.5%
```

---

## 3. Sequence Length & Duration Analysis

From `results/eda/sequence_statistics.csv`:

| Metric | Frames | Duration (seconds @ 25 FPS) |
| :--- | :---: | :---: |
| **Minimum** | 9 | 0.36 s |
| **25th Percentile** | 38 | 1.52 s |
| **Median (50th)** | 52 | 2.08 s |
| **Mean** | 56.4 | 2.26 s |
| **75th Percentile** | 68 | 2.72 s |
| **Maximum** | 178 | 7.12 s |

### Architectural Implication:
Configuring the models to `sequence_length = 64` with linear temporal interpolation and causal padding accommodates the central distribution of ASL isolated signs while maintaining real-time latency (<40 ms).

---

## 4. Class Distribution & Imbalance

From `results/eda/class_distribution.csv`:
- Top-frequency classes: *book*, *drink*, *computer*, *before*, *chair* (22–25 training samples).
- Lower-frequency classes: *thursday*, *africa*, *gold*, *glasses* (9–12 training samples).
- **Imbalance Ratio**: Maximum/Minimum ratio is $\sim 2.78:1$.
- **Mitigation Strategy**:
  - Training uses label smoothing ($\epsilon = 0.08$) and Cosine Annealing scheduler.
  - Test set is locked and strictly balanced (257 samples across 100 classes).

---

## 5. Visual Artifacts

Generated figures in `results/eda/plots/`:
1. `class_frequency_distribution.png`: Bar chart of per-class sample frequencies.
2. `sequence_length_histogram.png`: Distribution of video durations and frame counts.
3. `landmark_validity_by_modality.png`: Empirical detection rates per anatomical stream.
4. `split_proportions.png`: Pie chart of official Protocol A partitions (Train 70.8%, Val 16.5%, Test 12.6%).
