# Feature Engineering Audit and Results — SignFlow WLASL-100

**Document Version**: 2.0.0 (Feature Contracts & Spatial Representations)  
**Author**: Lead Research & ML Systems Engineer  
**Date**: October 2026  
**Artifact Dependencies**:
- `src/sign_language/features/representations.py`
- `src/sign_language/landmarks/schema.py`
- `tests/test_features_contract.py`

---

## 1. Executive Summary & Defect Resolutions

This document provides the authoritative engineering specification and verification results for all landmark feature representations in SignFlow.

### 1.1 Resolution of Dimensionality Discrepancies
- **Previous Contradiction**: `_FEATURES["engineered"]` declared `feature_dim = 280` in `representations.py`, but the actual implementation concatenated:
  $$\text{Hands+Pose (225)} + \text{Left Wrist-Rel (63)} + \text{Right Wrist-Rel (63)} + \text{Inter-Wrist (4)} + \text{Fingertip Extensions (10)} = 365$$
  This 85-dimension contradiction caused silent padding/slicing errors in downstream models expecting 280 features.
- **Resolution**:
  - The specification has been officially updated to **`feature_dim = 365`**.
  - Strict mask propagation has been implemented for all derived features.
  - Zero-filling is enforced when any required keypoint is missing.
  - An automated regression test suite (`tests/test_features_contract.py`) validates that declared dimensions and actual outputs match 100%.

---

## 2. Supported Feature Representations (Representations A–F)

| Rep ID | Representation Name | Feature Dim | Input Modalities | Description & Formulation |
| :---: | :--- | :---: | :--- | :--- |
| **A** | `hands` | **126** | Left Hand (63), Right Hand (63) | 21 3D coordinates per hand ($21 \times 3 \times 2 = 126$). |
| **B** | `hands_pose` | **225** | Hands (126), Body Pose (99) | Hands plus 33 body pose landmarks ($33 \times 3 = 99$). |
| **C** | `holistic` | **1,659** | Hands (126), Pose (99), Face (1434) | Full MediaPipe holistic mesh (553 3D keypoints). Essential for face-anchored signs. |
| **D** | `engineered` | **365** | Hands (126), Pose (99) + Spatial Relations | Hands+Pose (225) + Wrist-rel (126) + Inter-wrist distance (4) + Fingertip extensions (10). |
| **E** | `holistic_dynamics` | **4,977** | Full Holistic + Velocity + Acceleration | Positions ($1659$) + First-order differences ($1659$) + Second-order differences ($1659$). |
| **F** | `anatomical_streams` | Multi-stream | Isolated LH, RH, Pose, Face | Stream-isolated tensor inputs for Prior-Guided Fusion (`UniSignPGFClassifier`). |

---

## 3. Strict Validity Mask Propagation

Derived kinematic and spatial features must propagate missing landmark masks rather than falsely reporting valid data for zero-filled values:

1. **Wrist-Relative Coordinates (63 dims per hand)**:
   $$\mathbf{p}_{i}^{\text{rel}} = \mathbf{p}_{i} - \mathbf{p}_{\text{wrist}}, \quad m_{i}^{\text{rel}} = m_{i} \times m_{\text{wrist}}$$
   *Rule*: If the wrist is undetected, all wrist-relative coordinates for that hand are masked out ($m = 0.0$) and zero-filled.
2. **Inter-Wrist Displacement & Distance (4 dims)**:
   $$\mathbf{d} = \mathbf{p}_{\text{rh\_wrist}} - \mathbf{p}_{\text{lh\_wrist}}, \quad d_{\text{norm}} = \|\mathbf{d}\|_2, \quad m_{\text{inter}} = m_{\text{lh\_wrist}} \times m_{\text{rh\_wrist}}$$
   *Rule*: Both wrists must be visible for inter-hand interaction features to be considered valid.
3. **Fingertip Extension Distances (10 dims)**:
   $$e_k = \|\mathbf{p}_{\text{tip}_k} - \mathbf{p}_{\text{wrist}}\|_2, \quad m_{e_k} = m_{\text{tip}_k} \times m_{\text{wrist}}$$
   *Rule*: Extension distance is only valid if both the fingertip (indices 4, 8, 12, 16, 20) and the wrist are detected.

---

## 4. Geometry-Aware Spatial Normalization

SignFlow rejects naive flat-vector mean-centering. Instead, spatial normalization is computed per-frame based on anatomically meaningful reference frames:
- **Torso / Shoulder Frame**: Shoulders (pose landmarks 11, 12) establish the horizontal scale and body center.
- **Local Hand Coordinates**: Each hand's internal geometry is preserved relative to its wrist coordinate.
- **Face Mesh Independence**: The 478 face landmarks are centered on the nose bridge / eye midpoint so their high point count does not dominate the global coordinate center of the signing space.

---

## 5. Automated Feature-Contract Test Results

Running `pytest tests/test_features_contract.py -v`:

```
tests/test_features_contract.py::test_feature_specifications_registered_and_dimensions_match PASSED [ 16%]
tests/test_features_contract.py::test_representation_a_hands_dimension_and_mask PASSED        [ 33%]
tests/test_features_contract.py::test_representation_b_hands_pose_dimension_and_mask PASSED   [ 50%]
tests/test_features_contract.py::test_representation_c_holistic_dimension_and_mask PASSED     [ 66%]
tests/test_features_contract.py::test_representation_d_engineered_strict_mask_propagation PASSED [ 83%]
tests/test_features_contract.py::test_representation_e_holistic_dynamics_temporal_boundaries PASSED [100%]

============================== 6 passed in 0.68s ==============================
```
