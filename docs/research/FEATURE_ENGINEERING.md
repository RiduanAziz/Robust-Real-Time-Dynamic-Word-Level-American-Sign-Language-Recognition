# SignFlow Feature Engineering & Preprocessing Specification

**Scope:** Landmark Schemas, Motion Representations, Normalization & Missing Data Strategy  
**Status:** Approved Research Specification  
**Author:** Senior Computer Vision & ML Engineering Team

---

## 1. Landmark Schemas & Coordinate Semantics

SignFlow relies on the MediaPipe Holistic perception model, which extracts up to 543 3D landmarks per video frame:

| Anatomical Subsystem | Landmark Count | Raw Coordinate Count ($X, Y, Z$) | Keypoints / Skeletons |
|---|---:|---:|---|
| **Left Hand** | 21 points | 63 floats | Wrist, Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20) |
| **Right Hand** | 21 points | 63 floats | Identical 21-joint kinematic chain |
| **Upper Body Pose** | 33 points | 99 floats | Nose, Eyes, Shoulders (11, 12), Elbows (13, 14), Wrists (15, 16), Hips (23, 24) |
| **Face Mesh** | 468 points | 1,404 floats | Facial contour, lips, eyebrows, emotional/grammatical markers |
| **Full Holistic** | 543 points | 1,659 floats | Combined hands, pose, and face stream |

---

## 2. Four Benchmark Representations

To determine empirically which input signals yield the highest signal-to-noise ratio, SignFlow implements four distinct representations in `src/sign_language/features/representations.py`:

### Representation A — Hands Only ($D = 126$)
- **Contents:** 21 Left Hand points + 21 Right Hand points.
- **Hypothesis:** High computational efficiency and direct focus on lexical handshapes. Weakness: Cannot disambiguate signs differentiated by body position (e.g., forehead vs. chin vs. chest).

### Representation B — Hands Plus Body Pose ($D = 225$)
- **Contents:** Left Hand (63) + Right Hand (63) + Body Pose (99).
- **Hypothesis:** Provides anatomical anchoring and spatial location relative to shoulders/chest while eliminating heavy facial mesh noise.

### Representation C — Full Holistic with Dynamics ($D = 4,977$)
- **Contents:** Position ($1,659$) + Velocity ($1,659$) + Acceleration ($1,659$).
- **Derivatives:** Computed via second-order finite differences with strict binary mask propagation:
  $$\mathbf{v}_t = (\mathbf{x}_t - \mathbf{x}_{t-1}) \cdot (m_t \cdot m_{t-1})$$
  $$\mathbf{a}_t = (\mathbf{v}_t - \mathbf{v}_{t-1}) \cdot (m_t \cdot m_{t-1} \cdot m_{t-2})$$

### Representation D — Engineered Spatial & Interaction Features ($D = 365$)
- **Contents:**
  1. Base Hands + Pose ($225$).
  2. Wrist-relative hand coordinates ($\mathbf{x}_i - \mathbf{x}_{\text{wrist}}$) for both hands ($126$).
  3. Inter-wrist 3D displacement vector and Euclidean distance ($\Delta \mathbf{x}_{\text{wrists}}, \|\Delta \mathbf{x}\|$) ($4$).
  4. Finger tip extension distances to wrist ($\|\mathbf{x}_{\text{tip}} - \mathbf{x}_{\text{wrist}}\|$ for 5 digits $\times$ 2 hands) ($10$).

---

## 3. Principled Missing-Keypoint Handling

A critical failure mode in earlier pipelines was interpreting $(0.0, 0.0, 0.0)$ missing detections as true coordinates located at the camera origin. SignFlow enforces:

1. **Explicit Binary Validity Masks:**
   - Every coordinate array $\mathbf{X} \in \mathbb{R}^{T \times D}$ is accompanied by a boolean mask $\mathbf{M} \in \{0, 1\}^{T \times D}$.
   - Confidence thresholding $\tau = 0.5$. Missing points remain zeroed with $m_{t, d} = 0$.
2. **Conservative Short-Gap Interpolation:**
   - Implemented in `src/sign_language/landmarks/missing_data.py`.
   - Gaps $\le 3$ consecutive frames between valid boundary detections are linearly interpolated.
   - Long gaps ($> 3$ frames) are **never** fabricated, preventing artificial trajectories.
3. **Spurious Velocity Suppression:**
   - Velocity across missing frames is forced to $0.0$, avoiding extreme acceleration spikes that destabilize neural network attention layers.

---

## 4. Sequence Normalization & Temporal Resampling

1. **Torso & Shoulder Anchor Alignment:**
   - Each frame is translated such that the midpoint between left and right shoulders (or the average valid landmark centroid if shoulders are occluded) is at the origin $(0, 0, 0)$.
   - Scale is normalized by the root-mean-square distance of visible keypoints to prevent scale distortion across varying distances to the camera.
2. **Vectorized Linear Resampling:**
   - Variable-length video sequences ($T \in [14, 180]$) are resampled to fixed length $T = 64$ via PyTorch 1D C++ interpolation (`F.interpolate`).
   - Runs in $1.3\text{ms}$ on CPU, preserving the full temporal dynamics across the sign duration.
   - **Prohibition:** Repeating a single static frame 64 times is strictly banned from training and evaluation.
