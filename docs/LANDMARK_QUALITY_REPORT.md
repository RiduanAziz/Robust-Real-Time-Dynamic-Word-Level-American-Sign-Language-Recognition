# SignFlow: Landmark Quality & Extraction Diagnostic Report

**Audit Date:** October 26, 2026  
**Extractor Asset:** `models/mediapipe/holistic_landmarker.task`  
**MediaPipe Version:** MediaPipe 1.0.1  

---

## 1. Landmark Schema & Coordinate Conventions

SignFlow processes human upper-body kinematics via Google MediaPipe Holistic Landmarker:

| Kinematic Stream | Joint Count | Coordinates per Joint | Flattened Dims | Coordinate Conventions |
| :--- | :---: | :---: | :---: | :--- |
| **Left Hand** | 21 keypoints | $(x, y, z)$ | 63 | Normalized $[0, 1]$ image space; $z$ represents relative depth. |
| **Right Hand** | 21 keypoints | $(x, y, z)$ | 63 | Normalized $[0, 1]$ image space; $z$ represents relative depth. |
| **Pose (Upper Body)** | 33 keypoints | $(x, y, z)$ | 99 | Body joints including shoulders, elbows, wrists, hips. |
| **Face Mesh** | 478 keypoints | $(x, y, z)$ | 1,434 | Detailed facial contours and lip articulation. |
| **Total Holistic** | **553 keypoints** | **$(x, y, z)$** | **1,659** | Full multi-stream holistic observation per frame. |

---

## 2. Missing-Data & Occlusion Handling Strategy

In real-world signing, one-handed signs or rapid out-of-frame gestures cause temporary tracking dropouts. Interpreting an absent hand as coordinates at the origin $(0, 0, 0)$ introduces severe artificial motion vectors that deceive temporal neural networks.

SignFlow enforces an explicit missing-data contract (`src/sign_language/landmarks/missing_data.py`):

1. **Explicit Validity Masks**:
   Every frame maintains a boolean vector $\mathbf{m}_t \in \{0, 1\}^D$ distinguishing actively observed keypoints from missing keypoints.
2. **Conservative Short-Gap Interpolation**:
   - For temporary dropouts of length $\Delta t \le 3$ frames (e.g. rapid motion blur), linear interpolation reconstructs the trajectory between confident keyframes:
     $$\mathbf{x}_t = \mathbf{x}_{t_{\text{prev}}} + \frac{t - t_{\text{prev}}}{t_{\text{next}} - t_{\text{prev}}} \left(\mathbf{x}_{t_{\text{next}}} - \mathbf{x}_{t_{\text{prev}}}\right)$$
   - Gaps longer than 3 frames are **never hallucinated**. Long gaps remain marked as invalid, and the proposed Reliability-Aware fusion layer (M6) dynamically downweights the absent modality.
3. **Dynamics Calculation Protection**:
   Velocities and accelerations are never computed across missing or discontinuous frames, preventing spurious extreme acceleration spikes.

---

## 3. Evaluated Input Representations

SignFlow supports four distinct geometric and kinematic representations:

- **Representation A (Hands Only, $D = 126$)**:
  - Left hand ($21 \times 3$) and right hand ($21 \times 3$). Minimal computation, but loses arm posture and shoulder orientation.
- **Representation B (Hands + Body Pose, $D = 225$)**:
  - Hands ($42 \times 3$) plus 33 body pose keypoints. Captures arm angle and torso anchoring without facial mesh overhead.
- **Representation C (Holistic Full, $D = 1,659$ static / $4,977$ dynamic)**:
  - Hands, pose, and face, with optional first-order velocity ($\Delta \mathbf{x}$) and second-order acceleration ($\Delta^2 \mathbf{x}$).
- **Representation D (Engineered Features, $D = 257$)**:
  - Implemented in `src/sign_language/features/representations.py`.
  - Computes wrist-relative joint coordinates (scale and translation invariant).
  - Computes 3D Euclidean inter-wrist distance (measuring two-hand proximity).
  - Computes finger flexion and extension angles across MCP, PIP, and DIP joints.

---

## 4. Extraction Failure & Latency Profile

Empirical benchmarking (`results/realtime_benchmark.json`) reveals:
- **Mean Extraction Latency on CPU**: ~0.7 to 1.5 ms per frame once initialized ($480\times360$).
- **Cold-Start Overhead**: First-frame MediaPipe kernel initialization requires ~500–2,000 ms on CPU. Subsequent frames stream steadily at 25–32 FPS.
- **Dimension Caching Edge Case**: MediaPipe's `SegmentationSmoothingCalculator` caches image dimensions. Switching resolutions dynamically requires extractor re-instantiation to prevent dimension check assertions.
