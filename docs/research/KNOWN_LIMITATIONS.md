# SignFlow Known Limitations & Scientific Disclosures

In accordance with strict research integrity standards, this document discloses known dataset constraints, structural limitations, hardware dependencies, and unresolved trade-offs in the current SignFlow system.

---

## 1. Dataset & Split Leakage Constraints

### 1.1 Signer Overlap in Official WLASL Splits (Protocol A)
- **Empirical Finding**: In the standard published WLASL splits (`train_manifest.json`, `validation_manifest.json`, `test_manifest.json`), **104 out of 105 signers (99.0%)** in the test split also appear in the training split.
- **Scientific Consequence**: Reported Top-1 test accuracies in the literature (e.g., 74.32%–88.37%) on Protocol A measure a mixture of sign semantic recognition and signer-specific memorization.
- **Remediation in SignFlow**: We engineered Protocol B (`data/manifests/signer_disjoint/`) with strictly zero signer overlap. Full multi-epoch GPU training on Protocol B is queued for Google Colab.

### 1.2 Class Imbalance Under Signer-Disjoint Partitioning
- When grouping strictly by signer to prevent leakage, rare signs (words performed by only 1 or 2 signers) cannot be represented in all three splits (train, validation, and test).
- In `data/manifests/signer_disjoint/`, 1,939 classes have training representations, but 332 rare classes lack test instances because those signers were assigned to the training split.
- Models evaluated on Protocol B must report macro metrics computed strictly over the evaluatable class subset, disclosing coverage counts.

---

## 2. Landmark Detection & MediaPipe Dependencies

### 2.1 MediaPipe Holistic Landmarker Architecture
- The pipeline depends on `models/mediapipe/holistic_landmarker.task`. If this asset is missing, real-time extraction cannot run, though pre-extracted `.npz` feature pipelines remain fully functional.
- MediaPipe calculates hand landmarks conditioned on detected wrist/palm bounding boxes. Fast lateral hand motions or severe motion blur can cause temporary hand tracking dropouts (yielding missing landmarks for 2–5 frames).
- SignFlow's `interpolate_short_gaps` and the proposed Reliability-Aware model (M6) attenuate this, but long occlusions (> 5 frames) cannot be reconstructed without RGB appearance context.

### 2.2 Resolution & Aspect Ratio Transitions
- MediaPipe's underlying `SegmentationSmoothingCalculator` caches image dimensions across consecutive calls. In the benchmark harness, switching resolutions dynamically from $320\times240$ to $480\times360$ on a single extractor instance triggered internal dimension assertions.
- Extractor instances in production must be reinitialized if the client changes webcam resolution dynamically.

---

## 3. Real-Time Latency on CPU vs GPU

- **CPU Latency Profile**: On host CPU (`Intel Core`), the end-to-end pipeline achieves 25–32 FPS ($480\times360$, $Q=75$) with median latency around 31.7 ms.
- **Cold-Start Latency**: The very first frame through MediaPipe and PyTorch encounters a cold-start compilation spike (up to ~2,000 ms) while XNNPACK kernels compile. Subsequent frames run steadily under 35 ms.
- **Webcam Real-World Validation**: Local testing used prerecorded frames from `data/WLASL-3/videos/00336.mp4`. Live user webcam performance remains dependent on physical lighting, user camera hardware, and browser WebRTC/WebSocket support.

---

## 4. Multi-Seed Training Status

- Full multi-epoch training across 3+ random seeds on the complete 2,000-class dataset requires ~8–12 hours of GPU compute on NVIDIA T4/A100.
- This compute must be executed in Google Colab using `train_colab.ipynb`. Local CPU execution is suitable for smoke testing, dataset validation, and real-time inference, but not full 100-epoch 2,000-class backpropagation.
