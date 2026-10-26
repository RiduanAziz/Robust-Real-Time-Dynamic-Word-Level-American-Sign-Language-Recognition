# SignFlow Model Card: Robust Holistic Fusion Classifier

## 1. Model Details
- **Model Name**: SignFlow Robust Holistic Fusion Classifier (`RobustHolisticFusionClassifier`)
- **Version**: 2.0-research
- **Model Type**: Deep Spatiotemporal Neural Network with Multi-Stream Gated Modality Fusion
- **Developers**: SignFlow Research Team
- **Framework**: PyTorch 2.0+
- **License**: MIT License
- **Target Vocabulary**: WLASL-100 (100 American Sign Language isolated dynamic words)

---

## 2. Intended Use
- **Primary Use**: Real-time assistive word-level American Sign Language recognition from webcam video streams or prerecorded signing clips.
- **Intended Users**: Deaf, hard-of-hearing, and hearing individuals communicating via ASL; accessibility researchers; educational applications.
- **Out-of-Scope Uses**:
  - Continuous sentence-level sign language translation without isolated word segmentation.
  - Non-ASL sign languages (e.g., British Sign Language, Bengali Sign Language, Auslan) without retraining.
  - Critical life-safety or emergency translation without human verification.

---

## 3. Input & Output Contract
- **Input Tensors**:
  - `landmarks`: Float32 tensor of shape `(Batch, SequenceLength=64, InputDim=1659)` representing normalized 3D upper-body landmarks (Left Hand: 63, Right Hand: 63, Pose: 99, Face Mesh: 1434).
  - `mask`: Float32 boolean tensor of shape `(Batch, SequenceLength=64)` indicating active vs padded frames.
- **Output Tensors**:
  - `logits`: Float32 tensor of shape `(Batch, NumClasses=100)`.
  - Top-1 and Top-5 class predictions with associated softmax probabilities.

---

## 4. Performance & Benchmark Summary
- **Protocol A (Official WLASL Benchmark, 257 held-out test samples)**:
  - Top-1 Accuracy: **74.32%**
  - Top-5 Accuracy: **88.72%**
  - Top-10 Accuracy: **94.16%**
  - Macro-F1 Score: **0.7213**
  - Weighted-F1 Score: **0.7257**
  - Expected Calibration Error: **0.1598**
- **Inference Latency (Host CPU, $480\times360$, $Q=75$)**:
  - Median End-to-End Latency: **31.66 ms (~31.6 FPS)**
  - 95th Percentile Latency: **36.64 ms**
  - Model-Only Forward Pass: **29.33 ms**

---

## 5. Ethical Considerations & Bias
- **Signer Diversity**: Protocol A training data exhibits signer overlap with the test set. Models deployed in production may experience accuracy degradation on signers with different skin tones, hand proportions, or signing cadences.
- **Assistive Framing**: The model should augment human communication rather than replace professional ASL interpreters.
