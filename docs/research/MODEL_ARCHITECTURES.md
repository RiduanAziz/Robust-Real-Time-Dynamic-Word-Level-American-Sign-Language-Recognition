# SignFlow Model Architectures: Benchmark & Research Specifications

This document specifies the architectural designs, input contracts, mathematical foundations, and implementation details for the SignFlow dynamic word-level sign language recognition model suite.

---

## 1. Unified Interface & Tensor Contract

All SignFlow model architectures adhere to a unified forward contract:

$$\mathbf{X} \in \mathbb{R}^{B \times T \times D}$$

where:
- $B$: Batch size
- $T$: Temporal sequence length ($T = 64$ frames by default)
- $D$: Per-frame feature dimension:
  - **Representation A (Hands only)**: $D = 126$ ($2 \times 21 \times 3$)
  - **Representation B (Hands + Pose)**: $D = 225$ ($75 \times 3$)
  - **Representation C (Holistic Full)**: $D = 1,659$ ($553 \times 3$ with hands, pose, and face) or $D = 4,977$ when velocity and acceleration are concatenated.
  - **Representation D (Engineered Features)**: $D = 257$ (joint-relative coordinates, inter-wrist distances, finger joint extension angles, movement velocities).
- $\mathbf{M} \in \{0, 1\}^{B \times T}$: Boolean validity mask indicating non-padded, active frames.

---

## 2. Model Baseline Specifications

### M0: Historical Robust Holistic Fusion Classifier (Baseline)
- **Path**: `src/sign_language/models/holistic_classifier.py` (`RobustHolisticFusionClassifier`)
- **Modality Streams**:
  - Separate dense projection heads for Left Hand ($D_{\text{LH}} = 63$), Right Hand ($D_{\text{RH}} = 63$), Pose ($D_{\text{Pose}} = 99$), and Face ($D_{\text{Face}} = 1,434$).
- **Temporal Modelling**: Bidirectional GRU (2 layers, hidden size 256, dropout 0.3).
- **Fusion**: Modality gating layer computing per-stream attention weights $\alpha_m = \sigma(\mathbf{W}_g \mathbf{h}_m + \mathbf{b}_g)$.
- **Status**: Historical baseline preserved without regression. Evaluated at 74.32% Top-1 on WLASL-100 (Protocol A).

### M1: Lightweight Pooled Representation Baseline
- **Path**: `src/sign_language/models/baseline.py`
- **Mechanism**: Masked temporal average and max pooling across valid sequence frames:
  $$\mathbf{h}_{\text{pooled}} = \left[ \frac{1}{\sum_t M_t} \sum_{t=1}^T M_t \mathbf{x}_t \,\|\, \max_{t: M_t=1} \mathbf{x}_t \right]$$
- **Classification Head**: 2-layer MLP with LayerNorm, GELU, and Dropout.
- **Purpose**: Establishes the lower-bound contribution of temporal ordering vs static sign geometry.

### M2: Temporal Convolutional Network (TCN)
- **Path**: `src/sign_language/models/tcn.py` (`TCNClassifier`)
- **Mechanism**: Stack of dilated causal/semi-causal 1D convolutions with residual skip connections:
  - Dilations: $d \in \{1, 2, 4, 8\}$ providing an exponential receptive field ($R = 1 + \sum 2(k-1)d_i$).
  - Weight normalization, Chomp1d / masked temporal padding, and spatial dropout ($p = 0.25$).
- **Advantages**: Fast parallel training, stable gradient propagation over long sequences, no recurrent memory bottlenecks.

### M3: Recurrent Sequence Model (BiGRU / BiLSTM)
- **Mechanism**: Multi-layer bidirectional recurrent network with packed sequence processing or validity mask zeroing to avoid padding corruption.
- **Pooling**: Attention-weighted temporal pooling over non-padded steps.

### M4: Skeleton Spatio-Temporal Graph Convolutional Network (ST-GCN)
- **Topology**: Anatomical adjacency matrix $\mathbf{A} \in \mathbb{R}^{V \times V}$ based on human skeletal kinematics (MediaPipe 21 hand joints + 33 pose joints).
- **Spatial Graph Convolution**:
  $$f_{\text{out}}(v_i) = \sum_{v_j \in B(v_i)} \frac{1}{Z_{i}(v_j)} f_{\text{in}}(v_j) \cdot \mathbf{W}(l(v_i, v_j))$$
- **Temporal Convolution**: 1D convolution across the temporal dimension for each skeleton node.

### M5: Feature-Isolated Temporal Transformer
- **Path**: `src/sign_language/models/temporal_transformer.py`
- **Mechanism**: Multi-head self-attention with learned temporal sinusoidal positional embeddings and causal/key padding masks.
- **Feature Isolation**: Separate spatial linear projections before multi-head temporal self-attention.

---

## 3. Proposed Research Architecture: Reliability-Aware Feature-Isolated Spatiotemporal Model (M6)

### Conceptual Innovation
Standard sign language models treat all coordinate streams equally, even when a hand is completely out of frame or MediaPipe fails detection. In natural signing, one-handed signs (e.g., "HELLO", "THANK YOU") or temporary detector occlusions produce missing-data frames that corrupt ordinary linear projections and attention heads.

The **Reliability-Aware Feature-Isolated Spatiotemporal Model** (`ReliabilityAwareIsolatedFusionClassifier` in `src/sign_language/models/reliability_fusion.py`) addresses this directly:

```
                         [ Input Frame Sequence: B x T x D ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [ Feature Slicing ]                         [ Modality Masks ]
      Left Hand   Right Hand   Pose               m_LH   m_RH   m_Pose
         │            │          │                  │      │       │
         ▼            ▼          ▼                  │      │       │
      [Spatial]    [Spatial]   [Spatial]            │      │       │
      Encoder      Encoder     Encoder              │      │       │
         │            │          │                  │      │       │
         ▼            ▼          ▼                  │      │       │
      [Temporal]   [Temporal]  [Temporal]           │      │       │
      Encoder      Encoder     Encoder              │      │       │
      (Isolated)   (Isolated)  (Isolated)           │      │       │
         │            │          │                  │      │       │
         └────────────┼──────────┘                  │      │       │
                      ▼                             ▼      ▼       ▼
       ┌──────────────────────────────────────────────────────────────┐
       │             Reliability-Aware Dynamic Fusion                 │
       │  g_m = σ( W_h h_m + W_r r_m + b )   where r_m = mean(mask_m) │
       │  z = Σ g_m ⊙ h_m / (Σ g_m + ε)                               │
       └──────────────────────────────┬───────────────────────────────┘
                                      ▼
                        [ Classification Head + ECE ]
                                      ▼
                            Class Logits & Top-K
```

### Mathematical Formulation
1. **Modality Isolation**:
   Left hand $\mathbf{X}_{\text{LH}} \in \mathbb{R}^{B \times T \times 63}$, Right hand $\mathbf{X}_{\text{RH}} \in \mathbb{R}^{B \times T \times 63}$, Body pose $\mathbf{X}_{\text{Pose}} \in \mathbb{R}^{B \times T \times 99}$.
2. **Reliability Metric**:
   For each modality $m \in \{\text{LH}, \text{RH}, \text{Pose}\}$, compute empirical sequence reliability:
   $$r_m = \frac{1}{T} \sum_{t=1}^T \mathbf{1}(\|\mathbf{x}_{m, t}\|_2 > \epsilon) \in [0, 1]$$
3. **Reliability Gating**:
   $$g_m = \sigma\left( \mathbf{W}_h \mathbf{h}_m + \mathbf{W}_r r_m + \mathbf{b} \right)$$
   If a hand is absent ($r_m \approx 0$), its contribution to the final sign representation is dynamically attenuated towards zero, preventing hallucinated classification evidence.
4. **Normalized Gated Fusion**:
   $$\mathbf{z} = \frac{\sum_m g_m \odot \mathbf{h}_m}{\sum_m g_m + \epsilon}$$
