# SignFlow Evaluation Protocol: Experimental Rigor & Zero-Leakage Standards

## 1. Motivation & Principles

Rigorous evaluation of word-level sign language recognition models requires strict separation of:
1. **Model evaluation vs runtime speed**: A model running at high FPS does not imply high recognition fidelity.
2. **Signer-dependent vs Signer-independent generalization**: Randomly shuffling video clips places the same signer in both training and test sets. High test accuracy on shared signers often reflects memorization of signer appearance, clothing, and background rather than sign semantics.
3. **Canonical Vocabulary Consistency**: Vocabulary mappings must be identical across training, validation, testing, and live inference.

---

## 2. Evaluation Protocols

### Protocol A: Official / Reference Split
- **Manifests**:
  - `data/manifests/train_manifest.json`
  - `data/manifests/validation_manifest.json`
  - `data/manifests/test_manifest.json`
- **Characteristics**:
  - Matches the standard published WLASL dataset splits (Li et al., 2020).
  - **Signer Overlap**: High. Pairwise audit demonstrates that 104 out of 105 test signers (99.0%) appear in the training split.
  - **Scientific Purpose**: Serves exclusively as a benchmark reference for comparison against published literature that used the official WLASL partition.
  - **Reporting Constraint**: Must NEVER be reported as "signer-independent" or "out-of-distribution" accuracy.

### Protocol B: Signer-Disjoint Split
- **Manifests**:
  - `data/manifests/signer_disjoint/train_manifest.json`
  - `data/manifests/signer_disjoint/val_manifest.json`
  - `data/manifests/signer_disjoint/test_manifest.json`
- **Characteristics**:
  - Generated via deterministic greedy signer-level grouping (`scripts/generate_signer_disjoint_split.py`).
  - **Signer Partition**:
    - Train Signers: 87
    - Validation Signers: 16
    - Test Signers: 16
    - **Signer Overlap**: Exactly **0** (Pairwise intersections: $\text{Train} \cap \text{Test} = \emptyset$, $\text{Train} \cap \text{Val} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
  - **Scientific Purpose**: Measures genuine generalization to unseen deaf and hearing signers.

---

## 3. Mathematical Metric Definitions

### 3.1 Top-K Categorical Accuracy
For a test set of $N$ samples with true integer labels $y_i \in \{0, \dots, C-1\}$ and predicted probability vectors $\hat{\mathbf{p}}_i \in [0, 1]^C$:

$$\text{Top-K} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left( y_i \in \operatorname{argtop}_K(\hat{\mathbf{p}}_i) \right)$$

where $K \in \{1, 5, 10\}$.

### 3.2 Macro-Averaged Precision, Recall, and F1
To prevent high-frequency classes from dominating performance metrics, macro-averaging computes unweighted arithmetic means across all $C$ classes:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c + \epsilon}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c + \epsilon}$$

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \epsilon}$$

### 3.3 Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=0}^{C-1} \text{Recall}_c$$

### 3.4 Expected Calibration Error (ECE)
Partitions model confidence scores into $M = 10$ equally spaced intervals $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|$$

where $\operatorname{acc}(B_m)$ is empirical accuracy in bin $B_m$ and $\operatorname{conf}(B_m)$ is mean confidence.

---

## 4. Integrity and Provenance Rules

1. **Non-Exclusion Disclosure**: Evaluation reports must disclose the exact number of manifest samples, evaluated samples, and excluded samples (with reasons).
2. **Vocabulary Check**: Every checkpoint includes a SHA256 `vocabulary_hash`. The evaluation script verifies that the test manifest labels match the checkpoint vocabulary hash before computing metrics.
3. **Missing Landmark Strict Mode**: In `--strict` mode, evaluation immediately aborts if any referenced feature file is missing, corrupted, or contains NaNs.
