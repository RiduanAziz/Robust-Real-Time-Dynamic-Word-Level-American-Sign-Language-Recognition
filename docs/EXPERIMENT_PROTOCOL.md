# SignFlow: Research Experiment Protocol & Validation Standards

## 1. Experimental Foundations & Non-Negotiable Rules

To ensure scientific validity and reproducibility, all SignFlow experiments must adhere to the following protocol:

1. **Protocol Separation**:
   - **Protocol A (Official WLASL Partition)**: Evaluates against published literature (Li et al. 2020, Sign2Pose, Siformer). Results must be explicitly disclosed as containing signer overlap (104 shared signers).
   - **Protocol B (Signer-Disjoint Partition)**: Evaluates strict out-of-distribution signer generalization. Zero signer overlap permitted ($\text{Train} \cap \text{Test} = \emptyset$).
2. **Canonical Vocabulary Locking**:
   - Vocabulary order and gloss mapping must be cryptographically locked via `vocabulary_hash` (SHA256).
   - The test set must not be re-indexed or filtered to artificially boost accuracy.
3. **Hyperparameter Selection Policy**:
   - All hyperparameter tuning, model checkpoint selection, early stopping, and ensemble weighting must be performed strictly using **validation set performance**.
   - The final test partition is evaluated **exactly once** per candidate model.
4. **Zero Missing-Sample Concealment**:
   - Evaluation reports must record: total manifest samples, evaluated samples, and excluded samples (with reasons).

---

## 2. Mathematical Metric Formulations

### 2.1 Top-1 and Top-K Categorical Accuracy
For $N$ evaluated test sequences with true class labels $y_i \in \{0, \dots, C-1\}$ and model logits vector $\mathbf{z}_i \in \mathbb{R}^C$:

$$\hat{y}_i^{(1)} = \operatorname{argmax}_{c} z_{i, c}, \quad \text{Top-1} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left(\hat{y}_i^{(1)} = y_i\right)$$

$$\text{Top-K} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left(y_i \in \operatorname{argtop}_K(\mathbf{z}_i)\right)$$

### 2.2 Macro-Averaged Precision, Recall, and F1
To prevent high-frequency vocabulary items from biasing metrics, macro-averaging computes unweighted arithmetic means across all classes $c \in \{0, \dots, C-1\}$:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c + \epsilon}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c + \epsilon}$$

$$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \epsilon}$$

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=0}^{C-1} \text{F1}_c$$

### 2.3 Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=0}^{C-1} \text{Recall}_c$$

### 2.4 Expected Calibration Error (ECE)
Partitions predicted probabilities into $M = 15$ confidence bins $B_m \in \left(\frac{m-1}{M}, \frac{m}{M}\right]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|$$

---

## 3. Provenance & Artifact Tracking

Every training and evaluation run produces an immutable JSON record containing:
- Experiment ID and descriptive run name.
- Model architecture name, parameter count, and tensor shapes.
- Timestamp, platform hardware, OS, and PyTorch version.
- Exact training/evaluation seed.
- Input feature representation (e.g., `holistic`, `hands_only`, `engineered`).
- Checkpoint SHA256 hash and manifest SHA256 hash.
- Full confusion matrix and per-class metrics dictionary.
