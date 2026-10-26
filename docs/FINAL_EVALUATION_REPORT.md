# SignFlow: Final Model Evaluation Report

**Evaluation Date:** October 26, 2026  
**Evaluator:** Automated Deterministic Evaluation Pipeline (`scripts/evaluate.py` & `scripts/ensemble_eval.py`)  
**Evaluation Protocol:** Protocol A (Official WLASL Benchmark Partition, 257 held-out test samples)  
**Target Vocabulary:** WLASL-100 (100 canonical classes, SHA256: `50a15ecd4abb546fe0e4859713ed6a6c34e4a35caddec1ddc9fdf79c41d19975`)  

---

## 1. Executive Summary: Benchmark & Ensemble Results

SignFlow has established a new defensible benchmark on WLASL-100 using a validation-tuned, zero-leakage probability ensemble combining three distinct inductive biases:
1. **Model 1 (`robust_holistic_fusion_best.pt`)**: Baseline BiGRU temporal modeling across holistic landmark coordinates.
2. **Model 2 (`wlasl100_unisign_pgf_best.pt`)**: Uni-Sign Prior-Guided Fusion (PGF) with score-aware reliability gating and anatomical cross-attention (`arXiv:2501.15187`).
3. **Model 3 (`wlasl100_max_accuracy_best.pt`)**: Deep 2-layer BiGRU with label smoothing ($\epsilon=0.08$) and cosine warmup.

### Comprehensive Model Comparison Table (Protocol A - Official Held-Out Test Split, N=257)

| Model / Ensemble Configuration | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy | Macro-F1 | Weighted-F1 | ECE | Status / Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Historical Baseline (`robust_holistic_fusion`)** | 74.32% | 88.72% | 94.16% | 0.7213 | 0.7257 | 0.1598 | Exact reproduction (`models/robust_holistic_fusion_best.pt`) |
| **Candidate 1 (`unisign_pgf`) Solo** | 70.43% | 88.33% | 92.61% | 0.6863 | 0.6900 | **0.0561** | Outstanding calibration; hand-to-pose cross-attention |
| **Candidate 2 (`wlasl100_max_accuracy`) Solo** | 72.76% | 89.88% | 93.77% | 0.7180 | 0.7215 | 0.0912 | Deep BiGRU (hidden 128, emb 96, 200 epochs) |
| **Track D: 2-Way Ensemble ($\alpha=0.55$)** | 76.26% | 90.66% | 93.77% | 0.7459 | 0.7490 | **0.0551** | Model 1 + Model 2 (tuned on validation split) |
| **Track D: 3-Way Ensemble (WINNING BENCHMARK)** | **76.65%** | **91.44%** | **94.16%** | **0.7558** | **0.7542** | **0.0838** | **Weights: $(0.25, 0.10, 0.65)$, Val Top-1: 78.34%** |

### Benchmark Gains Over Baseline
* **Top-1 Accuracy:** $+2.33\%$ absolute improvement ($74.32\% \rightarrow \mathbf{76.65\%}$, $197 / 257$ correct).
* **Top-5 Accuracy:** $+2.72\%$ absolute improvement ($88.72\% \rightarrow \mathbf{91.44\%}$, $235 / 257$ correct).
* **Top-10 Accuracy:** Reaches $\mathbf{94.16\%}$ ($242 / 257$ correct).
* **Macro-F1 Score:** Jump from $0.7213 \rightarrow \mathbf{0.7558}$ ($+0.0345$ macro harmonic gain across all 100 classes).
* **Expected Calibration Error (ECE):** Dropped from $0.1598 \rightarrow \mathbf{0.0838}$ (a $47.6\%$ reduction in prediction overconfidence).

---

## 2. Validation Grid Search & Zero-Leakage Guarantee

To prevent test-set data leakage, fusion weights were determined via an exhaustive 3-simplex grid search ($w_1 + w_2 + w_3 = 1.0$) evaluated **strictly on the 337 validation samples** (`data/manifests/validation_manifest.json`):

```
Validation Grid Search (N=337):
  - Optimal Weights: w1 = 0.25 (Baseline), w2 = 0.10 (Uni-Sign PGF), w3 = 0.65 (Deep BiGRU)
  - Validation Top-1 Accuracy: 78.34% (264 / 337 correct)
```

The locked weights $(0.25, 0.10, 0.65)$ were subsequently evaluated in one single forward pass on the 257 held-out test samples, producing the final **76.65% Top-1** and **91.44% Top-5** test performance.

---

## 3. Sample Census & Evaluation Provenance

```
Total Manifest Samples:             2,878
Matching 100-Class Target Samples:  257
Excluded / Missing Samples:         0 (0.00%)
Evaluated Test Samples:             257
Evaluated Validation Samples:       337
Number of Vocabulary Classes:       100
Vocabulary SHA256 Hash:             50a15ecd4abb546fe0e4859713ed6a6c34e4a35caddec1ddc9fdf79c41d19975
Evaluation Device:                  CUDA GPU / Host CPU
Throughput:                         ~42.5 sequences / second
```

---

## 4. Top Per-Class Performance Summary

### 4.1 Perfect Recognition Classes (F1 = 1.0, 100% Precision & Recall)
The ensemble achieved perfect recognition on the following signs:
- `computer` (support: 5)
- `go` (support: 3)
- `chair` (support: 3)
- `walk` (support: 3)
- `fine` (support: 3)
- `yes` (support: 3)
- `hot` (support: 3)
- `orange` (support: 3)
- `mother` (support: 4)
- `now` (support: 3)
- `white` (support: 4)
- `family` (support: 3)
- `hearing` (support: 3)
- `dance` (support: 3)
- `table` (support: 3)
- `apple` (support: 3)

### 4.2 Recovered Ambiguities Through Ensembling
- **`thin` vs `clothes`:** The baseline confused these two signs due to identical vertical finger-pinching motion. Uni-Sign PGF's hand-to-pose attention disambiguated the wrist elevation relative to shoulder anchors, allowing the ensemble to correctly classify `thin`.
- **`who`:** Circular chin motion obscured by head orientation variance was successfully stabilized by the deep BiGRU's temporal smoothing.

---

## 5. Deterministic Reproduction Commands

### Reproduce Solo Baseline (74.32% Top-1):
```bash
python scripts/evaluate.py \
    --config configs/colab/wlasl100_baseline_colab.yaml \
    --checkpoint models/robust_holistic_fusion_best.pt \
    --manifest data/manifests/test_manifest.json \
    --landmarks-dir data/landmarks \
    --protocol official
```

### Reproduce Solo Uni-Sign PGF (70.43% Top-1, ECE 0.0561):
```bash
python scripts/evaluate.py \
    --config configs/colab/wlasl100_unisign_pgf.yaml \
    --checkpoint models/wlasl100_unisign_pgf_best.pt \
    --manifest data/manifests/test_manifest.json \
    --landmarks-dir data/landmarks \
    --protocol official
```

### Reproduce Solo Deep BiGRU (72.76% Top-1, 89.88% Top-5):
```bash
python scripts/evaluate.py \
    --config configs/colab/wlasl100_max_accuracy.yaml \
    --checkpoint models/wlasl100_max_accuracy_best.pt \
    --manifest data/manifests/test_manifest.json \
    --landmarks-dir data/landmarks \
    --protocol official
```

### Reproduce 3-Way Winning Ensemble (76.65% Top-1, 91.44% Top-5):
```bash
python scripts/ensemble_eval.py \
    --model1-ckpt models/robust_holistic_fusion_best.pt \
    --model1-config configs/dataset.yaml \
    --model2-ckpt models/wlasl100_unisign_pgf_best.pt \
    --model2-config configs/colab/wlasl100_unisign_pgf.yaml \
    --model3-ckpt models/wlasl100_max_accuracy_best.pt \
    --model3-config configs/colab/wlasl100_max_accuracy.yaml \
    --output results/three_way_ensemble_report.json
```
