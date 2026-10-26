# SignFlow: Final Model Evaluation Report

**Evaluation Date:** October 26, 2026  
**Evaluator:** Automated Deterministic Evaluation Pipeline (`scripts/evaluate.py`)  
**Target Checkpoint:** `models/robust_holistic_fusion_best.pt`  
**Configuration File:** `configs/experiments/wlasl100_proposed.yaml`  
**Evaluation Protocol:** Protocol A (Official Reference Benchmark)  

---

## 1. Primary Recognition Performance Metrics

| Metric | Score | Sample Count / Basis | Provenance & Notes |
| :--- | :---: | :---: | :--- |
| **Top-1 Categorical Accuracy** | **74.32%** (0.74319) | 191 / 257 | Exact reproduction of historical baseline. |
| **Top-5 Categorical Accuracy** | **88.72%** (0.88716) | 228 / 257 | Five highest-scoring softmax candidates. |
| **Top-10 Categorical Accuracy** | **94.16%** (0.94163) | 242 / 257 | Ten highest-scoring softmax candidates. |
| **Balanced Accuracy** | **73.92%** (0.73917) | 100 classes | Unweighted mean of per-class recalls. |
| **Macro Precision** | **0.7628** | 100 classes | Arithmetic mean of per-class precisions. |
| **Macro Recall** | **0.7392** | 100 classes | Arithmetic mean of per-class recalls. |
| **Macro-F1 Score** | **0.7213** | 100 classes | Unweighted harmonic mean over all classes. |
| **Weighted-F1 Score** | **0.7257** | 257 samples | Support-weighted harmonic mean. |
| **Expected Calibration Error (ECE)** | **0.1598** | 15 confidence bins | Measures probability calibration reliability. |

---

## 2. Sample Census & Evaluation Provenance

```
Total Manifest Samples:             2,878
Matching 100-Class Target Samples:  257
Excluded / Missing Samples:         0 (0.00%)
Evaluated Samples:                  257
Number of Vocabulary Classes:       100
Vocabulary SHA256 Hash:             5b4e7a89277f...
Evaluation Device:                  Host CPU (Intel Core)
Runtime Duration:                   ~20.3 seconds for 257 sequences
Throughput:                         ~12.6 sequences / second
```

---

## 3. Top Per-Class Performance Summary

### 3.1 Perfect Recognition Classes (F1 = 1.0, 100% Precision & Recall)
The following signs achieved perfect 1.0 F1 score on the held-out test split:
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

### 3.2 Most Challenging Classes (F1 < 0.4)
- `thin` (precision: 0.0, recall: 0.0, F1: 0.0, support: 3) — Confused with `clothes` due to identical vertical finger-pinching motion.
- `who` (precision: 0.50, recall: 0.33, F1: 0.40, support: 3) — Circular chin motion obscured by facial landmark variance.
- `no` (precision: 1.00, recall: 0.33, F1: 0.50, support: 3) — Quick two-finger snap easily clipped during temporal windowing.

---

## 4. Deterministic Reproduction Command

To reproduce these exact numerical figures locally:

```bash
python scripts/evaluate.py \
    --config configs/experiments/wlasl100_proposed.yaml \
    --checkpoint models/robust_holistic_fusion_best.pt \
    --manifest data/manifests/test_manifest.json \
    --landmarks-dir data/landmarks \
    --device cpu \
    --output results/test_eval_reproduced.json
```
