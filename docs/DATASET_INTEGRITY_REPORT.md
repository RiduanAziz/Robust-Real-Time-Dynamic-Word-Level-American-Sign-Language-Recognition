# SignFlow: Dataset Integrity & Leakage Verification Report

**Verification Date:** October 26, 2026  
**Validator Tool:** `scripts/validate_dataset.py`  
**Dataset Base:** World-Level American Sign Language (WLASL)  

---

## 1. Physical Dataset Inventory

A complete audit of local file assets was conducted:
- **Total Registered Samples (`data/manifests/full_manifest.json`)**: 21,083
- **Total Extracted Landmark Files (`data/landmarks/*.npz`)**: 21,083
- **Missing or Unreadable Landmark Files**: **0 (100% extraction coverage)**
- **Feature File Format**: `.npz` archives containing float32 coordinate tensors and validity masks.
- **Corrupted / NaN Coordinate Arrays**: **0 detected**.

---

## 2. Split Partition Audit

### 2.1 Protocol A: Official WLASL Benchmark Splits
- **Source Files**: `data/manifests/train_manifest.json`, `data/manifests/validation_manifest.json`, `data/manifests/test_manifest.json`
- **Sample Distribution (Full 2,000 Classes)**:
  - Train: 14,289 samples (117 signers, 2,000 classes)
  - Validation: 3,916 samples (109 signers, 2,000 classes)
  - Test: 2,878 samples (105 signers, 1,999 classes)
- **Sample Distribution (WLASL-100 Subset)**:
  - Test Set: 257 samples matching the 100 classes in `configs/experiments/wlasl100_proposed.yaml`.
- **Signer Contamination Matrix**:
  - $\text{Train} \cap \text{Val}$: **108 shared signers**
  - $\text{Train} \cap \text{Test}$: **104 shared signers** (99.0% of test signers)
  - $\text{Val} \cap \text{Test}$: **98 shared signers**
  - *Integrity Assessment*: High signer leakage inherited from the source benchmark. Suitable for comparative literature baselines only.

### 2.2 Protocol B: Signer-Disjoint Splits
- **Source Files**: `data/manifests/signer_disjoint/`
- **Generated via**: `scripts/generate_signer_disjoint_split.py --seed 42`
- **Sample Distribution**:
  - Train: 14,757 samples (87 signers, 2,000 classes)
  - Validation: 3,163 samples (16 signers, 1,576 classes)
  - Test: 3,163 samples (16 signers, 1,783 classes)
- **Signer Contamination Matrix**:
  - $\text{Train} \cap \text{Val}$: **0 (ZERO)**
  - $\text{Train} \cap \text{Test}$: **0 (ZERO)**
  - $\text{Val} \cap \text{Test}$: **0 (ZERO)**
  - *Integrity Assessment*: Strictly signer-independent. Eliminates appearance memorization.

---

## 3. Strict Validation CLI Results

Executing the dataset validator with strict checks:
```bash
python scripts/validate_dataset.py --protocol signer_disjoint --strict
```
**Outcome**:
- Status: `PASSED (strict=True)`
- NaN / Infinite Values: None detected.
- Invalid Coordinates: None detected.
- Pairwise Leakage Violations: 0 detected.
- Report written to: `results/dataset_validation_signer_disjoint.json`
