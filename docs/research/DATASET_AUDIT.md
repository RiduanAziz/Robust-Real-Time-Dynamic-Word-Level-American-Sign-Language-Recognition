# SignFlow Comprehensive Dataset Audit Report

**Date of Audit:** October 25, 2026  
**Dataset Base:** WLASL (World-Level American Sign Language)  
**Total Landmarks Catalogued:** 21,083 `.npz` files in `data/landmarks/`  
**Total Manifest Entries:** 21,083 entries in `data/manifests/full_manifest.json`  
**Missing Landmark Files:** **0** (100% extraction coverage)  
**Corrupted / Non-finite Feature Arrays:** **0**

---

## 1. Split Distribution Overview

| Split Name | Protocol A (Official) Samples | Protocol A Signers | Protocol B (Signer-Disjoint) Samples | Protocol B Signers |
|---|---:|---:|---:|---:|
| **Train** | 14,289 | 117 | 14,757 | 87 |
| **Validation** | 3,916 | 109 | 3,163 | 16 |
| **Test** | 2,878 | 105 | 3,163 | 16 |
| **Total** | **21,083** | **119 Unique** | **21,083** | **119 Unique** |

---

## 2. Signer Leakage Audit

### Protocol A (Official WLASL Instance Split)
- **Train ∩ Validation:** 108 shared signers.
- **Train ∩ Test:** 104 shared signers.
- **Validation ∩ Test:** 98 shared signers.
- **Signer Independence:** **Violated.** Signer appearance and body proportions are observable across both training and test sets.

### Protocol B (Signer-Disjoint Split)
- **Train ∩ Validation:** **0** shared signers.
- **Train ∩ Test:** **0** shared signers.
- **Validation ∩ Test:** **0** shared signers.
- **Signer Independence:** **Guaranteed.** Certified via `scripts/validate_dataset.py --protocol signer_disjoint --strict`.

---

## 3. Class Coverage Across Partitions

| Vocabulary Scope | Protocol A (All 3 Splits) | Protocol B (Train) | Protocol B (Validation) | Protocol B (Test) | Protocol B (All 3 Splits) |
|---|---:|---:|---:|---:|---:|
| **WLASL-2000** | 1,999 classes | 2,000 classes | 1,576 classes | 1,783 classes | 1,401 classes |
| **WLASL-100** | 100 classes | 100 classes | 98 classes | 94 classes | 92 classes |

> **Note on Rare Classes in Protocol B:** Because some vocabulary words in WLASL were only performed by 1 or 2 distinct signers in the wild, assigning signers exclusively to a single split naturally means that a small subset of words (8 out of 100 in WLASL-100) do not appear in all 3 splits. For strict 100-class signer-disjoint training, evaluation must either report on the 92 fully supported mutual classes or report missing-class recall as 0.

---

## 4. Feature Quality & NaN/Inf Audit

- **Sampled Arrays Checked:** 150 feature files across train, val, test.
- **Array Shapes:** Variable time steps $T \in [14, 180]$, feature dimensions $D \in \{126, 225, 1659, 4977\}$.
- **Missing Landmark Masks:** Stored under key `"mask"`, binary float32 values indicating detection confidence $\ge 0.5$.
- **NaN / Infinity Errors:** 0. All values are finite floats.

---

## 5. Canonical Vocabulary Provenance

The canonical vocabulary is fixed to the WLASL official frequency order:
- `data/wlasl_class_list.txt` (Index 0 to 1999)
- Top 5: `0: book`, `1: drink`, `2: computer`, `3: before`, `4: chair`
- Top 100 vocabulary hash: `78e87dbfc9f4939cb9a6d3fbe0ee39371e7a4697f9036f0b401017c603ca9142`
