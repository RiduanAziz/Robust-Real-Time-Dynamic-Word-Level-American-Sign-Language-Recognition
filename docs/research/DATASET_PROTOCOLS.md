# SignFlow Dataset Evaluation Protocols

**Status:** Official Specification  
**Authors:** Senior Research Team  
**Scope:** WLASL-100 & WLASL-2000 Benchmark Protocols

---

## 1. Motivation: Preventing Signer Identity Leakage

A fundamental challenge in dynamic sign language recognition is distinguishing between **true sign recognition** (learning invariant hand, arm, and facial motion trajectories) and **signer identification** (memorizing specific individuals' physical appearance, clothing, skin tone, or idiosyncratic signing cadence).

Prior literature on the WLASL dataset often conflates these two objectives by reporting results on the standard instance-based partition without auditing signer overlap. To provide rigorous, research-grade evaluation, SignFlow defines and maintains two explicitly named evaluation protocols.

---

## 2. Protocol Specifications

### Protocol A — Official / Reference Split

- **Manifest Location:** `data/manifests/{train,validation,test}_manifest.json`
- **Partition Methodology:** Per-instance stratified split inherited from the original WLASL paper (Li et al., 2020).
- **Properties:**
  - Total Samples: 21,083
  - Train: 14,289 samples (117 signers, 2,000 classes)
  - Validation: 3,916 samples (109 signers, 2,000 classes)
  - Test: 2,878 samples (105 signers, 1,999 classes)
- **Signer Overlap Characteristics:**
  - $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Val}) = 108$ (99.1% overlap)
  - $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Test}) = 104$ (99.0% overlap)
  - $\text{Signers}(\text{Val}) \cap \text{Signers}(\text{Test}) = 98$
- **Research Classification:** **Signer-Dependent / Mixed-Signer Benchmark.**
- **Policy:** Maintained strictly for historical comparison against published WLASL papers. Must **never** be cited as a "signer-independent" result.

### Protocol B — Signer-Disjoint Split

- **Manifest Location:** `data/manifests/signer_disjoint/{train,validation,test}_manifest.json`
- **Partition Methodology:** Deterministic bin-packed signer-level grouping (random seed 42). Every signer is assigned to exactly one split.
- **Properties:**
  - Total Unique Signers: 119
  - Train Signers: 87 signers (14,757 samples)
  - Validation Signers: 16 signers (3,163 samples)
  - Test Signers: 16 signers (3,163 samples)
- **Signer Overlap Verification:**
  - $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Val}) = 0$
  - $\text{Signers}(\text{Train}) \cap \text{Signers}(\text{Test}) = 0$
  - $\text{Signers}(\text{Val}) \cap \text{Signers}(\text{Test}) = 0$
- **Class Coverage:**
  - WLASL-2000: Train has 2,000 classes; Validation has 1,576; Test has 1,783. 1,401 classes appear in all three splits.
  - WLASL-100: Train has 100 classes; Validation has 98; Test has 94. 92 classes appear across all three splits.
- **Research Classification:** **Strict Signer-Independent Benchmark.**
- **Policy:** Evaluates true out-of-distribution generalization to completely unseen signers.

---

## 3. Protocol Validation Commands

To programmatically verify manifest integrity and confirm that zero signer or sample leakage exists:

```powershell
# Validate Protocol A (Official)
python scripts/validate_dataset.py --protocol official

# Validate Protocol B (Signer-Disjoint) with strict zero-leakage enforcement
python scripts/validate_dataset.py --protocol signer_disjoint --strict
```

Both commands generate timestamped JSON reports in `results/`.
