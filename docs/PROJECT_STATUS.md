# Project status audit

## Repository metadata

- Repository: `RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`
- Current branch: `main`
- Latest commit: `94d10c36e5c13dc9d4134656854c976d8ab0c33d`
- Working tree: ahead of `origin/main` by 2 commits (`git status -sb`)
- File count: 12,089 files in the working tree
- Source file count: 69 Python files
- Test count: 17 Python test files
- Configuration count: 12 YAML/JSON config files
- Dataset presence: yes, local dataset detected under `data/`
- Dataset structure: `data/raw/` contains 11,980 MP4 videos; `data/` also contains WLASL metadata JSON files and a class list file

## Current dataset findings

The local dataset is not the placeholder 5-class, 20-signer toy dataset described in the scaffold code.

- Local video directory: `data/raw/`
- Raw MP4 files: 11,980
- Metadata files present: `WLASL_v0.3.json`, `nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json`
- WLASL class count: 2,000 glosses (`WLASL_v0.3.json`)
- Metadata instance count: 21,083 instances across train/val/test
- Local raw videos matched to metadata: 11,980 videos, leaving 9,103 metadata entries unrepresented on disk
- Signer count in metadata: 119 unique signer IDs
- Split counts from metadata: train = 14,289; val = 3,916; test = 2,878
- Signer overlap across splits: train∩val = 108, train∩test = 104, val∩test = 98

This means the repository's configuration and code assumptions are materially inconsistent with the actual dataset.

## Phase status summary

| Phase | Status | Evidence | Remaining work |
| --- | --- | --- | --- |
| 0 | 🟡 PARTIALLY COMPLETE | Python package metadata, CI, Docker, env files, and docs exist. | End-to-end validation and environment reproducibility are not demonstrated in this session. |
| 1 | 🟡 PARTIALLY COMPLETE | Canonical package exists under `src/sign_language/`, but a compatibility layer remains in `src/data/`. | Remove or strictly deprecate duplicate API surface and centralize the data pipeline. |
| 2 | 🔴 NOT COMPLETE | `configs/dataset.yaml` still declares 5 labels, 20 signers, 6 samples/class, 16-frame sequences, and `hands`-only feature mode. | Replace placeholder config with real dataset-derived values. |
| 3 | ✅ COMPLETE (audited) | The actual local dataset was discovered in `data/raw/` and cross-checked against WLASL metadata in `data/WLASL_v0.3.json`. | Generate and persist the dataset inventory and manifest files. |
| 4 | 🔴 NOT COMPLETE | No `data/manifests/` directory or manifest JSON files exist. | Build a validated manifest from actual video files and metadata. |
| 5 | 🔴 NOT COMPLETE | Signer IDs overlap across train/val/test; therefore the current metadata does not provide a signer-independent split. | Implement real split generation and validation with explicit signer checks. |
| 6 | 🔴 NOT COMPLETE | No EDA notebook or figure outputs in `results/figures/`. | Build dataset-quality analysis from real sample statistics. |
| 7 | 🟡 PARTIALLY COMPLETE | MediaPipe-related modules exist, but the extraction pipeline uses synthetic zeros and placeholder logic. | Replace synthetic landmark logic with real MediaPipe extraction over actual video files. |
| 8 | 🔴 NOT COMPLETE | No canonical landmark schema document or validation tests exist. | Define the schema and add tests for modality ordering and missing-data semantics. |
| 9 | 🔴 NOT COMPLETE | Normalization utilities exist, but no mathematically justified, dataset-driven validation is present. | Implement and verify translation/scale invariance on real landmarks. |
| 10 | 🔴 NOT COMPLETE | Sequence handling utilities exist, but no real padded/variable-length validation against true dataset sequences is demonstrated. | Implement actual temporal batching with masking and truncation checks. |
| 11 | 🔴 NOT COMPLETE | Feature-variant modules exist but no proven real feature extraction for hands/pose/face is demonstrated. | Run controlled feature experiments on actual data. |
| 12 | 🔴 NOT COMPLETE | Real dataset loader is not in place; synthetic data still appears in `src/sign_language/data/dataset.py`. | Remove synthetic-only training path from the research workflow. |
| 13 | 🔴 NOT COMPLETE | Model family definitions exist, but no validated real training baseline has been executed. | Train MLP/LSTM/GRU/Transformer on real data. |
| 14 | 🔴 NOT COMPLETE | Training loop exists but there is no evidence of a real tracked experiment on the local dataset. | Run end-to-end training and checkpointing. |
| 15 | 🔴 NOT COMPLETE | Evaluation utilities exist, but they are not tied to a trained checkpoint on actual data. | Run test-set evaluation and export metrics. |
| 16 | 🔴 NOT COMPLETE | No controlled model-comparison summary with real metrics was generated. | Compare baseline models using a consistent protocol. |
| 17 | 🔴 NOT COMPLETE | Multimodal architecture modules exist, but no holistic representation evaluation is demonstrated. | Evaluate hands vs hands+pose vs hands+pose+face. |
| 18 | 🔴 NOT COMPLETE | No real spatial-noise experiment with severity/seed control exists. | Implement and evaluate jitter, translation, scale, and mask perturbations. |
| 19 | 🔴 NOT COMPLETE | No real temporal-noise pipeline exists. | Implement and evaluate frame dropping/duplication/truncation. |
| 20 | 🔴 NOT COMPLETE | Robustness metric logic is not validated against actual recognition degradation. | Measure clean vs noisy model performance across noise scenarios. |
| 21 | 🔴 NOT COMPLETE | No combined spatial-temporal robustness grid has been produced. | Run severity matrix and generate CSV/JSON/heatmap. |
| 22 | 🔴 NOT COMPLETE | No proposed robust method has been implemented and justified with evidence. | Design and validate a constrained method using earlier findings. |
| 23 | 🔴 NOT COMPLETE | No ablation study exists. | Run the required feature and robustness ablations. |
| 24 | 🔴 NOT COMPLETE | No unseen-signer final evaluation is in place. | Execute signer-independent final evaluation. |
| 25 | 🔴 NOT COMPLETE | Real-time inference code exists, but it is not demonstrated as a real pipeline using model input. | Connect webcam feed to trained model and verify actual output. |
| 26 | 🔴 NOT COMPLETE | No benchmark artifacts for latency/FPS/memory were generated. | Measure real-time performance and report results. |
| 27 | 🔴 NOT COMPLETE | API scaffold exists, but no verified model-backed prediction route was executed. | Connect FastAPI to a trained checkpoint. |
| 28 | 🔴 NOT COMPLETE | Tests exist, but they are mostly scaffold checks and not true scientific/regression tests. | Add real dataset, landmark, sequence, and robustness validations. |
| 29 | 🟡 PARTIALLY COMPLETE | Docker files and CI config exist, but the environment is not currently installable/validated here. | Fix and validate container and CI execution. |
| 30 | 🟡 PARTIALLY COMPLETE | Documentation exists, but much of it is generic or placeholder text and not grounded in the real dataset. | Rewrite the thesis and reproducibility docs around actual evidence. |

## Critical blockers

1. The configuration still assumes a toy dataset (5 labels, 20 signers, 16 frames, hands-only), which conflicts with the real WLASL inventory.
2. The local dataset is partially populated: 11,980 MP4 files are present, but the metadata includes 21,083 instances, leaving 9,103 entries absent from `data/raw/`.
3. Split metadata is not signer-independent: signers overlap across train/val/test.
4. `src/sign_language/data/dataset.py` still contains a synthetic dataset generator and synthetic feature generation logic.
5. The landmark extraction pipeline uses placeholder zeros and synthetic outputs instead of MediaPipe inference over the actual videos.
6. `data/manifests/`, `results/metrics/`, and `results/figures/` are still absent.
7. The environment does not currently have the project dependencies installed (`pytest` and core scientific libraries are missing).
8. The repository README claims the project is complete and validated even though no real end-to-end scientific validation has been run.
9. Duplicate data layer code remains in both `src/data/` and `src/sign_language/data/` without a single authoritative source.
10. No real experiment outputs or result files exist for the required research question.

## Technical debt and research risks

- Placeholder assumptions leaked into configuration, tests, and documentation.
- Synthetic fallback behavior masks missing dataset/landmark implementation details.
- The project is not yet reproducible because dependencies are not installed and no execution logs exist.
- Signer overlap prevents a valid signer-independent evaluation until a new split is created or the dataset source is clarified.
- The current code base is a research scaffold, not a validated pipeline.

## Recommended execution order

1. Fix configuration and dataset assumptions.
2. Build and validate the dataset manifest.
3. Create a real signer-aware split with explicit overlap validation.
4. Replace synthetic landmark generation with MediaPipe extraction over the actual dataset.
5. Implement normalization and sequence handling on true extracted assets.
6. Train and validate the baseline models on real data.
7. Add controlled robustness experiments and ablations.
8. Build the final real-time inference and API around a verified checkpoint.
9. Produce results tables, figures, and thesis-ready documentation.

## Audit conclusion

The repository contains a real local ASL dataset and a substantial research scaffold, but it is not yet an end-to-end validated system. A bounded real-data manifest check now succeeds (`sample10_manifest.json`: one locally available video decoded; nine metadata references absent). The most urgent work is to replace placeholder configuration and synthetic data logic with real dataset-derived behavior before any model training or inference claims are made.
