# Master TODO

## Overall Status
- [~] Dataset verified (metadata inventory and bounded real-video validation complete)
- [ ] Real landmark extraction verified
- [ ] Baseline trained
- [ ] Evaluation completed
- [ ] Robustness experiments completed
- [ ] Proposed method completed
- [ ] Final signer-independent evaluation completed
- [ ] Real-time inference completed
- [ ] API completed
- [ ] Thesis documentation completed

## Phase 0 — Foundation
- [x] Python package structure exists in `src/`
- [x] `pyproject.toml` exists and declares the package metadata
- [x] `environment.yml` exists
- [x] `.env.example` exists
- [x] `.gitignore` exists
- [x] logging utilities exist under `src/sign_language/utils/`
- [x] device utility exists under `src/sign_language/utils/device.py`
- [x] seed utility exists under `src/sign_language/utils/seed.py`
- [x] basic CI config exists under `.github/workflows/`
- [x] repository documentation exists
- [ ] environment reproducibility is validated in this session

## Phase 1 — Architecture Consolidation
- [x] Canonical package selected as `src/sign_language/`
- [x] Compatibility wrapper exists under `src/data/`
- [ ] duplicate implementation is fully retired or enforced to a single source of truth
- [ ] imports and package layout validated against real usage

## Phase 2 — Configuration Management
- [ ] one authoritative configuration mechanism is enforced
- [ ] dataset configuration reflects the actual WLASL inventory
- [ ] model configuration is aligned with the research pipeline
- [ ] training/evaluation/robustness config is consistent with actual code
- [ ] all hard-coded dataset assumptions are removed

## Phase 3 — Actual Dataset Discovery
- [x] Local dataset root inspected: `data/raw/`
- [x] WLASL metadata file discovered: `data/WLASL_v0.3.json`
- [x] dataset inventory drafted from actual files and metadata
- [ ] final dataset inventory checklist is committed to results and docs

## Phase 4 — Dataset Manifest
- [~] bounded sample manifest exists at `data/manifests/sample10_manifest.json`
- [ ] full manifest is generated after complete metadata/file reconciliation
- [x] frame counts, FPS, and durations extracted from an actual local MP4
- [x] label mapping is deterministic from sorted WLASL glosses
- [~] signer mapping is read from WLASL metadata; 9 of the first 10 records are absent locally

## Phase 5 — Signer-independent Splitting
- [ ] signer IDs validated and recorded
- [ ] train/validation/test splits are mutually disjoint by signer
- [ ] overlap checks are enforced in code and tests
- [ ] split manifests generated under `data/manifests/`

## Phase 6 — Dataset Quality Analysis
- [ ] class distribution computed
- [ ] signer distribution computed
- [ ] sequence-length distribution computed
- [ ] figure outputs generated in `results/figures/`

## Phase 7 — Real MediaPipe Landmark Extraction
- [ ] actual video reader implemented for real MP4 files
- [ ] MediaPipe extraction verified on local dataset
- [ ] missing-landmark handling validated
- [ ] output schema matches documented expectations

## Phase 8 — Landmark Data Schema
- [ ] canonical schema definition documented
- [ ] coordinate ordering defined
- [ ] modality ordering defined
- [ ] missing-data semantics documented

## Phase 9 — Landmark Normalization
- [ ] translation normalization implemented and validated
- [ ] scale normalization implemented and validated
- [ ] missing-landmark handling tested
- [ ] dimensional consistency verified

## Phase 10 — Temporal Sequence Processing
- [ ] variable-length sequence handling implemented
- [ ] padding/truncation verified on realistic lengths
- [ ] attention masks or equivalent used for transformers
- [ ] collation validated with actual dataset samples

## Phase 11 — Feature Representations
- [ ] hands-only model version implemented
- [ ] hands + pose version implemented
- [ ] hands + pose + face version implemented
- [ ] holistic representation feature dimensions recorded

## Phase 12 — Real Baseline Dataset Loading
- [ ] real dataset class implemented
- [ ] train/validation/test manifests loaded
- [ ] missing-sample detection implemented
- [ ] synthetic dataset path restricted to smoke tests only

## Phase 13 — Baseline Models
- [ ] MLP baseline trained and validated
- [ ] LSTM baseline trained and validated
- [ ] GRU baseline trained and validated
- [ ] Transformer baseline trained and validated

## Phase 14 — Real Model Training
- [ ] actual training loop complete
- [ ] validation metrics recorded
- [ ] best checkpoint logic operating
- [ ] training logs and metadata persisted

## Phase 15 — Real Evaluation
- [ ] accuracy, precision, recall, macro F1, weighted F1 computed
- [ ] per-class metrics exported
- [ ] confusion matrix generated
- [ ] evaluation against actual checkpoint completed

## Phase 16 — Model Comparison
- [ ] baseline comparison table generated
- [ ] latency and parameter count recorded
- [ ] final comparison uses consistent protocol

## Phase 17 — Holistic / Multimodal Model
- [ ] hands only vs holistic vs multimodal comparisons built
- [ ] feature fusion developed and validated
- [ ] multimodal checkpoint evaluated on held-out data

## Phase 18 — Spatial Robustness
- [ ] jitter, translation, scaling, rotation, and dropout perturbations implemented
- [ ] deterministic severity controls validated
- [ ] model performance degradation measured

## Phase 19 — Temporal Robustness
- [ ] frame dropping, duplication, truncation, and speed variation implemented
- [ ] deterministic controls validated
- [ ] performance degradation measured

## Phase 20 — True Recognition Robustness
- [ ] clean vs noisy recognition metrics computed
- [ ] accuracy degradation and F1 degradation reported
- [ ] robustness metric definitions documented scientifically

## Phase 21 — Combined Spatial-Temporal Robustness
- [ ] severity-grid experiments built
- [ ] CSV/JSON outputs generated
- [ ] heatmap and summary statistics produced

## Phase 22 — Proposed Robust Method
- [ ] research hypothesis documented
- [ ] proposed approach implemented
- [ ] evidence generated against prior baselines

## Phase 23 — Ablation Study
- [ ] feature ablations implemented
- [ ] robustness ablations implemented
- [ ] final model selected from evidence, not loss alone

## Phase 24 — Final Signer-independent Evaluation
- [ ] unseen-signer test completed
- [ ] per-class and signer-wise diagnostics produced
- [ ] error analysis documented

## Phase 25 — Real-time Inference
- [ ] webcam pipeline built and tested
- [ ] sequence buffer, smoothing, and confidence logic validated
- [ ] actual model output used for prediction

## Phase 26 — Real-time Benchmark
- [ ] landmark extraction latency measured
- [ ] preprocessing latency measured
- [ ] end-to-end latency and FPS reported
- [ ] CPU/GPU benchmark recorded where available

## Phase 27 — FastAPI
- [ ] `POST /predict` route validated against real model input
- [ ] model loading is based on trained checkpoint, not placeholder logits
- [ ] output includes class and confidence

## Phase 28 — Testing
- [ ] dataset validation tests implemented
- [ ] landmark validation tests implemented
- [ ] sequence masking and padding tests implemented
- [ ] model checkpoint tests implemented
- [ ] robustness tests implemented

## Phase 29 — CI / Docker / Engineering
- [ ] pytest passes in a clean environment
- [ ] ruff/black checks pass
- [ ] Docker configuration verified for training and inference

## Phase 30 — Thesis-ready Documentation
- [ ] README clearly distinguishes implemented, validated, experimental, prototype, blocked, and planned work
- [ ] `docs/PROJECT_STATUS.md` is present and grounded in real evidence
- [ ] `docs/MASTER_TODO.md` is updated
- [ ] `docs/dataset.md` reflects the actual dataset and not placeholder values

## Critical Blockers
- [ ] actual dataset config is inconsistent with the local WLASL inventory
- [ ] signer overlap across metadata splits prevents valid signer-independent evaluation
- [ ] no complete real manifest exists for the live dataset (bounded sample exists)
- [ ] landmark extraction is still synthetic rather than MediaPipe-based
- [ ] dependencies and CI are not currently validated in this environment

## Research Risks
- [ ] claiming a research result before dataset validation and split verification
- [ ] training on synthetic data under the guise of real research execution
- [ ] using placeholder zeros/linspace-based landmarks as if they were extracted features
- [ ] reporting performance on unlabeled or mismatched dataset assumptions

## Current Next Action
- [ ] Replace the placeholder dataset assumptions in `configs/dataset.yaml` and generate a real manifest from the local WLASL videos and metadata.

## Evidence Notes
- Local WLASL metadata is real and includes 2,000 glosses and 119 signers.
- Local raw video inventory is present but incomplete relative to metadata.
- Current code still contains synthetic dataset generation and synthetic placeholder landmark generation.
- The environment is not yet prepared for an end-to-end model run.
