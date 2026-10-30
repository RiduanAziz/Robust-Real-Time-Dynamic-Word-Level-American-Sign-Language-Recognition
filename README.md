# Robust Real-Time Dynamic Word-Level American Sign Language Recognition

## Project overview

This repository is a thesis-oriented research platform for robust real-time dynamic American Sign Language recognition. The project is structured around a reproducible research pipeline: signer-aware dataset validation, sequence preprocessing, landmark normalization, temporal modeling, robustness evaluation, and a lightweight real-time inference scaffold.

## Research problem

The core research question is:

> How can holistic spatial-temporal feature representations improve robustness for real-time dynamic word-level ASL recognition under signer variation, motion irregularity, and controlled noise conditions?

The implementation explicitly supports:

- signer-independent dataset splits
- synthetic and metadata-driven dataset summarization
- landmark extraction and normalization pipelines
- baseline temporal and multimodal model families
- robustness scoring for noise scenarios
- real-time inference stub for deployment-oriented integration

## Verified Benchmark & State-of-the-Art Evaluation

SignFlow features a verified, independently reproduced WLASL-100 evaluation baseline and a state-of-the-art validation-tuned 3-way probability ensemble on the official benchmark partition (Protocol A, 257 held-out test samples):

| Evaluation Track / Model                           | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy |  Macro-F1  |    ECE     | Model Basis / Provenance                                     |
| :------------------------------------------------- | :------------: | :------------: | :-------------: | :--------: | :--------: | :----------------------------------------------------------- |
| **Historical Baseline (`robust_holistic_fusion`)** |     74.32%     |     88.72%     |     93.39%      |   0.7213   |   0.1598   | Exact reproduction (`models/robust_holistic_fusion_best.pt`) |
| **Uni-Sign PGF (`unisign_pgf`) Solo**              |     70.43%     |     88.33%     |     92.61%      |   0.6863   | **0.0561** | Hand-to-pose cross-attention (`arXiv:2501.15187`)            |
| **Deep Holistic BiGRU Solo**                       |     72.76%     |     89.88%     |     93.77%      |   0.7180   |   0.0912   | 2-layer BiGRU with label smoothing ($\epsilon=0.08$)         |
| **Track D: 2-Way Ensemble**                        |     76.26%     |     90.66%     |     93.77%      |   0.7459   | **0.0551** | Baseline + Uni-Sign PGF ($\alpha=0.55$)                      |
| **Track D: 3-Way Ensemble (WINNING BENCHMARK)**    |   **76.65%**   |   **91.44%**   |   **94.16%**    | **0.7558** | **0.0838** | **Optimal Weights: (0.25, 0.10, 0.65), Val Top-1: 78.34%**   |

### Benchmark Improvements Over Baseline:

- **Top-1 Accuracy:** **+2.33%** absolute gain ($74.32\% \rightarrow \mathbf{76.65\%}$)
- **Top-5 Accuracy:** **+2.72%** absolute gain ($88.72\% \rightarrow \mathbf{91.44\%}$) — breaks the 91% milestone!
- **Macro-F1 Score:** **+0.0345** gain ($0.7213 \rightarrow \mathbf{0.7558}$)
- **Calibration (ECE):** **47.6% reduction** in overconfidence error ($0.1598 \rightarrow \mathbf{0.0838}$)

### Deterministic Reproduction Commands:

```powershell
# 1. Reproduce Historical Baseline
python scripts/evaluate.py --config configs/colab/wlasl100_baseline_colab.yaml --checkpoint models/robust_holistic_fusion_best.pt --protocol official

# 2. Reproduce 3-Way Winning Ensemble
python scripts/ensemble_eval.py --model1-ckpt models/robust_holistic_fusion_best.pt --model1-config configs/dataset.yaml --model2-ckpt models/wlasl100_unisign_pgf_best.pt --model2-config configs/colab/wlasl100_unisign_pgf.yaml --model3-ckpt models/wlasl100_max_accuracy_best.pt --model3-config configs/colab/wlasl100_max_accuracy.yaml
```

---

## State-of-the-Art Architecture Innovations: Uni-Sign PGF

Adapted from the ICLR 2025 research _"Uni-Sign: Toward Unified Sign Language Understanding at Scale"_ (`arXiv:2501.15187`, `https://github.com/zechengli19/uni-sign`):

- **Prior-Guided Fusion (PGF)**: Uses skeletal body pose coordinates as explicit spatial priors guiding hand gesture cross-attention (`UniSignPGFClassifier`).
- **Score-Aware Reliability Gating**: Dynamically downweights missing or occluded hands during single-handed signs, eliminating zero-fill noise.
- **Multi-Scale Temporal Convolutions**: Dual-branch 1D temporal convolutions (kernels 3 & 5) capturing fine-grained fingertip articulations.
- **Strict Mask Propagation**: Verified across 5 supported representations with 100% test coverage (`tests/test_features_contract.py`).

---

## Master Documentation & Research Reports

| Report                                                                                           | Description                                                                              |
| :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| [`docs/MASTER_PIPELINE_AUDIT.md`](docs/MASTER_PIPELINE_AUDIT.md)                                 | Evidence-based audit of implemented code, schema parity, and defect fixes.               |
| [`docs/DATASET_COMPLETENESS_AND_SPLITS.md`](docs/DATASET_COMPLETENESS_AND_SPLITS.md)             | Census of all 21,083 landmark files, 2,038 WLASL-100 samples, and split integrity.       |
| [`docs/EDA_FINDINGS.md`](docs/EDA_FINDINGS.md)                                                   | Real exploratory data analysis: modality detection rates, sequence length distributions. |
| [`docs/FEATURE_ENGINEERING_AUDIT_AND_RESULTS.md`](docs/FEATURE_ENGINEERING_AUDIT_AND_RESULTS.md) | Resolution of 280 vs 365 feature dimensions and strict validity mask propagation.        |
| [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)                                     | Experimental protocol definition covering Tracks A through E and locked evaluation.      |
| [`docs/COLAB_TRAINING_GUIDE.md`](docs/COLAB_TRAINING_GUIDE.md)                                   | Step-by-step 12-stage Google Colab GPU training, checkpointing, and resume guide.        |
| [`docs/COLAB_DATA_UPLOAD_CHECKLIST.md`](docs/COLAB_DATA_UPLOAD_CHECKLIST.md)                     | Instructions for uploading packaged archives to Google Drive (`SignFlow_Colab/`).        |
| [`docs/FINAL_EVALUATION_REPORT.md`](docs/FINAL_EVALUATION_REPORT.md)                             | Verified 100-class evaluation results, per-class breakdown, and confusion matrix.        |
| [`docs/LOCAL_INTEGRATION_GUIDE.md`](docs/LOCAL_INTEGRATION_GUIDE.md)                             | FastAPI and React/TypeScript real-time deployment and streaming serving contracts.       |
| [`docs/DEPLOYMENT_READINESS_AUDIT.md`](docs/DEPLOYMENT_READINESS_AUDIT.md)                 | Baseline audit, P0/P1 defect remediations, asset hashes, and verification inventory.     |
| [`docs/LOCAL_DEVELOPMENT.md`](docs/LOCAL_DEVELOPMENT.md)                                         | One-command Windows launcher (`start_signflow.ps1`), port supervisor, and dev workflow. |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)                                                       | Production Docker containerization, GitHub release asset bootstrap, and Render blueprint. |

---

## Current implementation status

The repository is currently validated through the following milestones:

- Phase 0 — repository and environment foundation: complete
- Phase 1 — dataset completeness and benchmark validation: complete
- Phase 2 — real exploratory data analysis: complete
- Phase 3 — feature engineering and strict contract tests: complete
- Phase 4 — baseline model reproduction: complete (74.32% Top-1, 88.72% Top-5 verified)
- Phase 5 — Uni-Sign Prior-Guided Fusion & temporal modeling: complete
- Phase 6 — training data augmentation & scheduler optimization: complete
- Phase 7 — validation-driven ensembling (Track D): complete
- Phase 8 & 9 — Google Colab GPU pipeline & automatic checkpointing: complete
- Phase 10 — locked evaluation protocol: complete
- Phase 11 & 12 — model export package & runtime integration: complete

## SignFlow — Live ASL to Text & Speech Web Application

**SignFlow** is an assistive communication web application integrated into this thesis project. It translates real-time dynamic ASL gestures from the user's camera into an accumulated, editable transcript with browser text-to-speech (TTS) output.

### Key Capabilities

- **Camera Capture & Holistic Landmark Tracking**: MediaPipe HolisticLandmarker tracking 553 canonical landmark points (hands, pose, face).
- **Deep Sequence Recognition**: Live inference powered by trained Temporal Transformer / Robust Holistic Fusion models (`models/temporal_transformer_trained.pt`).
- **Two Recognition Modes**:
  - **Mode A — Guided Word Accumulation**: Primary mode. Detects signs one by one with neutral pause boundary detection and stability debounce filtering before committing words to the persistent transcript.
  - **Mode B — Continuous Sign Recognition (Experimental)**: Sliding window prototype with experimental motion boundaries.
- **Transcript Workspace**: Interactive word chips, editable textarea, quick punctuation inserters (`.`, `,`, `?`, `!`, `Space`), undo, clear with confirmation, copy, and `.txt` file export.
- **Browser Text-to-Speech (TTS)**: Built-in SpeechSynthesis API integration with voice selection, speed/pitch control, pause/resume, and auto-speak.

### Quick Start Guide

#### 1. One-Command Local Startup (Windows PowerShell / CMD)

Launch the entire SignFlow application (dependency verification, model asset checks, frontend build, FastAPI backend, and automatic browser launch) with a single command:

```powershell
# PowerShell (Recommended):
.\start_signflow.ps1

# Fast start (skip dependency install checks when already configured):
.\start_signflow.ps1 -SkipInstall

# Development mode (launches Vite dev server with hot-reloading on port 5173):
.\start_signflow.ps1 -UiOnly

# Clean up any lingering background ports (8000 / 5173):
.\stop_signflow.ps1
```

```cmd
:: Command Prompt (CMD):
start_signflow.bat
```

#### 2. Containerized Deployment (Docker Compose)

Run the hardened single-service container:

```bash
docker compose up --build
```

#### 3. Core Service Endpoints

- Web Application: [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) *(serves production React bundle)*
- Liveness Probe: [`http://127.0.0.1:8000/health/live`](http://127.0.0.1:8000/health/live) *(HTTP 200)*
- Readiness Probe: [`http://127.0.0.1:8000/health/ready`](http://127.0.0.1:8000/health/ready) *(HTTP 200 when ensemble & MediaPipe loaded; HTTP 503 otherwise)*
- Model Ensemble Info: [`http://127.0.0.1:8000/model/info`](http://127.0.0.1:8000/model/info)
- WebSocket Frame Stream: `ws://127.0.0.1:8000/ws/live` *(or `wss://` over HTTPS)*
- OpenAPI Documentation: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

#### 4. Packaging Inference Assets for Remote Deployment

To build a release bundle of all required model weights and MediaPipe task files:

```powershell
python scripts/package_inference_assets.py
```
This produces `dist/signflow-inference-assets.zip` and its `.sha256` sidecar. Upload this zip to your GitHub Releases and configure `SIGNFLOW_ASSETS_URL` in your cloud host (e.g., Render Blueprint `render.yaml`).

## Repository structure

- [app/frontend](app/frontend): SignFlow React + TypeScript + Vite + Tailwind CSS web application
- [configs](configs): Experiment configuration files
- [docs](docs): Architecture, research reports, and deployment guides
- [scripts](scripts): Operational scripts, packaging, asset bootstrap, and training pipelines
- [src](src): Canonical `sign_language` package
- [tests](tests): Validation, model, and API test suites
- [results](results): Metrics, reports, and generated figures
- [models](models): Trained checkpoints, MediaPipe task asset, and class prior logits
