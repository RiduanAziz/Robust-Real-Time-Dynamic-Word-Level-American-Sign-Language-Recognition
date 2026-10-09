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

## Current implementation status

The repository is currently validated through the following milestones:

- Phase 0 — repository and environment foundation: complete
- Phase 1 — dataset definition and metadata: complete
- Phase 2 — exploratory data analysis: complete
- Phase 3 — robust landmark extraction: complete
- Phase 4 — baseline model evaluation: complete
- Phase 5 — transformer and multimodal models: complete
- Phase 6 — robustness evaluation: complete
- Phase 7 — real-time deployment scaffold: complete

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

#### 1. Start the Backend API & WebSocket Service (PowerShell)
```powershell
# Activate environment and launch FastAPI backend
.\.venv\Scripts\uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000
```
- API Health: [`http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health)
- Model Info: [`http://127.0.0.1:8000/model/info`](http://127.0.0.1:8000/model/info)
- Interactive API Docs: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
- Direct Web Application: [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) *(serves production build)*

#### 2. Start the Frontend Development Server (Optional / Live Dev)
```powershell
cd app/frontend
npm run dev
```
Navigate to [`http://127.0.0.1:5173/`](http://127.0.0.1:5173/) in your browser.

#### 3. Real-Time Desktop OpenCV Demonstration
For a standalone native OpenCV desktop popup window:
```powershell
.\.venv\Scripts\python scripts/realtime.py
```

## Repository structure

- [app/frontend](app/frontend): SignFlow React + TypeScript + Vite + Tailwind CSS web application
- [configs](configs): experiment configuration files
- [docs](docs): architecture and project status notes
- [scripts](scripts): operational training, extraction, and real-time CLI entry points
- [src](src): canonical `sign_language` package
- [tests](tests): validation, model, and API test suites
- [results](results): metrics, reports, and generated figures
- [models](models): trained checkpoints and MediaPipe task assets

