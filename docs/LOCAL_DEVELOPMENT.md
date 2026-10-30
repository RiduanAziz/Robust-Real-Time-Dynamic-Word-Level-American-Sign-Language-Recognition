# SignFlow Local Development Guide

This guide covers running, developing, and testing the **SignFlow Real-Time ASL Recognition** platform on local machines (Windows PowerShell / CMD / Linux).

---

## Architecture Overview

SignFlow consists of two core components:
1. **Python FastAPI Backend (`src/sign_language/api/main.py`)**:
   - Manages the MediaPipe Holistic landmark extraction pipeline.
   - Hosts the 3-Way Ensemble inference engine (`robust_holistic_fusion`, `unisign_pgf`, `max_accuracy`).
   - Serves real-time bidirectional frame streaming via WebSocket at `/ws/live`.
   - Exposes REST endpoints (`/health/live`, `/health/ready`, `/model/info`).
   - Serves the compiled React bundle at root `/` when `app/frontend/dist` is built.

2. **React + TypeScript + Vite Frontend (`app/frontend/`)**:
   - Captures camera video stream.
   - Renders landmark overlays, live confidence scores, and predictions.
   - Provides transcript workspace with continuous and guided accumulation modes.
   - Speech synthesis (browser TTS) and practice vocabulary modes.

---

## One-Command Local Startup (Recommended)

### Using PowerShell (Windows)

From the project root:
```powershell
.\start_signflow.ps1
```

The script will:
1. Validate Python (>=3.10) and Node.js (>=18).
2. Create or verify Python virtual environment (`.venv`).
3. Verify or bootstrap model checkpoints in `models/`.
4. Ensure frontend dependencies are installed (`npm ci` / `npm install`).
5. Build the React frontend production bundle to `app/frontend/dist`.
6. Launch FastAPI backend on `http://127.0.0.1:8000`.
7. Poll `/health/live` until the server is ready.
8. Automatically open your default web browser to `http://127.0.0.1:8000`.
9. Trap `Ctrl+C` to cleanly terminate all background child processes.

#### Supported PowerShell Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-Port <int>` | `8000` | Port for the backend API and single-service web UI. |
| `-SkipInstall` | `false` | Skip `pip install` and `npm install` checks for faster startup. |
| `-NoBrowser` | `false` | Start servers without launching the web browser. |
| `-UiOnly` | `false` | Start both backend API and Vite live dev server (`npm run dev`) for hot-module reloading. |

**Examples:**
```powershell
# Fastest startup when dependencies and frontend are already built:
.\start_signflow.ps1 -SkipInstall

# Frontend UI hot-reloading development mode:
.\start_signflow.ps1 -UiOnly
```

### Using Command Prompt (Windows CMD)

Double-click or run from command prompt:
```cmd
start_signflow.bat
```

### Stopping & Port Cleanup

If a background uvicorn or Vite process gets stuck or holds a port:
```powershell
.\stop_signflow.ps1
```
This inspects ports `8000` and `5173` and safely terminates the associated Python or Node processes.

---

## Manual Development Mode (Frontend Hot-Reloading)

If you prefer running components in separate terminal windows:

### Terminal 1: Python Backend
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run uvicorn server
uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: React Frontend (Vite Dev Server)
```powershell
cd app/frontend
npm run dev
```
Open [`http://127.0.0.1:5173`](http://127.0.0.1:5173). Vite automatically proxies API calls and WebSocket connections to `127.0.0.1:8000`.

---

## Verifying Model Checkpoints

SignFlow requires the following files inside `models/`:
- `models/robust_holistic_fusion_best.pt` (Baseline, 4.2 MB)
- `models/wlasl100_unisign_pgf_best.pt` (Uni-Sign PGF, 22.2 MB)
- `models/wlasl100_max_accuracy_best.pt` (Deep BiGRU, 28.8 MB)
- `models/mediapipe/holistic_landmarker.task` (MediaPipe Holistic, 13.7 MB)
- `models/prior_logits_100.npy` (Vocabulary priors, 528 bytes)

To package or verify these assets:
```powershell
# Package existing assets into a deterministic distributable archive:
python scripts/package_inference_assets.py

# Verify local assets or download missing assets:
python scripts/bootstrap_inference_assets.py
```

---

## Running Test Suites

### Backend Unit & Integration Tests (pytest)
```powershell
pytest -q
```
Runs the 89+ test suite covering feature engineering, model architecture, ensemble voting, API endpoints (`/health/live`, `/health/ready`, `/model/info`), and strict tensor contracts.

### Frontend Unit Tests (Vitest)
```powershell
cd app/frontend
npm test -- --run
```
Runs the test suite for React components and URL derivation utilities (`url.test.ts`).

### Bytecode Compilation & Linting
```powershell
python -m compileall src scripts tests
```

---

## API Endpoints Reference

| Endpoint | Method | Purpose | Response |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Single-Page Application | Serves compiled `index.html` from `app/frontend/dist` |
| `/health/live` | `GET` | Liveness probe | `{"status": "alive", "service": "..."}` (200 OK) |
| `/health/ready` | `GET` | Readiness probe | `{"status": "ready", ...}` (200 OK) or `{"status": "unready"}` (503 Service Unavailable) |
| `/model/info` | `GET` | Active model ensemble info | Model names, ensemble weights `(0.25, 0.10, 0.65)`, vocabulary size (100) |
| `/ws/live` | `WebSocket` | Real-time camera frames | Accepts JPEG/Base64 frame; emits bounding boxes, landmarks, recognized word, confidence |
| `/docs` | `GET` | Interactive Swagger UI | OpenAPI documentation |

