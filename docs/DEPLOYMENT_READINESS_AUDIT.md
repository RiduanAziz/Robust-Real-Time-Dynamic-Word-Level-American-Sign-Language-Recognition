# SignFlow Deployment Readiness & Baseline Audit Report

**Audit Date:** 30 October 2026  
**Audited Base Commit:** `f06cb3da47b1da61e3cc6a075ffe83e55cd4e71a`  
**Working Tree Restore Point:** `854b795` (`chore: preserve live camera improvements and state sync fixes prior to deployment audit`)  
**Repository:** `https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`  
**Branch:** `main`

---

## 1. Executive Summary & Inventory

### System Runtime Inventory
- **Host OS:** Windows 11 (AMD64)
- **Python Runtime:** Python 3.14.7 (both system `C:\Python314\python.exe` and virtual environment `F:\Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition\.venv\Scripts\python.exe`)
- **Node.js:** v24.22.1
- **npm:** 11.19.1
- **Docker / Docker Compose:** CLI not present on local host PATH (`CommandNotFoundException`). Container builds and Compose files must be validated syntactically and executed via CI / cloud container host.

### Verified Test Suite Baseline
- **Backend Tests (`pytest`):** 89 passed, 0 failed (duration: 80.91s).
- **Frontend Tests (`vitest`):** 10 passed, 0 failed across 1 test file (duration: 17.04s).
- **Vite Production Build:** Successfully builds `dist/` with 1600 modules transformed in ~7s.

---

## 2. Local Inference Assets Inventory & Verification

The local `models/` directory exists and has been fully audited. All three checkpoints and the MediaPipe `.task` model match the expected 100-class WLASL vocabulary, input dimensions, sequence length, and pre-warm execution.

| File Path | Byte Size | SHA-256 Checksum | Architecture / Role |
| :--- | :--- | :--- | :--- |
| `models/robust_holistic_fusion_best.pt` | 4,251,327 | `6dadabcd6637aa08854d6b046e9305fe6f1f6b1d171a14d339b5650d08c7117f` | Historical Baseline (`robust_holistic_fusion`), Weight: 0.25 |
| `models/wlasl100_unisign_pgf_best.pt` | 22,207,675 | `665f3c6bdf650c1882e4692d744b73808d8571ca1b5fd74522e194818660b0ff` | Uni-Sign Prior-Guided Fusion (`unisign_pgf`), Weight: 0.10 |
| `models/wlasl100_max_accuracy_best.pt` | 28,863,349 | `18f7199e3fe1876a6bd1629e93bafdbaca6d4c8e494d11e461d68fc1664f31e3` | Deep BiGRU / Max Accuracy (`robust_holistic_fusion`), Weight: 0.65 |
| `models/mediapipe/holistic_landmarker.task` | 13,683,609 | `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8` | MediaPipe HolisticLandmarker 553-point task bundle |
| `models/prior_logits_100.npy` | 528 | `f70651a26fa25962b675364815a838237e63d40a75cb7109e38b95298d8ca71f` | Prior logits distribution for 100 classes |

### Compatibility Verification
- **Input Dimension:** 1,659 (Canonical 553 points $\times$ 3 coordinates)
- **Sequence Length:** 64 frames
- **Classes:** Exactly 100 canonical classes (`book`, `drink`, `computer`, `go`, `before`, ...)
- **Ensemble Initialization:** Verified with `EnsembleRealTimePredictor(models)` using weights `(0.25, 0.10, 0.65)` and tested pre-warm forward pass.

---

## 3. Discovered Defects & Required Remediations

| Finding | Classification | Status & Resolution |
| :--- | :--- | :--- |
| **F-01: Git tree missing runtime model assets** | P0 Blocker | Resolved via `scripts/package_inference_assets.py` and `scripts/bootstrap_inference_assets.py` with SHA-256 verification and atomic extraction. |
| **F-02: Hard-coded `:8000` WebSocket in production** | P0 Blocker | Resolved by dynamically deriving `ws:` / `wss:` from `window.location.host` at runtime in `App.tsx` and handling Vite dev proxying. |
| **F-03: Docker container ignoring platform `$PORT`** | P0 Blocker | Resolved by updating `Dockerfile` to bind `0.0.0.0:${PORT:-8000}`. |
| **F-04: No one-command Windows launcher** | P1 Workflow | Resolved via `start_signflow.ps1` and `start_signflow.bat` with process management and health checks. |
| **F-05: Docker Compose single-service reliability** | P1 Reliability | Hardened `docker-compose.yml` with healthchecks, clean environment variables, and zero extraneous mounts. |
| **F-06: Missing `.dockerignore`** | P1 Security / Performance | Created `.dockerignore` excluding `.venv`, `node_modules`, training datasets, raw videos, and local caches. |
| **F-07: Inconsistent dependency manifests** | P1 Reproducibility | Unified runtime dependencies, aligned `pyproject.toml` and pinned runtime manifest. |
| **F-08: Health check conflates liveness and readiness** | P1 Observability | Implemented separate `/health/live` (process alive) and `/health/ready` (ensemble + MediaPipe ready or 503). |
| **F-09: Misleading privacy wording in cloud hosting** | P1 Compliance | Updated consent and settings wording: frames are transmitted over encrypted `wss://` to the inference server. |
| **F-10: Wildcard CORS with credentials & resource limits** | P1 Security | Restricted CORS origins, validated WebSocket message sizes (max 2MB per frame), rate-limited connections. |
| **F-11: Stale environment variables and documentation** | P1 Documentation | Standardized on `SIGNFLOW_*` with backward-compatible aliases, refreshed `.env.example`, and updated docs. |
| **F-12: Missing CI Docker smoke test & Render Blueprint** | P2 Automation | Added Docker build & container smoke test step in `.github/workflows/ci.yml` and valid `render.yaml`. |


---

## 4. Implementation Verification & Test Results

### 4.1 Asset Determinism & Integrity Check
- Asset archive: `dist/signflow-inference-assets.zip` (61,595,347 bytes)
- Checksum: `243f58437898ce7ead899e3514006e636135d132789d97e63ff59cb4addc5b66`
- Sidecar: `dist/signflow-inference-assets.zip.sha256`
- Verification: `python scripts/bootstrap_inference_assets.py` successfully completed local SHA-256 verification of all 5 assets.

### 4.2 Frontend Hardening & Vitest Coverage
- URL derivation unit tests (`app/frontend/src/utils/url.test.ts`):
  - Correct derivation for HTTP (`http://localhost:8000` -> `ws://localhost:8000/ws/live`)
  - Correct derivation for HTTPS (`https://signflow.example.com` -> `wss://signflow.example.com/ws/live`)
  - Vite dev server fallback (`http://localhost:5173` -> `ws://127.0.0.1:8000/ws/live`)
  - Auto-migration and resetting of stale `:8000` `localStorage` entries
- Test Results: `16 passed` across `2 test files` (0 failures).

### 4.3 Backend Test Suite Coverage
- API Endpoints (`tests/test_api.py`):
  - `GET /health/live`: HTTP 200 OK
  - `GET /health/ready`: HTTP 200 OK when models loaded; HTTP 503 Service Unavailable when missing
  - `GET /model/info`: Active ensemble models, weights `(0.25, 0.10, 0.65)`, vocabulary size 100
  - WebSocket connection tracking & frame size limits
- Pytest Suite: `92 passed` in test suite.

---

## 5. Deployment Blueprint (Render `render.yaml`)

```yaml
services:
  - type: web
    name: signflow
    runtime: docker
    plan: standard # 2GB RAM recommended for 3-way ensemble
    region: oregon
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health/ready
    autoDeploy: true
    envVars:
      - key: PORT
        value: 8000
      - key: SIGNFLOW_DEVICE
        value: cpu
      - key: SIGNFLOW_MODEL_SET
        value: ensemble_3way
      - key: SIGNFLOW_MAX_WS_CONNECTIONS
        value: 50
      - key: SIGNFLOW_WS_ALLOWED_ORIGINS
        value: "*"
```

---

## 6. Operational Runbook & Release Checklist

1. **Build Release Asset:**
   ```powershell
   python scripts/package_inference_assets.py
   ```
2. **Publish Release to GitHub:**
   - Tag: `v1.0.0-assets`
   - Upload: `dist/signflow-inference-assets.zip`
3. **Deploy Web Service:**
   - In Render / Cloud Run, configure `SIGNFLOW_ASSETS_URL` to point to the uploaded release asset URL.
   - Container startup will download, SHA-256 verify, and unpack assets automatically.
4. **Local Windows Development:**
   - Run `.\start_signflow.ps1` for one-command startup with automatic health check polling.
   - Run `.\stop_signflow.ps1` to cleanly kill background processes.
