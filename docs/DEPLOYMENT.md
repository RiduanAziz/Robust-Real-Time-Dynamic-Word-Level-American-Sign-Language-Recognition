# SignFlow Production Deployment Guide

This guide describes how to deploy the **SignFlow Real-Time Dynamic ASL Recognition** platform as a production-ready containerized service.

---

## 1. System Architecture

SignFlow uses a **single-service container architecture**:

```
+------------------------------------------------------------------------+
|                      Docker Container (Linux x86_64)                   |
|                                                                        |
|   +-----------------------+              +-------------------------+   |
|   |   Static Frontend     |              |   FastAPI REST Engine   |   |
|   |   (React 18 + Vite)   |  Served At   |   - /health/live        |   |
|   |   app/frontend/dist/  |<-------------|   - /health/ready       |   |
|   +-----------------------+    Route /   |   - /model/info         |   |
|                                          +-------------------------+   |
|                                                       |                |
|   +---------------------------------------------------+            |   |
|   |                                                                    |
|   v                                                                    |
| +------------------------------------------------------------------+   |
| |                    WebSocket Server (/ws/live)                   |   |
| |  - Decodes base64/binary frames from browser camera              |   |
| |  - Runs MediaPipe Holistic Landmarker (553 coordinates)          |   |
| |  - Evaluates 3-Way Ensemble (Fusion + Uni-Sign PGF + BiGRU)      |   |
| |  - Streams recognized signs & confidence back to client          |   |
| +------------------------------------------------------------------+   |
+------------------------------------------------------------------------+
```

### Key Architectural Strengths:
- **Zero CORS Issues in Production**: The React frontend and WebSocket live endpoint run on the **same origin and port**.
- **Dynamic Port Binding**: The Docker container automatically reads `$PORT` (default: `8000`) and binds to `0.0.0.0:${PORT}` for seamless compatibility with cloud PaaS providers (Render, Cloud Run, Fly.io, Railway).
- **Dynamic Secure WebSocket (`wss://`)**: The frontend dynamically resolves its WebSocket target from `window.location.host`. When accessed via HTTPS (e.g., `https://signflow.onrender.com`), the client automatically connects to `wss://signflow.onrender.com/ws/live`.

---

## 2. Resource & Sizing Requirements

| Metric | Minimum (Single Model) | Recommended (3-Way Ensemble) |
| :--- | :--- | :--- |
| **CPU** | 1 vCPU | 2 vCPU |
| **RAM** | 768 MB | 1.5 GB – 2.0 GB |
| **Storage** | 2 GB | 3 GB |
| **Cloud Tier Example** | Render *Starter* / Fly.io 1GB | Render *Standard* (2 GB) / AWS t3.small |

> **Low-Memory Optimization:** If running on a 512 MB memory tier (e.g., Render Free/Starter), set environment variable `SIGNFLOW_MODEL_SET=robust_fusion`. This loads only the lightweight 4.2 MB baseline model instead of the full 3-way ensemble, keeping memory usage well below 500 MB.

---

## 3. Inference Asset Distribution

Because trained checkpoint weights (`*.pt`) and MediaPipe task files (`*.task`) are excluded from Git version control, SignFlow provides an automated packaging and bootstrap pipeline.

### Step 1: Package Assets (Local)
On your development machine where trained models reside:
```powershell
python scripts/package_inference_assets.py
```
This generates:
- `dist/signflow-inference-assets.zip` (~61.6 MB)
- `dist/signflow-inference-assets.zip.sha256`

### Step 2: Upload to GitHub Releases
1. Create a release tag on GitHub (e.g., `v1.0.0-assets`).
2. Attach `dist/signflow-inference-assets.zip` to the release.
3. Copy the public asset direct download URL:
   `https://github.com/<owner>/<repo>/releases/download/v1.0.0-assets/signflow-inference-assets.zip`

### Step 3: Configure Cloud Environment
Add the following environment variable to your deployment service:
```bash
SIGNFLOW_ASSETS_URL=https://github.com/<owner>/<repo>/releases/download/v1.0.0-assets/signflow-inference-assets.zip
```
*(If the repository or release is private, also supply `SIGNFLOW_ASSETS_TOKEN=<github_pat_token>`)*

During container startup, `scripts/bootstrap_inference_assets.py` will automatically:
1. Verify whether models are already mounted or present locally.
2. If missing, securely download the zip archive.
3. Verify the file integrity against built-in SHA-256 hashes.
4. Atomically extract models into `models/`.

---

## 4. Deploying to Render (Recommended)

Render provides native Docker support, HTTPS termination, and WebSocket streaming.

### Method A: Git-Connected Blueprint (`render.yaml`)

1. Ensure `render.yaml` is committed to your `main` branch.
2. Navigate to [dashboard.render.com](https://dashboard.render.com).
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically read `render.yaml` and configure the web service:
   - **Type:** Docker Web Service
   - **Plan:** Standard (2 GB RAM)
   - **Health Check Path:** `/health/ready`
   - **Port:** Automatically managed via `$PORT`
6. Under **Environment Variables**, set:
   - `SIGNFLOW_ASSETS_URL` = `<your_release_zip_url>`
7. Click **Apply**.

### Method B: Manual Web Service on Render

1. Click **New +** -> **Web Service**.
2. Select your repository.
3. Select **Docker** environment.
4. Set Health Check path to: `/health/ready`.
5. Select **Standard** plan (or **Starter** with `SIGNFLOW_MODEL_SET=robust_fusion`).
6. Add environment variables:
   - `PORT`: `8000`
   - `SIGNFLOW_MODEL_SET`: `ensemble_3way`
   - `SIGNFLOW_ASSETS_URL`: `<your_release_zip_url>`
7. Deploy.

---

## 5. Self-Hosted Docker / Compose Deployment

For dedicated Linux servers (Ubuntu / Debian / RHEL / Docker Engine):

### Step 1: Clone Repository
```bash
git clone https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition.git signflow
cd signflow
```

### Step 2: Bootstrap Assets
Place your `models/` directory in the repository root, or run:
```bash
python3 scripts/bootstrap_inference_assets.py --url "<https://github.com/RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition/releases/tag/v1.0.0-assets>"
```

### Step 3: Launch with Docker Compose
```bash
docker compose up -d --build
```

### Step 4: Verify Deployment
```bash
# Check liveness
curl -I http://localhost:8000/health/live

# Check readiness (models loaded)
curl -I http://localhost:8000/health/ready

# View live container logs
docker compose logs -f api
```

---

## 6. Mobile Testing (iOS & Android)

To test the application from a smartphone:

1. Deploy the service to an HTTPS provider (such as Render) or use a reverse proxy with a valid SSL certificate.
   > **Note:** Modern mobile browsers (iOS Safari, Android Chrome) **require HTTPS** to grant camera permissions (`navigator.mediaDevices.getUserMedia`).
2. Open the deployed URL (e.g. `https://signflow.onrender.com`) on your mobile browser.
3. When prompted, tap **Allow** for Camera access.
4. The frontend will dynamically establish a secure WebSocket stream (`wss://.../ws/live`).
5. Ensure good lighting and hold your phone steady or place it on a stand while performing signs.

---

## 7. Privacy & Security Safeguards

- **Frame Encryption:** In production, all video frames sent to `/ws/live` are encrypted in transit over standard TLS/HTTPS (`wss://`).
- **Zero Video Retention:** Video frames are decoded in memory, processed through MediaPipe and PyTorch inference pipelines, and immediately discarded. Frames are never stored on disk.
- **Connection Limits:** The API enforces a concurrency cap (`SIGNFLOW_MAX_WS_CONNECTIONS`, default: 50) and drops incoming frame payloads larger than 2 MB (`MAX_FRAME_BYTES`).
- **Non-Root Execution:** The production Docker image runs as an unprivileged user (`signflow`, UID 1000).



---

## 8. 100% Free Zero-Cost Deployments (No Credit Card Required)

Cloud platforms like Render, Fly.io, and Railway increasingly require credit card verification even for "free" tiers to prevent abuse. Furthermore, free tiers typically provide only **512 MB RAM**, which causes PyTorch and MediaPipe to crash due to Out-Of-Memory (`OOMKilled`) errors.

For academic defenses, supervisor demos, and examiner reviews, use either of the two **100% free, zero-card** methods below:

### Method 1: Instant Cloudflare Quick Tunnel (Recommended for Demos & Defenses)

Cloudflare provides free, unlimited secure tunnels with **no account, no credit card, and no setup**.

- **How it works:** SignFlow runs locally on your computer with full CPU/GPU and RAM, and Cloudflare exposes an instant public `https://*.trycloudflare.com` domain with valid SSL encryption.
- **Mobile Camera:** Fully functional (mobile browsers require HTTPS to enable camera permissions).
- **WebSocket:** Natively supported (`wss://` automatic routing).

#### How to Launch:
Double-click `share_signflow.bat` or run:
```powershell
.\start_signflow.ps1 -Public
```

The script will:
1. Verify inference models and build the frontend bundle.
2. Start the FastAPI backend and verify health.
3. Establish a Cloudflare edge tunnel.
4. Output your **Public HTTPS Domain** and automatically copy it to your clipboard.
5. You can now send this URL to your supervisor or open it on your phone!

To stop the tunnel and clean up all ports, press `Ctrl+C` or run:
```powershell
.\stop_signflow.ps1
```

---

### Method 2: Hugging Face Spaces (Free 24/7 Cloud Host, 16 GB RAM, No Card)

Hugging Face Spaces is the premier free cloud hosting platform for ML/AI research:
- **Cost:** 100% Free forever.
- **Credit Card Required:** **NO** (Only a free email account).
- **Hardware:** 2 vCPU + **16 GB RAM** (plenty of room for the 3-Way Ensemble and MediaPipe).
- **Public Domain:** Free permanent `https://<user>-signflow.hf.space` HTTPS domain.

#### Step-by-Step Hugging Face Spaces Deployment:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** $\to$ name it `signflow`.
3. Space SDK: Choose **Docker** $\to$ **Blank**.
4. Set Space hardware to **CPU basic (free, 16GB RAM)**.
5. Clone the space repository locally or push your SignFlow repository to the space Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/signflow
   git push space main
   ```
6. In Space Settings $\to$ **Variables and Secrets**, set `SIGNFLOW_ASSETS_URL` to your GitHub Release asset zip URL.
7. Hugging Face will build the Dockerfile and deploy your app with a permanent, public HTTPS link that you can share with anyone!
