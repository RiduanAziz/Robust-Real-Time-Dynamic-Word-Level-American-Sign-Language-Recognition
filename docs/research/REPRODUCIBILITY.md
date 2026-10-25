# SignFlow Reproducibility Guide

This guide provides exact, deterministic commands to reproduce dataset audits, model training, evaluation, and real-time benchmarks.

---

## 1. Environment & Hardware Specifications

- **Python Version**: 3.10+ (tested on Python 3.10, 3.11, 3.14)
- **Node.js**: v18+ (tested with Vite 6.4.4, TypeScript 5.7)
- **PyTorch**: 2.0+ (CUDA 11.8 / 12.1 supported on Colab; CPU runtime supported locally)
- **MediaPipe**: 0.10.x / 1.0.x (Holistic Landmarker task model)
- **Operating Systems**: Windows 11 / Linux (Ubuntu 22.04 LTS)

---

## 2. Deterministic Seeds & Provenance

All training experiments fix global seeds for `random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` (default `seed: 42`). Checkpoints automatically serialize:
- Architecture name and configuration parameters.
- Ordered class vocabulary list.
- Deterministic SHA256 `vocabulary_hash`.
- Landmark schema version (`v2.0`).

---

## 3. Dataset Integrity & Leakage Verification

### Protocol A (Official Splits) Verification:
```bash
python scripts/validate_dataset.py --protocol official
```

### Protocol B (Signer-Disjoint Splits) Strict Verification:
```bash
python scripts/validate_dataset.py --protocol signer_disjoint --strict
```

### Generate New Signer-Disjoint Splits from Scratch:
```bash
python scripts/generate_signer_disjoint_split.py \
    --input data/manifests/full_manifest.json \
    --output-dir data/manifests/signer_disjoint \
    --seed 42
```

---

## 4. Local Smoke-Test Workflows (CPU-Friendly)

Verify complete end-to-end training and evaluation without consuming heavy GPU compute:

### Model Training Smoke-Test:
```bash
python scripts/train.py \
    --config configs/experiments/wlasl100_proposed.yaml \
    --smoke-test \
    --device cpu
```

### Model Checkpoint Evaluation:
```bash
python scripts/evaluate.py \
    --config configs/experiments/wlasl100_proposed.yaml \
    --checkpoint models/robust_holistic_fusion_best.pt \
    --manifest data/manifests/test_manifest.json \
    --landmarks-dir data/landmarks \
    --output results/reproduced_evaluation_report.json \
    --protocol official \
    --device cpu
```

### Real-Time Latency & Compression Benchmark:
```bash
python scripts/benchmark_latency.py \
    --checkpoint models/robust_holistic_fusion_best.pt \
    --config configs/experiments/wlasl100_proposed.yaml \
    --num-frames 15 \
    --device cpu \
    --output results/realtime_benchmark.json
```

---

## 5. Google Colab GPU Accelerated Training (T4 GPU)

1. Open `train_colab.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. In Step 4 form:
   - Choose `configs/experiments/reliability_fusion.yaml` or `configs/experiments/tcn.yaml`.
   - Select protocol: `official` or `signer_disjoint`.
   - Set epochs (e.g., 80) and batch size (e.g., 32 or 64).
4. Run Steps 1 through 8. Checkpoints and evaluation metrics will automatically back up to Google Drive `ASL_Colab/checkpoints`.

---

## 6. Frontend Verification

Build and type-check the React/TypeScript frontend:
```bash
cd app/frontend
npm run build
```
Start the development server:
```bash
npm run dev
```

---

## 7. Backend API Verification

Start the FastAPI backend with configurable CORS and WebSocket live recognition:
```bash
uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000
```
