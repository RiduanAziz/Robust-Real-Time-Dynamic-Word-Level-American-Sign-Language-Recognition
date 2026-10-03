# Robust Real-Time Dynamic Word-Level American Sign Language Recognition

This repository provides a modular research scaffold for a real-time dynamic sign-language recognition system. The implementation follows a staged strategy: foundation, dataset pipeline, landmark processing, model benchmarking, and deployment.

## Project goals

- Build a signer-aware multimodal landmark pipeline for dynamic word-level recognition.
- Compare baseline, recurrent, and transformer temporal models.
- Support robust research evaluation under cross-signer and environmental variation.
- Keep the project locally runnable with CPU fallback and GPU acceleration when available.

## Directory overview

- `configs/`: YAML experiment and model configuration files.
- `docs/`: architecture and implementation plans.
- `scripts/`: training, evaluation, and report utilities.
- `src/sign_language/`: application package for data, models, training, and API code.
- `tests/`: validation suite for data and model behavior.

## Quick start

```bash
git clone <repository>
cd sign-language-recognition
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

```bash
python scripts/train.py --config configs/transformer.yaml
```

## Evaluation

```bash
python scripts/evaluate.py
```

## Real-time inference

```bash
python scripts/realtime.py
```

## API

```bash
uvicorn src.sign_language.api.main:app --reload
```

## Docker

```bash
docker compose up --build
```

## Research notes

The repository intentionally starts with a clean, testable baseline and a signer-aware dataset abstraction. This avoids common leakage and reproducibility problems while allowing future additions like MediaPipe landmark extraction, multimodal fusion, and robustness testing.
