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

## Repository structure

- [configs](configs): experiment configuration files
- [docs](docs): architecture and planning notes
- [scripts](scripts): reproducible operational entry points
- [src](src): reusable project source code
- [tests](tests): validation and regression checks
- [results](results): generated metrics and figures
- [data](data): dataset manifests and metadata storage

## Installation

```bash
git clone <repository-url>
cd Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment files

- A pip-based installation is defined in [requirements.txt](requirements.txt)
- A Conda environment definition is available in [environment.yml](environment.yml)

## Validation status

The project is under continuous test validation and currently passes the repository test suite.

## Notes

This repository intentionally keeps the research and deployment layers modular. Model training and full webcam integration remain scoped as future research extensions, while the current codebase provides the validated research pipeline and runtime inference scaffolding needed for the next experimental iteration.
