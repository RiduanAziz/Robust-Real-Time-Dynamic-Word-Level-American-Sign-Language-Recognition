# WLASL100 Training Reproduction Guide

This guide details how to reproduce the training and evaluation of the genuine 100-class dynamic American Sign Language (ASL) recognition model for the thesis project: **Robust Real-Time Dynamic Word-Level American Sign Language Recognition: Mitigating Spatial-Temporal Noise via Holistic Feature Extraction**.

## 1. Prerequisites
Ensure the workspace is correctly set up with the local virtual environment and MediaPipe holistic assets.
```bash
# Verify environment
python --version  # Expected: >= 3.10
pytest tests/
```

## 2. Configuration
The canonical 100-class vocabulary has been extracted and placed in `configs/experiments/wlasl100_v1.json`.
The training configuration is defined in `configs/experiments/wlasl100_proposed.yaml`.

This configuration defines a `robust_holistic_fusion` model with 100 output classes dynamically tied to the dataset vocabulary list.

## 3. Data Processing
Before training, you must process the raw video dataset into normalized `.npz` sequences using the holistic landmark extractor.
```bash
# Process raw WLASL dataset using the 100-class subset
python -m sign_language.data.processing --config configs/experiments/wlasl100_proposed.yaml --output data/landmarks/wlasl100
```
This extracts the 553 holistic points (1659 coordinates) for all videos in the target vocabulary.

## 4. Training
Once the `.npz` sequences are generated, begin training using the local script. The model will initialize with 100 classes dynamically based on the YAML file.
```bash
python scripts/train.py --config configs/experiments/wlasl100_proposed.yaml --landmarks-dir data/landmarks
```

The training process uses the existing overlap-free temporal splitting strategy defined in `src/sign_language/data/splitting.py` to guarantee no cross-contamination between training and validation signers.

## 5. Evaluation & Robustness Testing
Evaluate the trained model checkpoint against the test split, producing the confusion matrix and macro F1 scores.
```bash
python scripts/evaluate.py --model models/robust_holistic_fusion_best.pt --config configs/experiments/wlasl100_proposed.yaml --landmarks-dir data/landmarks
```

To run the spatial-temporal noise ablation studies:
```bash
python scripts/evaluate_robustness.py --model models/robust_holistic_fusion_best.pt --config configs/experiments/wlasl100_proposed.yaml --landmarks-dir data/landmarks
```

## 6. Live Inference
Launch the backend server with the trained 100-class model by explicitly referencing the new configuration:
```bash
# Windows PowerShell
$env:MODEL_PATH="models/robust_holistic_fusion_best.pt"
$env:CONFIG_PATH="configs/experiments/wlasl100_proposed.yaml"
uvicorn sign_language.api.main:app --host 0.0.0.0 --port 8000
```
Then navigate to the SignFlow web interface and verify the 100 classes are loaded in the Practice Mode vocabulary.
