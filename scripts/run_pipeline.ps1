# 1. Dataset Preparation
python scripts/prepare_dataset.py --raw-dir data/raw --metadata data/WLASL_v0.3.json --output data/manifests/full_manifest.json --allow-missing

# 2. Split Dataset
python scripts/split_dataset.py --manifest data/manifests/full_manifest.json --output-dir data/manifests

# 3. Extract Landmarks
python scripts/extract_landmarks.py --manifest data/manifests/full_manifest.json --raw-dir data/raw --output-dir data/landmarks --model-asset-path models/mediapipe/holistic_landmarker.task

# 4. Train Model
python scripts/train.py --config configs/base.yaml --manifest data/manifests/train_manifest.json --landmarks-dir data/landmarks

# 5. Evaluate Robustness
python scripts/evaluate_robustness.py --config configs/base.yaml --manifest data/manifests/test_manifest.json --landmarks-dir data/landmarks --model-path models/temporal_transformer.pt
