import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.models import build_model
from sign_language.robustness import apply_combined_noise, NoiseScenario, evaluate_robustness, recognition_robustness_report
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.training.trainer import collate_fn, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", default="data/manifests/test_manifest.json")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="results/robustness/report.json")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    pipeline = LandmarkPipeline(feature_dim=config.model.input_dim, sequence_length=config.dataset.sequence_length)
    
    samples = build_dataset_from_manifest(args.manifest, args.landmarks_dir, config.dataset.labels)
    dataset = SignLanguageDataset(samples, label_to_index={l: i for i, l in enumerate(config.dataset.labels)})
    
    model = build_model(config)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    
    # We will test multiple noise types and severities
    spatial_noises = ["coordinate_jitter", "translation", "scale", "landmark_dropout"]
    temporal_noises = ["frame_drop", "frame_duplicate", "sequence_truncate"]
    severities = [0.1, 0.3, 0.5]
    
    results = []
    
    # Generate clean predictions first
    clean_targets = []
    clean_preds = []
    
    for item in dataset:
        seq = item["landmarks"]
        norm_seq = pipeline.normalize_sequence(seq)
        with torch.no_grad():
            x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            pred = logits.argmax(-1).item()
        clean_targets.append(item["label"])
        clean_preds.append(pred)
        
    # Test noisy variants
    for noise_type in spatial_noises:
        for sev in severities:
            noisy_preds = []
            for item in dataset:
                seq = item["landmarks"]
                noisy_seq = apply_combined_noise(seq, noise_type, "frame_drop", sev, 0.0)
                norm_seq = pipeline.normalize_sequence(noisy_seq)
                with torch.no_grad():
                    x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
                    logits = model(x)
                    pred = logits.argmax(-1).item()
                noisy_preds.append(pred)
                
            report = recognition_robustness_report(
                clean_targets, clean_preds, noisy_preds, config.dataset.labels, noise_type, sev
            )
            results.append(report)
            
    for noise_type in temporal_noises:
        for sev in severities:
            noisy_preds = []
            for item in dataset:
                seq = item["landmarks"]
                noisy_seq = apply_combined_noise(seq, "coordinate_jitter", noise_type, 0.0, sev)
                norm_seq = pipeline.normalize_sequence(noisy_seq)
                with torch.no_grad():
                    x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
                    logits = model(x)
                    pred = logits.argmax(-1).item()
                noisy_preds.append(pred)
                
            report = recognition_robustness_report(
                clean_targets, clean_preds, noisy_preds, config.dataset.labels, noise_type, sev
            )
            results.append(report)
            
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
