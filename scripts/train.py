from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.models import build_model
from sign_language.training import train_model
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed
from sign_language.landmarks.pipeline import LandmarkPipeline
import torch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--manifest", type=str, default="data/manifests/train_manifest.json")
    parser.add_argument("--landmarks-dir", type=str, default="data/landmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training.")
    args = parser.parse_args()
    
    config = load_experiment_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return
        
    set_seed(config.seed)

    samples = build_dataset_from_manifest(
        manifest_path=args.manifest,
        landmarks_dir=args.landmarks_dir,
        allowed_classes=config.dataset.labels
    )
    dataset = SignLanguageDataset(samples, label_to_index={l: i for i, l in enumerate(config.dataset.labels)})
    
    raw_feature_dim = config.model.input_dim // 3
    pipeline = LandmarkPipeline(feature_dim=raw_feature_dim, sequence_length=config.dataset.sequence_length)
    
    def collate_with_norm(batch):
        normalized_landmarks = []
        labels = []
        for item in batch:
            seq = item["landmarks"]
            norm_seq = pipeline.normalize_sequence(seq)
            normalized_landmarks.append(norm_seq)
            labels.append(item["label"])
            
        return {
            "landmarks": torch.tensor(np.stack(normalized_landmarks), dtype=torch.float32),
            "label": torch.tensor(labels, dtype=torch.long)
        }
    
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_with_norm)

    from sign_language.training.trainer import evaluate_model
    
    val_samples = build_dataset_from_manifest(
        manifest_path=args.manifest.replace("train_", "validation_"),
        landmarks_dir=args.landmarks_dir,
        allowed_classes=config.dataset.labels
    )
    val_dataset = SignLanguageDataset(val_samples, label_to_index={l: i for i, l in enumerate(config.dataset.labels)})
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_with_norm)
    
    model = build_model(config)
    device = get_device(config.device)
    loss_history = train_model(
        model,
        loader,
        device,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        gradient_clip=config.training.gradient_clip,
    )
    
    print(f"Training completed; final loss: {loss_history[-1]:.4f}")
    
    # Evaluate on validation
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["landmarks"].to(device)
            targets = batch["label"].to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            
    val_metrics = evaluate_model(torch.cat(all_logits), torch.cat(all_targets))
    print("Validation Metrics:")
    print(json.dumps(val_metrics, indent=2))
    
    # Save the metrics
    metrics_out = {
        "loss_history": loss_history,
        "validation_metrics": val_metrics
    }
    
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{config.results_dir}/training_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
        
    print(f"Metrics saved to {config.results_dir}/training_metrics.json")
    
    # Save the model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"models/{config.model.name}.pt")

if __name__ == '__main__':
    main()
