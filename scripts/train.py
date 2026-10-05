from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config
<<<<<<< HEAD
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
=======
from sign_language.data import SignLanguageDataset, build_synthetic_dataset
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
from sign_language.models import build_model
from sign_language.training import train_model
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed
from sign_language.landmarks.pipeline import LandmarkPipeline
import torch


<<<<<<< HEAD
def collate_fn(batch):
    # Padding sequences to max length in the batch
    max_len = max(item["landmarks"].shape[0] for item in batch)
    
    # But wait, we should pad to max_sequence_length from config if provided,
    # or just pad batch locally.
    padded_landmarks = []
    masks = []
    labels = []
    lengths = []
    
    for item in batch:
        seq = item["landmarks"]
        length = seq.shape[0]
        feature_dim = seq.shape[1]
        
        pad_len = max_len - length
        if pad_len > 0:
            padded_seq = np.pad(seq, ((0, pad_len), (0, 0)), mode="constant")
            mask = np.pad(np.ones(length, dtype=np.float32), (0, pad_len), mode="constant")
        else:
            padded_seq = seq
            mask = np.ones(length, dtype=np.float32)
            
        padded_landmarks.append(padded_seq)
        masks.append(mask)
        labels.append(item["label"])
        lengths.append(length)
        
    return {
        "landmarks": torch.tensor(np.stack(padded_landmarks), dtype=torch.float32),
        "mask": torch.tensor(np.stack(masks), dtype=torch.float32),
        "label": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }

=======
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
<<<<<<< HEAD
    parser.add_argument("--manifest", type=str, default="data/manifests/train_manifest.json")
    parser.add_argument("--landmarks-dir", type=str, default="data/landmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training.")
=======
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training.")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a synthetic smoke test; never use this for research results.",
    )
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return
<<<<<<< HEAD
        
    set_seed(config.seed)

    samples = build_dataset_from_manifest(
        manifest_path=args.manifest,
        landmarks_dir=args.landmarks_dir,
        allowed_classes=config.dataset.labels
    )
    dataset = SignLanguageDataset(samples, label_to_index={l: i for i, l in enumerate(config.dataset.labels)})
    
    # We must normalize the sequence data using the pipeline
    # Wait, the dataset yields raw sequences. Let's create a collate that normalizes.
    
    pipeline = LandmarkPipeline(feature_dim=config.model.input_dim, sequence_length=config.dataset.sequence_length)
    
    def collate_with_norm(batch):
        # We can just apply the pipeline normalize on the fly
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
=======
    if not args.synthetic_smoke:
        raise RuntimeError(
            "Real training is blocked until a validated landmark dataset is available. "
            "Use --synthetic-smoke only for interface checks."
        )
    set_seed(config.seed)

    samples = build_synthetic_dataset(
        labels=config.dataset.labels or None,
        signers=config.dataset.signers or None,
        samples_per_class=config.dataset.samples_per_class,
        sequence_length=config.dataset.sequence_length,
        feature_dim=config.model.input_dim,
    )
    dataset = SignLanguageDataset(samples)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
>>>>>>> a17401fdbdf9b5cbe0015cc0edd2694dc2ff5332

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
    
    # Save the model
    torch.save(model.state_dict(), f"models/{config.model.name}.pt")

if __name__ == '__main__':
    main()
