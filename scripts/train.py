from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_synthetic_dataset
from sign_language.models import build_model
from sign_language.training import train_model
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config without training.")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a synthetic smoke test; never use this for research results.",
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return
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


if __name__ == "__main__":
    main()
