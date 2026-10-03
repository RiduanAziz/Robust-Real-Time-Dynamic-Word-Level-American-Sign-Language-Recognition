from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from sign_language.data import SignLanguageDataset, build_synthetic_dataset
from sign_language.models import MLPClassifier, GRUClassifier, LSTMClassifier, TemporalTransformerClassifier
from sign_language.training import train_model
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed


def load_config(path: str | None) -> dict:
    if path is None:
        config_path = Path("configs/base.yaml")
    else:
        config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_model(config: dict) -> torch.nn.Module:
    model_name = config.get("model", {}).get("name", "mlp")
    params = config.get("model", {})
    if model_name == "mlp":
        return MLPClassifier(input_dim=params.get("input_dim", 42), seq_len=params.get("sequence_length", 16), num_classes=params.get("num_classes", 5))
    if model_name == "lstm":
        return LSTMClassifier(
            input_dim=params.get("input_dim", 42),
            hidden_dim=params.get("hidden_dim", 64),
            num_layers=params.get("num_layers", 2),
            num_classes=params.get("num_classes", 5),
            bidirectional=params.get("bidirectional", False),
            dropout=params.get("dropout", 0.1),
        )
    if model_name == "gru":
        return GRUClassifier(
            input_dim=params.get("input_dim", 42),
            hidden_dim=params.get("hidden_dim", 64),
            num_layers=params.get("num_layers", 2),
            num_classes=params.get("num_classes", 5),
            bidirectional=params.get("bidirectional", False),
            dropout=params.get("dropout", 0.1),
        )
    return TemporalTransformerClassifier(
        input_dim=params.get("input_dim", 42),
        embedding_dim=params.get("embedding_dim", 64),
        num_heads=params.get("num_heads", 4),
        num_layers=params.get("num_layers", 2),
        ff_dim=params.get("ff_dim", 128),
        num_classes=params.get("num_classes", 5),
        dropout=params.get("dropout", 0.1),
        max_sequence_length=params.get("max_sequence_length", 32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    samples = build_synthetic_dataset(samples_per_class=4, sequence_length=config.get("data", {}).get("sequence_length", 16), feature_dim=42)
    dataset = SignLanguageDataset(samples)
    loader = DataLoader(dataset, batch_size=config.get("training", {}).get("batch_size", 8), shuffle=True)

    model = build_model(config)
    device = get_device()
    loss_history = train_model(
        model,
        loader,
        device,
        epochs=config.get("training", {}).get("epochs", 2),
        learning_rate=config.get("training", {}).get("learning_rate", 1e-3),
        weight_decay=config.get("training", {}).get("weight_decay", 1e-4),
    )
    print(f"Training completed; final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
