from __future__ import annotations

import argparse

import torch
from sklearn.metrics import accuracy_score

from sign_language.config import load_experiment_config
from sign_language.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation smoke path.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    model = build_model(config)
    logits = model(torch.randn(2, config.model.sequence_length, config.model.input_dim))
    preds = logits.argmax(dim=1).tolist()
    labels = [0, 1]
    print(f"accuracy={accuracy_score(labels, preds[:2]):.4f}")


if __name__ == "__main__":
    main()
