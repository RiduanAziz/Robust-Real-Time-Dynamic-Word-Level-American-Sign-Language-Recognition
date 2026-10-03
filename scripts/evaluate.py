from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score

from sign_language.data import SignLanguageDataset, build_synthetic_dataset
from sign_language.models import MLPClassifier


def main() -> None:
    samples = build_synthetic_dataset(samples_per_class=2, sequence_length=16, feature_dim=42)
    dataset = SignLanguageDataset(samples)
    model = MLPClassifier(input_dim=42, seq_len=16, num_classes=len({sample.label for sample in samples}))
    logits = model(torch.randn(2, 16, 42))
    preds = logits.argmax(dim=1).tolist()
    labels = [0, 1]
    print(f"accuracy={accuracy_score(labels, preds[:2]):.4f}")


if __name__ == "__main__":
    main()
