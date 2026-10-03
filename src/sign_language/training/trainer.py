from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> list[float]:
    """Train a classification model with a simple AdamW loop."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["landmarks"].to(device)
            targets = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / max(1, len(train_loader)))
    return history
