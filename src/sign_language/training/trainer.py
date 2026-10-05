from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _ensure_tensor(value: torch.Tensor | list[int] | list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    return tensor.detach()


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float | int]:
    """Compute accuracy, macro-F1, and cross-entropy loss for classification logits."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.long)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if targets.ndim == 0:
        targets = targets.unsqueeze(0)

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same number of rows.")

    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    loss = F.cross_entropy(logits, targets).item()

    num_classes = logits.shape[-1]
    true_positive = torch.zeros(num_classes, dtype=torch.float32)
    false_positive = torch.zeros(num_classes, dtype=torch.float32)
    false_negative = torch.zeros(num_classes, dtype=torch.float32)

    for class_index in range(num_classes):
        class_targets = targets == class_index
        class_preds = predictions == class_index
        true_positive[class_index] = torch.logical_and(class_targets, class_preds).sum().float()
        false_positive[class_index] = torch.logical_and(~class_targets, class_preds).sum().float()
        false_negative[class_index] = torch.logical_and(class_targets, ~class_preds).sum().float()

    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean().item()

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "loss": float(loss),
        "num_samples": int(targets.shape[0]),
    }


def evaluate_model(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    """Lightweight wrapper for evaluating model outputs against ground-truth labels."""
    metrics = compute_classification_metrics(logits, targets)
    metrics["accuracy"] = float(metrics["accuracy"])
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip: float | None = None,
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
            model_kwargs = {}
            if "lengths" in batch:
                model_kwargs["lengths"] = batch["lengths"].to(device)
            if "mask" in batch:
                model_kwargs["mask"] = batch["mask"].to(device)
            logits = model(inputs, **model_kwargs)
            loss = criterion(logits, targets)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / max(1, len(train_loader)))
    return history
