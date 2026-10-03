from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """GRU classifier for temporal sequences of landmark features."""

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        direction = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        last_hidden = outputs[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)
