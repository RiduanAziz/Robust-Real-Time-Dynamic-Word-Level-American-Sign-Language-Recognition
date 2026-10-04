from __future__ import annotations

import torch
from torch import nn


class TemporalTransformerClassifier(nn.Module):
    """A compact temporal transformer encoder for landmark sequences."""

    def __init__(
        self,
        input_dim: int = 42,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
        max_sequence_length: int = 32,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, embedding_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.positional_embedding[:, :seq_len, :]
        x = self.dropout(x)
        padding_mask = ~mask.bool() if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
