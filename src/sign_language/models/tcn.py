"""Temporal Convolutional Network (TCN) Baseline Classifier for ASL Recognition."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Residual block with dilated 1D convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.dropout1(self.act1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.act2(self.bn2(self.conv2(out))))
        return out + res


class TCNClassifier(nn.Module):
    """Dilated Temporal Convolutional Network for dynamic sign classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        num_channels = num_channels or [128, 128, 256, 256]
        self.input_proj = nn.Linear(input_dim, num_channels[0])

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_channels[0] if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input: [B, T, D]
        if mask is not None and mask.shape == x.shape:
            x = x * mask

        # Project input features: [B, T, D] -> [B, T, C]
        h = F.gelu(self.input_proj(x))
        # Transpose to [B, C, T] for 1D convolution
        h = h.transpose(1, 2)
        h = self.network(h)  # [B, C_out, T]

        # Global temporal pooling (mean across time)
        if mask is not None:
            if mask.ndim == 2:
                time_mask = mask.unsqueeze(1).float()  # [B, 1, T]
            elif mask.ndim == 3:
                time_mask = (mask.sum(dim=-1) > 0).float().unsqueeze(1)  # [B, 1, T]
            else:
                time_mask = mask.float()
            h_masked = h * time_mask
            pooled = h_masked.sum(dim=-1) / torch.clamp(time_mask.sum(dim=-1), min=1.0)
        else:
            pooled = h.mean(dim=-1)

        out = self.fc(self.dropout(pooled))
        return out
