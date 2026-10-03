from __future__ import annotations

import torch
from torch import nn


class MultimodalFusionClassifier(nn.Module):
    """Combine pose-like spatial features with motion features before classification."""

    def __init__(
        self,
        spatial_dim: int = 42,
        motion_dim: int = 42,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, spatial: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-1] != self.spatial_proj[0].in_features:
            raise ValueError(f"Expected spatial feature dim {self.spatial_proj[0].in_features}, got {spatial.shape[-1]}")
        if motion.shape[-1] != self.motion_proj[0].in_features:
            raise ValueError(f"Expected motion feature dim {self.motion_proj[0].in_features}, got {motion.shape[-1]}")

        spatial_features = self.spatial_proj(spatial)
        motion_features = self.motion_proj(motion)
        pooled = torch.cat([spatial_features.mean(dim=1), motion_features.mean(dim=1)], dim=-1)
        fused = self.fusion(pooled)
        return self.classifier(fused)
