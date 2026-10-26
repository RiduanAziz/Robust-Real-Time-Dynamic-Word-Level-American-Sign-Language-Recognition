"""Reliability-Aware Feature-Isolated Spatiotemporal Classifier (Proposed Architecture).

Anatomical feature isolation + reliability-aware modality gating + temporal modelling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamEncoder(nn.Module):
    """Isolated spatial projection and recurrent/temporal encoder for one anatomical stream."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, in_dim]
        if mask is not None:
            x = x * mask
        h = self.proj(x)
        out, _ = self.rnn(h)  # [B, T, out_dim]

        # Temporal pooling weighted by valid frames
        if mask is not None:
            time_mask = (mask.sum(dim=-1, keepdim=True) > 0).float()  # [B, T, 1]
            out = out * time_mask
            denom = torch.clamp(time_mask.sum(dim=1), min=1.0)
            pooled = out.sum(dim=1) / denom  # [B, out_dim]
        else:
            pooled = out.mean(dim=1)

        return pooled


class ReliabilityAwareIsolatedFusionClassifier(nn.Module):
    """Proposed Reliability-Aware Feature-Isolated Spatiotemporal Classifier.

    Isolates Left Hand, Right Hand, and Body Pose into separate kinematic encoders,
    estimates per-modality reliability metrics, applies adaptive gating, and classifies
    with temperature calibration.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

        # Slice dimensions: Left Hand (63), Right Hand (63), Pose (99), Face (remainder)
        self.lh_dim = 63
        self.rh_dim = 63
        self.pose_dim = 99 if input_dim >= 225 else max(0, input_dim - 126)
        self.face_dim = max(0, input_dim - (self.lh_dim + self.rh_dim + self.pose_dim))

        self.lh_encoder = StreamEncoder(self.lh_dim, emb_dim, hidden_dim, num_layers, dropout)
        self.rh_encoder = StreamEncoder(self.rh_dim, emb_dim, hidden_dim, num_layers, dropout)

        if self.pose_dim > 0:
            self.pose_encoder = StreamEncoder(self.pose_dim, emb_dim, hidden_dim, num_layers, dropout)
        else:
            self.pose_encoder = None

        if self.face_dim > 0:
            self.face_encoder = StreamEncoder(self.face_dim, emb_dim // 2, hidden_dim // 2, num_layers, dropout)
            self.face_out_dim = hidden_dim
        else:
            self.face_encoder = None
            self.face_out_dim = 0

        # Reliability gating gates
        enc_out = hidden_dim * 2
        self.lh_gate = nn.Linear(enc_out + 1, enc_out)
        self.rh_gate = nn.Linear(enc_out + 1, enc_out)
        if self.pose_encoder is not None:
            self.pose_gate = nn.Linear(enc_out + 1, enc_out)
        if self.face_encoder is not None:
            self.face_gate = nn.Linear(self.face_out_dim + 1, self.face_out_dim)

        total_fused = enc_out * 2 + (enc_out if self.pose_encoder is not None else 0) + self.face_out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_fused, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        if mask is not None:
            if mask.ndim == 2:
                m = mask.unsqueeze(-1).expand(-1, -1, d)
            else:
                m = mask
        else:
            m = torch.ones_like(x)

        # 1. Slice streams
        lh_x = x[:, :, :63]
        lh_m = m[:, :, :63]
        rh_x = x[:, :, 63:126]
        rh_m = m[:, :, 63:126]

        # 2. Compute reliability metrics per stream [B, 1]
        # Ratio of valid frames detected
        lh_rel = (lh_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * 63))).squeeze(-1)
        rh_rel = (rh_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * 63))).squeeze(-1)

        # 3. Stream encoding
        lh_feat = self.lh_encoder(lh_x, lh_m)
        rh_feat = self.rh_encoder(rh_x, rh_m)

        # 4. Reliability gating
        lh_gate_in = torch.cat([lh_feat, lh_rel], dim=-1)
        lh_gated = lh_feat * torch.sigmoid(self.lh_gate(lh_gate_in))

        rh_gate_in = torch.cat([rh_feat, rh_rel], dim=-1)
        rh_gated = rh_feat * torch.sigmoid(self.rh_gate(rh_gate_in))

        fused_parts = [lh_gated, rh_gated]

        if self.pose_encoder is not None:
            pose_x = x[:, :, 126 : 126 + self.pose_dim]
            pose_m = m[:, :, 126 : 126 + self.pose_dim]
            pose_rel = (pose_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * self.pose_dim))).squeeze(-1)
            pose_feat = self.pose_encoder(pose_x, pose_m)
            pose_gate_in = torch.cat([pose_feat, pose_rel], dim=-1)
            pose_gated = pose_feat * torch.sigmoid(self.pose_gate(pose_gate_in))
            fused_parts.append(pose_gated)

        if self.face_encoder is not None:
            face_x = x[:, :, 126 + self.pose_dim : 126 + self.pose_dim + self.face_dim]
            face_m = m[:, :, 126 + self.pose_dim : 126 + self.pose_dim + self.face_dim]
            face_rel = (face_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * self.face_dim))).squeeze(-1)
            face_feat = self.face_encoder(face_x, face_m)
            face_gate_in = torch.cat([face_feat, face_rel], dim=-1)
            face_gated = face_feat * torch.sigmoid(self.face_gate(face_gate_in))
            fused_parts.append(face_gated)

        # 5. Fusion & classification
        fused = torch.cat(fused_parts, dim=-1)
        emb = self.fusion_mlp(fused)
        logits = self.classifier(emb)

        # Temperature calibration
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.1)
        return calibrated_logits
