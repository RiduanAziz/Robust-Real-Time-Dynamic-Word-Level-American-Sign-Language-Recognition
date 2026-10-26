"""Uni-Sign Inspired Prior-Guided Fusion (PGF) Classifier.

Based on:
  'Uni-Sign: Toward Unified Sign Language Understanding at Scale' (ICLR 2025)
  by Zecheng Li et al. (https://github.com/zechengli19/uni-sign, arXiv:2501.15187)

Key Innovations Adapted for Dynamic ASL Recognition:
1. Multi-Stream Anatomical Encoding:
   - Left Hand (21 keypoints -> 63 coords)
   - Right Hand (21 keypoints -> 63 coords)
   - Body Pose (33 keypoints -> 99 coords)
   - Face Mesh (478 keypoints -> 1434 coords, when available)
2. Multi-Scale Temporal Convolution & Recurrent Kinematic Modeling:
   - Multi-scale 1D temporal convolutions (kernels 3, 5) with residual connections
   - Bidirectional recurrent kinematic encoding capturing rapid finger articulation
3. Prior-Guided Cross-Modal Attention (PGF):
   - Skeletal Pose acts as structural/spatial prior for Hand gestures
   - Hands attend to Pose/Face reference anchors (e.g., chest, face, shoulders)
4. Score-Aware Reliability Gating:
   - Dynamic landmark confidence/visibility masking modulates stream representations
   - Seamlessly handles one-handed signs, occlusions, and missing keypoints
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal convolution block capturing short- and long-term kinematic transitions."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        res = self.res(x)
        x_t = x.transpose(1, 2)  # [B, C, T]
        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        out = torch.cat([c3, c5], dim=1).transpose(1, 2)  # [B, T, out_channels]
        out = self.norm(out + res)
        return self.drop(self.act(out))


class AnatomicalStreamEncoder(nn.Module):
    """Encodes an anatomical landmark stream with spatial projection, temporal multi-scale conv, and BiGRU."""

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
        self.temporal_conv = MultiScaleTemporalEncoder(emb_dim, emb_dim, dropout=dropout)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, in_dim]
        if mask is not None:
            x = x * mask
        h = self.proj(x)
        h = self.temporal_conv(h)
        seq_out, _ = self.rnn(h)  # [B, T, out_dim]

        # Temporal pooling weighted by valid frames
        if mask is not None:
            time_mask = (mask.sum(dim=-1, keepdim=True) > 0).float()
            denom = torch.clamp(time_mask.sum(dim=1), min=1.0)
            pooled = (seq_out * time_mask).sum(dim=1) / denom
        else:
            pooled = seq_out.mean(dim=1)

        return seq_out, pooled


class PriorGuidedCrossAttention(nn.Module):
    """Prior-Guided Cross-Attention: Hand features attend to Pose/Face structural anchors."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # query: [B, T, D] (e.g. hands)
        # key_value: [B, T, D] (e.g. pose anchors)
        attn_out, _ = self.mha(query, key_value, key_value)
        return self.norm(query + self.drop(attn_out))


class UniSignPGFClassifier(nn.Module):
    """Uni-Sign Inspired Prior-Guided Fusion Classifier for Dynamic ASL Recognition.

    Integrates:
    - Multi-stream anatomical feature isolation (LH, RH, Pose, Face)
    - Prior-guided cross-attention (Pose prior anchors Hand gestures)
    - Score-aware reliability gating (handles single-handed signs & missing keypoints)
    - Multi-scale temporal convolution + BiGRU kinematics
    - Calibrated classification head
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        emb_dim: int = 96,
        hidden_dim: int = 96,
        num_layers: int = 2,
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

        # Anatomical encoders
        self.lh_encoder = AnatomicalStreamEncoder(self.lh_dim, emb_dim, hidden_dim, num_layers, dropout)
        self.rh_encoder = AnatomicalStreamEncoder(self.rh_dim, emb_dim, hidden_dim, num_layers, dropout)

        stream_out_dim = hidden_dim * 2

        if self.pose_dim > 0:
            self.pose_encoder = AnatomicalStreamEncoder(self.pose_dim, emb_dim, hidden_dim, num_layers, dropout)
            # Prior-guided cross attention: hands attend to body pose structural anchor
            self.lh_prior_attn = PriorGuidedCrossAttention(stream_out_dim, nhead=4, dropout=dropout)
            self.rh_prior_attn = PriorGuidedCrossAttention(stream_out_dim, nhead=4, dropout=dropout)
        else:
            self.pose_encoder = None
            self.lh_prior_attn = None
            self.rh_prior_attn = None

        if self.face_dim > 0:
            face_hidden = max(32, hidden_dim // 2)
            face_emb = max(32, emb_dim // 2)
            self.face_encoder = AnatomicalStreamEncoder(self.face_dim, face_emb, face_hidden, num_layers=1, dropout=dropout)
            self.face_out_dim = face_hidden * 2
        else:
            self.face_encoder = None
            self.face_out_dim = 0

        # Score-aware reliability gating
        self.lh_gate = nn.Linear(stream_out_dim + 1, stream_out_dim)
        self.rh_gate = nn.Linear(stream_out_dim + 1, stream_out_dim)
        if self.pose_encoder is not None:
            self.pose_gate = nn.Linear(stream_out_dim + 1, stream_out_dim)
        if self.face_encoder is not None:
            self.face_gate = nn.Linear(self.face_out_dim + 1, self.face_out_dim)

        # Inter-hand interaction module
        self.inter_hand_proj = nn.Sequential(
            nn.Linear(stream_out_dim * 2, stream_out_dim),
            nn.LayerNorm(stream_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        total_fused = (
            stream_out_dim * 2  # Gated LH + RH
            + stream_out_dim     # Inter-hand interaction
            + (stream_out_dim if self.pose_encoder is not None else 0)
            + self.face_out_dim
        )

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if mask is not None:
            if mask.ndim == 2:
                m = mask.unsqueeze(-1).expand(-1, -1, d)
            else:
                m = mask
        else:
            m = torch.ones_like(x)

        # 1. Slices
        lh_x = x[:, :, :63]
        lh_m = m[:, :, :63]
        rh_x = x[:, :, 63:126]
        rh_m = m[:, :, 63:126]

        # 2. Score-Aware Reliability Metrics [B, 1]
        lh_rel = (lh_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * 63))).squeeze(-1)
        rh_rel = (rh_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * 63))).squeeze(-1)

        # 3. Stream Encoding
        lh_seq, lh_pooled = self.lh_encoder(lh_x, lh_m)
        rh_seq, rh_pooled = self.rh_encoder(rh_x, rh_m)

        # 4. Prior-Guided Cross-Modal Attention with Pose
        if self.pose_encoder is not None:
            pose_x = x[:, :, 126 : 126 + self.pose_dim]
            pose_m = m[:, :, 126 : 126 + self.pose_dim]
            pose_rel = (pose_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * self.pose_dim))).squeeze(-1)
            pose_seq, pose_pooled = self.pose_encoder(pose_x, pose_m)

            # Hands attend to body pose priors
            lh_seq = self.lh_prior_attn(lh_seq, pose_seq)
            rh_seq = self.rh_prior_attn(rh_seq, pose_seq)

            # Re-pool with attention-enhanced features
            lh_pooled = lh_seq.mean(dim=1)
            rh_pooled = rh_seq.mean(dim=1)

            # Gating for pose
            pose_gate_in = torch.cat([pose_pooled, pose_rel], dim=-1)
            pose_gated = pose_pooled * torch.sigmoid(self.pose_gate(pose_gate_in))
        else:
            pose_gated = None

        # 5. Score-aware reliability gating for hands
        lh_gate_in = torch.cat([lh_pooled, lh_rel], dim=-1)
        lh_gated = lh_pooled * torch.sigmoid(self.lh_gate(lh_gate_in))

        rh_gate_in = torch.cat([rh_pooled, rh_rel], dim=-1)
        rh_gated = rh_pooled * torch.sigmoid(self.rh_gate(rh_gate_in))

        # Inter-hand spatial interaction
        inter_hand = self.inter_hand_proj(torch.cat([lh_gated, rh_gated], dim=-1))

        fused_parts = [lh_gated, rh_gated, inter_hand]
        if pose_gated is not None:
            fused_parts.append(pose_gated)

        # 6. Face Stream (when available)
        if self.face_encoder is not None:
            face_x = x[:, :, 126 + self.pose_dim : 126 + self.pose_dim + self.face_dim]
            face_m = m[:, :, 126 + self.pose_dim : 126 + self.pose_dim + self.face_dim]
            face_rel = (face_m.sum(dim=(1, 2), keepdim=True) / max(1.0, float(t * self.face_dim))).squeeze(-1)
            _, face_pooled = self.face_encoder(face_x, face_m)
            face_gate_in = torch.cat([face_pooled, face_rel], dim=-1)
            face_gated = face_pooled * torch.sigmoid(self.face_gate(face_gate_in))
            fused_parts.append(face_gated)

        # 7. Global Fusion & Classification
        fused = torch.cat(fused_parts, dim=-1)
        emb = self.fusion_mlp(fused)
        logits = self.classifier(emb)

        # Temperature calibration
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.1)
        return calibrated_logits
