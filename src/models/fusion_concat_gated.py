"""
src/models/fusion_concat_gated.py
=================================
Fusion heads and capacity-matched text control for IndicSent-M:
  - TctrlHead          : text-only head with same MLP capacity as M1/M3
  - ConcatFusionHead   : M1 (project text 768->512, image 2048->512, concat, 2-layer MLP)
  - GatedMultimodalUnit: M3 (GMU: h_t=tanh(W_t t), h_v=tanh(W_v v),
                             z=sigmoid(W_z[t;v]), h=z*h_t+(1-z)*h_v, 2-layer MLP)

Owner: Friend 1 (Text & Paper Lead)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TctrlHead(nn.Module):
    """
    Capacity-matched text-only control head:
      text (768) -> proj (512) + ReLU + Dropout -> 2-layer MLP (512 -> 256 -> 3).
    """

    def __init__(
        self,
        text_dim: int = 768,
        proj_dim: int = 512,
        mlp_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        h = self.proj(text_features)
        return self.mlp(h)


class ConcatFusionHead(nn.Module):
    """
    M1: Project text (768 -> 512) and image (2048 -> 512), concatenate [1024],
    then 2-layer MLP (1024 -> 256 -> 3) with dropout.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 2048,
        proj_dim: int = 512,
        mlp_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
        )
        self.image_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        ht = self.text_proj(text_features)
        hv = self.image_proj(image_features)
        return self.mlp(torch.cat([ht, hv], dim=-1))


class GatedMultimodalUnit(nn.Module):
    """
    M3: Gated Multimodal Unit (Arevalo et al., 2017):
      h_t = tanh(W_t t)
      h_v = tanh(W_v v)
      z   = sigmoid(W_z [t; v])
      h   = z * h_t + (1 - z) * h_v
      logits = MLP(h)
    Returns (logits, z) when return_gate=True, or logits when return_gate=False.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        mlp_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.W_t = nn.Linear(text_dim, hidden_dim)
        self.W_v = nn.Linear(image_dim, hidden_dim)
        self.W_z = nn.Linear(text_dim + image_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t = self.drop(text_features)
        v = self.drop(image_features)
        h_t = torch.tanh(self.W_t(t))
        h_v = torch.tanh(self.W_v(v))
        z = torch.sigmoid(self.W_z(torch.cat([t, v], dim=-1)))  # [B, hidden_dim]
        h = z * h_t + (1.0 - z) * h_v
        logits = self.mlp(h)
        if return_gate:
            return logits, z
        return logits
