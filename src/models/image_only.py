"""
src/models/image_only.py
========================
I1: Image-only ResNet-50 linear head trained on Friend 2's pre-extracted
data/features/image_resnet50_{split}.pt ([N, 2048]).

Owner: Friend 1 (Text & Paper Lead)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ImageOnlyClassifier(nn.Module):
    """Linear(2048 -> 3) + dropout head for frozen ResNet-50 features."""

    def __init__(self, input_dim: int = 2048, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(image_features))
