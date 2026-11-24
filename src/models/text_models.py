"""
src/models/text_models.py
=========================
Text encoder wrappers for IndicSent-M (T1/T2/T3).

Architecture per plan.md §4:
  - Frozen transformer encoder (XLM-R, MuRIL, IndicBERT)
  - Trained head: Linear(768 -> 3) + Dropout

Owner: Friend 1
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Canonical model hub IDs
MODEL_IDS = {
    "xlmr":      "xlm-roberta-base",
    "muril":     "google/muril-base-cased",
    "indicbert": "csebuetnlp/banglabert",
}

HIDDEN_DIM = 768   # all three models output 768-dim CLS/token embeddings
NUM_CLASSES = 3
DROPOUT_RATE = 0.1


class FrozenTextClassifier(nn.Module):
    """
    Frozen-encoder linear head for text-only sentiment classification.

    Args
    ----
    model_key : one of 'xlmr', 'muril', 'indicbert'
    pooling   : 'cls' | 'mean'  — which pooled rep to pass to the head
    num_classes: 3
    dropout   : dropout probability on the head input
    """

    def __init__(
        self,
        model_key: str,
        pooling: str = "cls",
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        if model_key not in MODEL_IDS:
            raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MODEL_IDS)}")
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")

        self.model_key = model_key
        self.pooling   = pooling
        self.encoder   = AutoModel.from_pretrained(MODEL_IDS[model_key])

        # Freeze all encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Trainable head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]                        # [B, H]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()        # [B, L, 1]
            summed = (last_hidden * mask).sum(dim=1)           # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)           # [B, 1]
            return summed / counts                             # [B, H]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            out = self.encoder(**kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))           # [B, num_classes]


class PrecomputedTextClassifier(nn.Module):
    """
    Linear head over pre-extracted features (cls or mean vectors from .pt files).
    Used during experiment runs to avoid re-running the encoder each epoch.

    Args
    ----
    input_dim  : feature dimension (default 768)
    hidden_dim : if > 0, adds a hidden Linear+ReLU layer (for ablation MLP)
    num_classes: 3
    dropout    : dropout probability
    """

    def __init__(
        self,
        input_dim: int = HIDDEN_DIM,
        hidden_dim: int = 0,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(features))               # [B, num_classes]
