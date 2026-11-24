"""
src/common/data.py
==================
PyTorch Dataset and collator classes for IndicSent-M.

Provides:
  - MemoSenDataset     : loads from a processed CSV + optional pre-extracted features
  - TextFeatureDataset : loads from data/features/text_{model}_{split}.pt
  - ImageFeatureDataset: loads from data/features/image_{model}_{split}.pt (Friend 2 writes)
  - MultimodalFeatureDataset: combines text + image features for M1/M3/I1 experiments

Owner: Friend 1 (Text & Paper Lead)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Label constants (matches plan.md §3 data contract)
# ---------------------------------------------------------------------------
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Base CSV dataset
# ---------------------------------------------------------------------------
class MemoSenDataset(Dataset):
    """
    Loads a processed CSV (data/processed/{split}.csv).
    Returns raw text + label_id; suitable for use with a live tokenizer.

    Args
    ----
    csv_path : path to the split CSV
    tokenizer: a HuggingFace tokenizer (optional; if None, only raw text is returned)
    max_length: max tokenisation length (default 128, see plan.md D-07)
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer=None,
        max_length: int = 128,
    ):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate required columns
        for col in ("id", "image_path", "text", "label", "label_id"):
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {
            "id":         int(row["id"]),
            "text":       str(row["text"]),
            "label_id":   int(row["label_id"]),
            "image_path": str(row["image_path"]),
        }
        if self.tokenizer is not None:
            enc = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"]      = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        return item


# ---------------------------------------------------------------------------
# Pre-extracted feature datasets
# ---------------------------------------------------------------------------
class TextFeatureDataset(Dataset):
    """
    Loads a pre-extracted text feature file:
        data/features/text_{model}_{split}.pt

    The .pt file is a dict with keys:
        cls    : [N, 768]       float32  — CLS token embedding
        mean   : [N, 768]       float32  — masked mean-pool
        tokens : [N, L, 768]   float16  — all token embeddings
        mask   : [N, L]        bool     — attention mask
        ids    : [N]           int64    — row IDs (align with CSV)

    Args
    ----
    pt_path   : path to the .pt feature file
    use_key   : which embedding to expose as "features" ("cls" or "mean")
    with_tokens: whether to also return the full token tensor (for cross-attention)
    """

    def __init__(
        self,
        pt_path: str | Path,
        csv_path: str | Path,
        use_key: str = "cls",
        with_tokens: bool = False,
    ):
        if use_key not in ("cls", "mean"):
            raise ValueError(f"use_key must be 'cls' or 'mean', got {use_key!r}")

        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.features = data[use_key].float()     # [N, 768]
        self.ids      = data["ids"]               # [N]
        self.with_tokens = with_tokens
        if with_tokens:
            self.tokens = data["tokens"]          # [N, L, 768] fp16
            self.mask   = data["mask"]            # [N, L] bool

        # Load labels from CSV (aligned by id)
        df = pd.read_csv(csv_path, encoding="utf-8-sig").set_index("id")
        self.labels = torch.tensor(
            [int(df.loc[int(i), "label_id"]) for i in self.ids],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "features": self.features[idx],    # [768]
            "label_id": self.labels[idx],       # scalar
            "id":       self.ids[idx],
        }
        if self.with_tokens:
            item["tokens"] = self.tokens[idx]   # [L, 768] fp16
            item["mask"]   = self.mask[idx]     # [L] bool
        return item


class ImageFeatureDataset(Dataset):
    """
    Loads Friend 2's pre-extracted image feature file:
        data/features/image_{model}_{split}.pt

    Expected format: dict with at minimum:
        features : [N, D]   float32  — image embeddings (e.g. D=2048 for ResNet-50)
        ids      : [N]      int64    — row IDs (align with CSV)

    Args
    ----
    pt_path  : path to the .pt feature file
    csv_path : path to the split CSV (for label alignment)
    """

    def __init__(self, pt_path: str | Path, csv_path: str | Path):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.features = data["features"].float()  # [N, D]
        self.ids      = data["ids"]

        df = pd.read_csv(csv_path, encoding="utf-8-sig").set_index("id")
        self.labels = torch.tensor(
            [int(df.loc[int(i), "label_id"]) for i in self.ids],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "label_id": self.labels[idx],
            "id":       self.ids[idx],
        }


class MultimodalFeatureDataset(Dataset):
    """
    Zips text and image feature datasets for multimodal experiments (M1, M3, I1).

    Assumes both datasets are already aligned (same row order, same IDs).
    Verifies alignment by comparing `ids` tensors.

    Args
    ----
    text_ds  : TextFeatureDataset
    image_ds : ImageFeatureDataset
    """

    def __init__(self, text_ds: TextFeatureDataset, image_ds: ImageFeatureDataset):
        if not torch.equal(text_ds.ids, image_ds.ids):
            raise ValueError(
                "TextFeatureDataset and ImageFeatureDataset have mismatched row IDs. "
                "Re-extract features or check the pt files."
            )
        self.text_ds  = text_ds
        self.image_ds = image_ds

    def __len__(self) -> int:
        return len(self.text_ds)

    def __getitem__(self, idx: int) -> dict:
        t = self.text_ds[idx]
        v = self.image_ds[idx]
        return {
            "text_features":  t["features"],    # [768]
            "image_features": v["features"],    # [D]
            "label_id":       t["label_id"],
            "id":             t["id"],
            **({"tokens": t["tokens"], "mask": t["mask"]} if "tokens" in t else {}),
        }


# ---------------------------------------------------------------------------
# Class weights helper
# ---------------------------------------------------------------------------
def compute_class_weights(csv_path: str | Path, num_classes: int = NUM_CLASSES) -> Tensor:
    """
    Compute inverse-frequency class weights from a training CSV.
    w_c = N_total / (num_classes * N_c)

    Returns a [num_classes] float tensor suitable for nn.CrossEntropyLoss(weight=...).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    counts = df["label_id"].value_counts().sort_index()
    total  = len(df)
    weights = torch.tensor(
        [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights
