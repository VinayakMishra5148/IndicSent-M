"""
src/common/metrics.py
=====================
Metric computation for IndicSent-M.

Provides:
  compute_metrics(y_true, y_pred, probs) -> dict
  bootstrap_macro_f1_ci(y_true, y_pred, n=1000, alpha=0.05) -> (lo, hi)
  plot_confusion_matrix(y_true, y_pred, out_path)

Owner: Friend 1
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
NUM_CLASSES  = 3


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    probs:  np.ndarray | None = None,
) -> dict:
    """
    Returns a dict with:
      accuracy, macro_f1, weighted_f1,
      per_class: {Negative:{precision,recall,f1,support}, ...}
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc  = float(accuracy_score(y_true, y_pred))
    mac  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    wt   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    per_class = {
        LABEL_NAMES[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(s[i]),
        }
        for i in range(NUM_CLASSES)
    }

    out = {
        "accuracy":    acc,
        "macro_f1":    mac,
        "weighted_f1": wt,
        "per_class":   per_class,
    }
    return out


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95% CI for macro-F1 over test samples.
    Resamples (y_true, y_pred) pairs with replacement n times.
    Returns (lower, upper).
    """
    rng = np.random.default_rng(rng_seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n      = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(
            float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        )
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lo, hi


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Save a row-normalised confusion matrix heatmap as PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot  = cm.astype(float) / row_sums
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot, fmt, vmax = cm, "d", None

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        xlabel="Predicted", ylabel="True", title=title,
    )
    thresh = (cm_plot.max() + cm_plot.min()) / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_plot[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
