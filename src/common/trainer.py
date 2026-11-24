"""
src/common/trainer.py
=====================
Generic seeded training loop for IndicSent-M.

Implements per plan.md §4:
  - AdamW, lr=3e-4 (head only), batch=32, epochs=20, patience=5
  - Class weights from TRAIN labels only (w_c = N / (3 * N_c))
  - Best checkpoint chosen by VALIDATION macro-F1
  - Test evaluated ONCE per seed at the best checkpoint
  - Per-seed predictions saved as results/runs/<ID>/seed<k>.csv + metrics JSON
  - Never uses test metrics for any choice

Owner: Friend 1
"""
from __future__ import annotations
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.metrics import (
    compute_metrics,
    bootstrap_macro_f1_ci,
    plot_confusion_matrix,
    save_metrics_json,
)

log = logging.getLogger("trainer")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def class_weights_from_csv(csv_path: str | Path, num_classes: int = 3) -> torch.Tensor:
    """w_c = N_total / (num_classes * N_c), from TRAIN csv only."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    counts = df["label_id"].value_counts().sort_index()
    total  = len(df)
    w = torch.tensor(
        [total / (num_classes * int(counts.get(c, 1))) for c in range(num_classes)],
        dtype=torch.float32,
    )
    log.info(f"  Class weights (from {Path(csv_path).name}): {w.tolist()}")
    return w


# ---------------------------------------------------------------------------
# Feature-based dataset (no live tokeniser)
# ---------------------------------------------------------------------------
def make_feature_dataset(pt_path: str | Path, csv_path: str | Path, key: str = "cls"):
    """
    Load pre-extracted .pt features and labels; return a TensorDataset.
    key: 'cls' or 'mean'
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    features = data[key].float()           # [N, 768]
    ids_      = data["ids"]                # [N] int64

    df = pd.read_csv(csv_path, encoding="utf-8-sig").set_index("id")
    labels = torch.tensor(
        [int(df.loc[int(i), "label_id"]) for i in ids_],
        dtype=torch.long,
    )
    return TensorDataset(features, labels, ids_)


# ---------------------------------------------------------------------------
# Single-seed training run
# ---------------------------------------------------------------------------
def run_one_seed(
    model: nn.Module,
    train_pt: str | Path,
    valid_pt: str | Path,
    test_pt:  str | Path,
    train_csv: str | Path,
    valid_csv: str | Path,
    test_csv:  str | Path,
    feature_key: str,          # 'cls' or 'mean'
    run_dir: str | Path,
    seed: int,
    lr: float          = 3e-4,
    batch_size: int    = 32,
    max_epochs: int    = 20,
    patience: int      = 5,
    device: str | None = None,
) -> dict:
    """
    Train one seed. Returns dict of test metrics.
    Saves:
      run_dir/seed{seed}.csv        (id, y_true, y_pred, prob_neg, prob_neu, prob_pos)
      run_dir/seed{seed}_metrics.json
      run_dir/seed{seed}_cm.png     (row-normalised confusion matrix)
    """
    seed_everything(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  [seed {seed}] device={device}")

    # Datasets
    train_ds = make_feature_dataset(train_pt, train_csv, key=feature_key)
    valid_ds = make_feature_dataset(valid_pt, valid_csv, key=feature_key)
    test_ds  = make_feature_dataset(test_pt,  test_csv,  key=feature_key)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device=="cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # Class weights from train CSV
    cw = class_weights_from_csv(train_csv).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Only head params are trainable (encoder is already frozen at model level)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    model = model.to(dev)

    best_val_mf1  = -1.0
    best_state    = None
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        for feats, lbls, _ in train_loader:
            feats, lbls = feats.to(dev), lbls.to(dev)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for feats, lbls, _ in valid_loader:
                logits = model(feats.to(dev))
                val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                val_true.extend(lbls.tolist())

        val_m = compute_metrics(val_true, val_preds)
        val_mf1 = val_m["macro_f1"]
        log.info(f"    epoch {epoch:02d}  val_macro_f1={val_mf1:.4f}")

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info(f"    Early stop at epoch {epoch} (patience={patience})")
                break

    # --- Test (once, at best checkpoint) ---
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true, test_probs, test_ids = [], [], [], []
    with torch.no_grad():
        for feats, lbls, ids_ in test_loader:
            logits = model(feats.to(dev))
            probs  = torch.softmax(logits, dim=-1).cpu()
            test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            test_true.extend(lbls.tolist())
            test_probs.extend(probs.tolist())
            test_ids.extend(ids_.tolist())

    test_m = compute_metrics(test_true, test_preds)
    ci_lo, ci_hi = bootstrap_macro_f1_ci(
        np.array(test_true), np.array(test_preds), n_resamples=1000, rng_seed=seed
    )
    test_m["macro_f1_ci95"] = [ci_lo, ci_hi]
    test_m["best_val_macro_f1"] = best_val_mf1
    test_m["seed"] = seed

    # Save per-seed CSV
    pred_df = pd.DataFrame({
        "id":       test_ids,
        "y_true":   test_true,
        "y_pred":   test_preds,
        "prob_neg": [p[0] for p in test_probs],
        "prob_neu": [p[1] for p in test_probs],
        "prob_pos": [p[2] for p in test_probs],
    })
    pred_df.to_csv(run_dir / f"seed{seed}.csv", index=False)
    save_metrics_json(test_m, run_dir / f"seed{seed}_metrics.json")
    plot_confusion_matrix(
        np.array(test_true), np.array(test_preds),
        run_dir / f"seed{seed}_cm.png",
        title=f"Seed {seed} — {run_dir.name}",
    )
    log.info(
        f"  [seed {seed}] test macro_f1={test_m['macro_f1']:.4f} "
        f"CI=[{ci_lo:.4f},{ci_hi:.4f}]"
    )
    return test_m


# ---------------------------------------------------------------------------
# Multi-seed orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    experiment_id: str,
    make_model_fn,            # callable() -> nn.Module (called fresh each seed)
    train_pt: str | Path,
    valid_pt: str | Path,
    test_pt:  str | Path,
    train_csv: str | Path,
    valid_csv: str | Path,
    test_csv:  str | Path,
    feature_key: str,
    results_root: str | Path,
    seeds: list[int] = None,
    **kwargs,
) -> dict:
    """
    Run 5 seeds for one experiment. Aggregate mean±std over seeds.
    Returns summary dict; saves results/runs/<experiment_id>/summary.json.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    run_dir = Path(results_root) / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for seed in seeds:
        model = make_model_fn()
        m = run_one_seed(
            model=model,
            train_pt=train_pt, valid_pt=valid_pt, test_pt=test_pt,
            train_csv=train_csv, valid_csv=valid_csv, test_csv=test_csv,
            feature_key=feature_key,
            run_dir=run_dir,
            seed=seed,
            **kwargs,
        )
        all_metrics.append(m)

    # Aggregate
    keys_scalar = ["accuracy", "macro_f1", "weighted_f1"]
    summary = {
        "experiment_id": experiment_id,
        "seeds": seeds,
        "n_seeds": len(seeds),
    }
    for k in keys_scalar:
        vals = [m[k] for m in all_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))

    # Per-class mean±std
    summary["per_class"] = {}
    for cls_name in ["Negative", "Neutral", "Positive"]:
        summary["per_class"][cls_name] = {}
        for metric in ["precision", "recall", "f1"]:
            vals = [m["per_class"][cls_name][metric] for m in all_metrics]
            summary["per_class"][cls_name][f"{metric}_mean"] = float(np.mean(vals))
            summary["per_class"][cls_name][f"{metric}_std"]  = float(np.std(vals))

    # Bootstrap CI on seed-mean predictions (average probabilities across seeds)
    seed_dfs = [pd.read_csv(run_dir / f"seed{s}.csv") for s in seeds]
    y_true_ref = seed_dfs[0]["y_true"].values
    mean_probs = np.mean(
        [df[["prob_neg", "prob_neu", "prob_pos"]].values for df in seed_dfs],
        axis=0,
    )
    seed_mean_preds = mean_probs.argmax(axis=1)
    ci_lo, ci_hi = bootstrap_macro_f1_ci(
        y_true_ref, seed_mean_preds,
        n_resamples=1000, rng_seed=42
    )
    summary["macro_f1_bootstrap_ci95"] = [ci_lo, ci_hi]
    summary["seed_mean_metrics"] = compute_metrics(y_true_ref, seed_mean_preds)

    # Save row-normalised confusion matrix of seed-mean predictions to results/figures/
    if not experiment_id.endswith("_smoke"):
        fig_path = Path(results_root) / "figures" / f"{experiment_id}_cm.png"
        plot_confusion_matrix(
            y_true_ref, seed_mean_preds,
            fig_path,
            title=f"{experiment_id} (5-seed mean, test)",
            normalize=True,
        )

    save_metrics_json(summary, run_dir / "summary.json")
    log.info(
        f"[{experiment_id}] macro_f1={summary['macro_f1_mean']:.4f}"
        f"±{summary['macro_f1_std']:.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]"
    )
    return summary
