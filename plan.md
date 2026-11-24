# IndicSent-M — Project Plan
> **Single source of truth.** All decisions, ownership boundaries, data contracts, and milestones live here.
> Last updated: 2026-11-24 (v2 — amendments A-01 through A-06 applied)

---

## 1. Project Goal

Build **IndicSent-M**, a multimodal (text + image) 3-class sentiment classifier
(Positive / Neutral / Negative) for Bengali internet memes, using the **MemoSen**
dataset. We compare text-only and multimodal architectures, Indic versus multilingual
encoders, and two fusion strategies, then explain predictions with XAI tools.

### Research Questions

| ID | Question |
|----|----------|
| **RQ1** | Does multimodal input (text + image) outperform text-only models in macro-F1 on Bengali meme sentiment, particularly for the rare Neutral class? |
| **RQ2** | Does attention-based cross-modal fusion outperform simple concatenation and gated fusion in macro-F1 and especially on the minority class? |
| **RQ3** | Among Indic-specialised encoders (MuRIL, IndicBERT) and the multilingual baseline (XLM-R), which achieves best macro-F1 on Bengali mixed-script meme text? |
| **RQ4** | For samples where text-only and image-only models disagree, or where fusion changes the prediction, which modality is the decisive driver — and what text tokens or image regions are most attributable? *(MemoSen has no sarcasm labels; disagreement cases serve as the analysis lens.)* |

---

## 2. Ownership Table

> [!IMPORTANT]
> Friend 2's files are **off-limits**. Never create or modify anything in their scope.
> **Friend 2 must use `data/processed/*.csv` exactly as written — no re-splitting,
> no re-ordering. See §6 for the MD5 hashes that lock the CSVs.**

| Owner | Files / Directories |
|-------|---------------------|
| **Friend 1 (Text & Paper Lead — me)** | `src/common/` (data loading, metrics, training loop) |
| | `src/models/text_models.py` (XLM-R, MuRIL, IndicBERT text heads) |
| | `src/models/fusion_concat_gated.py` (M1 concat + ResNet-50, M3 gated) |
| | `src/models/image_only.py` (I1 image-only linear head on Friend 2's features) |
| | `src/xai/text_xai.py` (Integrated Gradients, attention rollout) |
| | `scripts/prepare_data.py` (data preparation pipeline) |
| | `run_text_experiments.py` (orchestrates T1/T2/T3/M1/M3/I1 runs) |
| | `results/` (all CSV logs, figures, `data_summary.md`) |
| | `plan.md` (this file) |
| **Friend 2 (Vision Lead)** | `src/models/vision_models.py` |
| | `src/models/fusion_crossattn.py` (M2 concat + best image enc, M4 cross-attention) |
| | `src/xai/grad_cam.py` |
| | `demo/` (Streamlit app) |

---

## 3. Directory Layout & Data Contract

```
c:\NLP Project\                        ← repo root
├── plan.md                            ← this file
├── run_text_experiments.py
├── data/
│   ├── multi-sent.xlsx                ← RAW (READ-ONLY, never modify)
│   ├── Memes/                         ← RAW image folder (READ-ONLY)
│   ├── processed/
│   │   ├── train.csv                  ← MD5 locked (see §6)
│   │   ├── valid.csv                  ← MD5 locked (see §6)
│   │   └── test.csv                   ← MD5 locked (see §6)
│   └── features/
│       ├── text_{model}_{split}.pt    ← token dict (see below)
│       └── image_{model}_{split}.pt  ← [N, D] float32 (Friend 2 writes these)
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── data.py                    ← Dataset class, collators
│   │   ├── metrics.py                 ← metric computation helpers
│   │   └── trainer.py                 ← generic training loop
│   ├── models/
│   │   ├── text_models.py             ← T1/T2/T3 architectures
│   │   ├── fusion_concat_gated.py     ← M1 / M3 architectures
│   │   └── image_only.py              ← I1 architecture
│   └── xai/
│       └── text_xai.py
├── scripts/
│   └── prepare_data.py
└── results/
    ├── data_summary.md
    ├── runs/                          ← one sub-dir per experiment run
    └── figures/
```

### Data Contract

#### Processed CSVs — `data/processed/{train,valid,test}.csv`

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | 0-based row index from the original Excel (before drop & split) |
| `image_path` | str | Relative path from repo root: `data/Memes/<filename>` |
| `text` | str | Cleaned caption (NFC, no URLs/handles, emoji→token, placeholder if empty) |
| `label` | str | Canonical form: `Negative`, `Neutral`, `Positive` (title-cased) |
| `label_id` | int | `Negative=0`, `Neutral=1`, `Positive=2` |
| `lang_tag` | str | `bn`, `en`, or `mixed` (from Bengali script character ratio) |
| `dup_group` | str | MD5 image hash (all rows sharing a byte-identical image get the same hash; unique images get the hash as their own group ID) |

> [!IMPORTANT]
> `dup_group` is used at split time to ensure **all members of an MD5 image-duplicate
> group land in the same split** (group-aware stratified split). It is also written
> to the CSV so Friend 2's code can inspect it.

#### Feature Files — `data/features/text_{model}_{split}.pt`

Each `.pt` is a **Python dict** saved with `torch.save`:

```python
{
  "cls":    Tensor[N, 768],           # float32, CLS token embedding
  "mean":   Tensor[N, 768],           # float32, masked mean-pool of token embeddings
  "tokens": Tensor[N, L, 768],        # float16, all token embeddings (L = max_seq_len)
  "mask":   Tensor[N, L],             # bool, attention mask
  "ids":    Tensor[N],                # int64, row IDs from processed CSV
}
```

> [!NOTE]
> `tokens` is stored fp16 to fit 4 GB VRAM. `mean` is float32 (same as `cls`).
> Row order in each `.pt` matches the row order of the corresponding processed CSV exactly.
>
> **`max_seq_len` is set to 128 tokens.** Justification (see D-07):
> Real XLM-R token length distribution on the cleaned train set (N=3056):
> p50=27, p90=47, p95=55, p99=72, max=235 tokens; mean=29.3, std=14.4.
> At 128 tokens, only **1 sample (0.03%)** in training is truncated.
> At 64 tokens, 51 samples (1.67%) would be truncated — unacceptable.
> 128 is the validated choice; reducing to 64 would cause unnecessary information
> loss on the longest captions without meaningful VRAM savings.

---

## 4. Experiment Matrix

> All experiments: **frozen encoder, trained head only** (RTX 3050 4 GB constraint).
> Each experiment: **5 seeds** (seeds 0, 1, 2, 3, 4). Report **mean ± std**.

### Text-Only Baselines

| ID | Encoder | Head | Input |
|----|---------|------|-------|
| **T1** | `xlm-roberta-base` | Linear(768→3) + dropout | CLS embedding |
| **T2** | `google/muril-base-cased` | Linear(768→3) + dropout | CLS embedding |
| **T3** | `csebuetnlp/banglabert` | Linear(768→3) + dropout | CLS embedding |

### Image-Only Baseline

| ID | Encoder | Head | Input |
|----|---------|------|-------|
| **I1** | ResNet-50 (frozen, Friend 2 extracts features) | Linear(2048→3) + dropout | GAP image embedding |

> [!NOTE]
> I1 uses **Friend 2's image feature files** (`data/features/image_resnet50_{split}.pt`).
> Friend 1 owns the I1 model head and training code in `src/models/image_only.py`.

### Multimodal Models

| ID | Text Enc | Image Enc | Fusion | Owner |
|----|----------|-----------|--------|-------|
| **M1** | Best from T1-T3 | ResNet-50 (frozen) | Concatenation [text ‖ image] → Linear | **Friend 1** |
| **M2** | Best from T1-T3 | Best image enc (Friend 2) | Concatenation [text ‖ image] → Linear | **Friend 2** |
| **M3** | Best from T1-T3 | ResNet-50 (frozen) | Gated multimodal unit (learned gate α) | **Friend 1** |
| **M4** | Best from T1-T3 | Best image enc (Friend 2) | Cross-attention | **Friend 2** |

> [!NOTE]
> The best text encoder (for M1-M4) is chosen after T1-T3 results are confirmed.
> The best image encoder (for M2/M4) is chosen after Friend 2's vision experiments.

### Hyperparameters (defaults, tunable)

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR | 3e-4 (head only) |
| Batch size | 32 |
| Epochs | 20 (early stop patience 5 on valid macro-F1) |
| Loss | Cross-entropy with class weights (inverse frequency) |
| Max seq len | 128 tokens (see §3 feature contract justification) |

---

## 5. Metrics

**Headline metric:** Macro-F1 (treats all 3 classes equally, critical given Neutral imbalance).

| Metric | Reported |
|--------|---------|
| Accuracy | Yes |
| Per-class Precision / Recall / F1 | Yes (Negative, Neutral, Positive separately) |
| **Macro-F1** | **Yes — primary headline** |
| Weighted-F1 | Yes |
| Confusion matrix | Heatmap saved as PNG |

**Imbalance handling:** Class-weighted cross-entropy. Weights computed per split as
`w_c = N_total / (3 × N_c)` using training-set counts only.

> [!IMPORTANT]
> Dataset label distribution after dropping 4 missing-image rows (N = 4368):
> - Negative: 2728 (62.5%)
> - Positive: 1349 (30.9%)
> - Neutral:   291  ( 6.7%)
>
> Neutral is severely under-represented. Class weighting is mandatory.

---

## 6. Split Strategy & CSV Integrity

> [!WARNING]
> **The raw Excel (`multi-sent.xlsx`) has no built-in train/valid/test column.**
> This was confirmed by inspection on 2026-11-24.
>
> **Decision (see D-01):** Stratified 70/15/15 split at seed=42, applied once to
> the 4368 valid rows. All subsequent runs read the saved CSVs. Never re-split.

### Group-Aware Split (Amendment A-02)

Before splitting, all rows are assigned a `dup_group` = MD5 hash of their image
file. The 5 rows whose images share content with another row (see §7 I-06) are
placed in the same split partition as their duplicate-of counterpart. Stratification
is performed at the **group level**, not the row level, to prevent leakage.

### Exact Expected Counts (N = 4368 after dropping 4 rows)

| Split | N | Negative | Positive | Neutral |
|-------|---|----------|----------|---------|
| Train | 3056 | 1909 | 944 | 203 |
| Valid | 658 | 410 | 203 | 45 |
| Test  | 654 | 409 | 202 | 43 |

*(Counts computed with floor rounding per class: 70%/15%/15%, remainder assigned to valid.)*

### Dropped Rows Before Split

| Excel Row | image_name | label |
|-----------|-----------|-------|
| 3517 | `KAM (103) (1).jpg` | positive |
| 3518 | `KAM (103) (1).png` | positive |
| 3569 | `KAM (128) (1).jpg` | negative |
| 3570 | `KAM (128) (1).png` | negative |

### CSV MD5 Hashes (written here after `prepare_data.py` runs)

> [!IMPORTANT]
> These hashes are the contract between Friend 1 and Friend 2.
> Friend 2 must verify their local copies match before training.

| File | MD5 |
|------|-----|
| `data/processed/train.csv` | `bd5c5dca4847c65fd46d34f0d6aca0f5` |
| `data/processed/valid.csv` | `f13825ad554d6d41b41741c365c9a3eb` |
| `data/processed/test.csv`  | `a9e11ade07d7d6c6c2c39f7066a4aeb4` |

---

## 7. Data Issues Found During Inspection (2026-11-24)

> [!WARNING]
> These issues must be acknowledged and handled explicitly in `prepare_data.py`.
> Do **not** silently fix them.

| # | Issue | Count | Severity | Plan |
|---|-------|-------|----------|------|
| I-01 | Missing-image rows | 4 rows | High | Drop before split (see §6). Log in `data_summary.md`. |
| I-02 | Orphaned disk image not in spreadsheet: `KAM (796).jpg` | 1 file | Low | Leave untouched. Note in `data_summary.md`. |
| I-03 | Mixed image extensions (.jpg 4099, .png 270) | — | Low | Use extension as-is from `image_name` column. |
| I-04 | No pre-existing train/valid/test split | — | High | Stratified 70/15/15 at seed=42 (see §6). |
| I-05 | Label strings are lowercase | — | Low | Title-case in `prepare_data.py`. |
| I-06 | **Image MD5 duplicates** | 5 rows | Medium | Assign same `dup_group`; keep all rows (do not drop); group-aware split ensures they land in same partition. Report in `data_summary.md`. Details: `KAM(152)==KAM(121)`, `KAM(206)==KAM(205)`, `KAM(214)==KAM(213)`, `KAM(631)==KAM(630)`, `KAM(780)==KAM(779)`. |
| I-07 | **Exact caption duplicates** | 36 rows across 11 distinct captions | Medium | Keep all rows (captions shared across different meme images is a real dataset property, not an error). Report in `data_summary.md`. Top case: 16 rows with caption `'No Caption'`. |
| I-08 | **Filename-stem "duplicates"** | The `(N)` suffix removal naively collapses all files with the same source page name (e.g. all `tangaila (N).jpg` → stem `tangaila`). These are **not** duplicates — they are distinct memes from the same page. | — | No action needed. The stem audit is misleading; real duplication is captured by I-06 (MD5) and I-07 (caption). |

---

## 8. Milestones & Checklist

### M0 — Planning
- [x] Inspect raw workspace and Excel structure
- [x] Write `plan.md` v1
- [x] Apply amendments A-01 through A-06; write `plan.md` v2
- [x] Get plan v2 approved by Friend 1

### M1 — Data Preparation
- [x] Run `scripts/prepare_data.py` → `data/processed/{train,valid,test}.csv`
- [x] Verify row counts match §6 expected counts
- [x] Duplicate audit output reviewed by Friend 1
- [x] `results/data_summary.md` written and reviewed
- [x] MD5 hashes of CSVs written into §6 of this plan
- [x] `plan.md` decisions log updated

### M2 — Feature Extraction
- [x] Extract text features for T1 (XLM-R): `data/features/text_xlmr_{train,valid,test}.pt`
- [x] Extract text features for T2 (MuRIL): `data/features/text_muril_{train,valid,test}.pt`
- [x] Extract text features for T3 (BanglaBERT, key `indicbert`): `data/features/text_indicbert_{train,valid,test}.pt`
- [x] Verify `cls`, `mean`, `tokens`, `mask`, `ids` shapes and row-ID alignment
- [x] Confirm `max_seq_len=128` after inspecting real token lengths across all 3 tokenizers

### M3 — Text-Only Experiments (T1, T2, T3)
- [x] `src/common/data.py`, `metrics.py`, `trainer.py` written and tested
- [x] `src/models/text_models.py` written
- [x] T1 (XLM-R) 5-seed run complete; results in `results/runs/T1/` (+ `T1b` ablation)
- [x] T2 (MuRIL) 5-seed run complete; results in `results/runs/T2/` (+ `T2b` ablation)
- [x] T3 (BanglaBERT) 5-seed run complete; results in `results/runs/T3/` (+ `T3b` ablation)
- [x] RQ3 answer drafted in `results/text_baselines.md`

### M4 — Image-Only Baseline (I1)
- [x] `src/models/image_only.py` written
- [ ] I1 5-seed run complete (blocked: waiting on Friend 2's `image_resnet50_*.pt`); results in `results/runs/I1/`

### M5 — Multimodal Experiments (M1, M3)
- [x] `src/models/fusion_concat_gated.py` written (`T-ctrl`, `M1`, `M3`)
- [x] `T-ctrl` (capacity-matched text control) 5-seed run complete; results in `results/runs/T-ctrl/`
- [ ] M1 (concat + ResNet-50) 5-seed run complete (blocked: waiting on Friend 2's `image_resnet50_*.pt`)
- [ ] M3 (gated + ResNet-50) 5-seed run complete (blocked: waiting on Friend 2's `image_resnet50_*.pt`)
- [ ] RQ1 and RQ2 (partial — concat vs gated, ResNet-50) answers drafted

### M6 — XAI
- [x] `src/xai/text_xai.py` written and run on `T1b` (Integrated Gradients + faithfulness + attention rollout)
- [ ] Disagreement case studies (text-only vs image-only vs fusion predictions differ) selected (blocked: waiting on Friend 2's `image_resnet50_*.pt`)
- [x] RQ4 (text side) answer drafted in `results/report_draft.md`

### M7 — Write-Up
- [ ] Results tables populated in paper
- [ ] All figures (confusion matrices, attention heatmaps) generated
- [ ] `data_summary.md` reviewed for dataset datasheet section
- [ ] `plan.md` all checkboxes ticked

---

## 9. Decisions Log

> Append a new entry every time a non-trivial decision is made.

| ID | Date | Topic | Decision | Rationale |
|----|------|-------|----------|-----------|
| D-01 | 2026-11-24 | Train/valid/test split | Stratified 70/15/15, seed=42, group-aware, applied once to 4368 valid rows | Raw Excel has no official split. Stratification preserves class imbalance. Group-awareness prevents image-duplicate leakage. Seed fixed for reproducibility. |
| D-02 | 2026-11-24 | Missing-image rows | Exclude rows 3517, 3518, 3569, 3570 before processing | Images not on disk; inclusion causes runtime crashes. Exclusion is 4 rows < 0.1% of data. Labels: 2× positive (KAM 103), 1× negative + 1 dup (KAM 128). |
| D-03 | 2026-11-24 | Label mapping | `negative→0`, `neutral→1`, `positive→2` | Alphabetical; Neutral in middle for confusion matrix readability. |
| D-04 | 2026-11-24 | Emoji handling | `emoji.demojize()` → descriptive tokens; do not delete | Emojis carry strong sentiment signal in meme captions. Deletion loses information. |
| D-05 | 2026-11-24 | lang_tag thresholds | bn > 70% Bengali chars, en < 30%, else mixed | Code-switching is common; three-way tag is more informative than binary. |
| D-06 | 2026-11-24 | Image MD5 duplicates | Keep all 5 duplicate-image rows; assign same `dup_group`; group-aware split | Dropping would silently change dataset size; the same image with different captions/labels is a real data property worth preserving. Group-aware split prevents leakage. |
| D-07 | 2026-11-24 | max_seq_len | **128 tokens** (confirmed) | Real XLM-R token distribution on cleaned train (N=3056): p50=27, p90=47, p95=55, p99=72, max=235. At 128: only 1 sample truncated (0.03%). At 64: 51 samples truncated (1.67%). 128 is the right choice. |
| D-08 | 2026-11-24 | Feature contract — add `mean` key | Add masked mean-pool `mean:[N,768]` float32 alongside `cls` in all feature .pt files | Mean-pool often outperforms CLS on shorter/noisier text; Friend 2's cross-attention may also benefit from it. No VRAM cost (float32 same as cls). |
| D-09 | 2026-11-24 | Experiment rename | M1=concat+ResNet-50 (me), M2=concat+best-img (Friend 2), M3=gated+ResNet-50 (me), M4=cross-attn (Friend 2); add I1=image-only-linear (me) | Separates Friend 1 and Friend 2 multimodal experiments clearly; I1 provides a unimodal image baseline owned by Friend 1 using Friend 2's extracted features. |
| D-10 | 2026-11-24 | RQ4 reworded | Replace "sarcasm-labelled samples" with "samples where text-only and image-only predictions disagree, or where fusion changes the prediction" | MemoSen has no sarcasm labels. Disagreement cases are a valid, label-free proxy for hard/ironic instances. |
| D-11 | 2026-11-24 | CSV integrity rule | MD5 hashes of the three processed CSVs are written into §6 of plan.md after first `prepare_data.py` run; Friend 2 must verify before training | Prevents silent divergence between teammates' copies of the processed data. |
| D-12 | 2026-11-24 | Caption duplicates | Keep all 36 rows with duplicated captions (11 distinct captions, top: 16× 'No Caption') | Different memes legitimately share captions (e.g., template memes). Dropping would silently alter the dataset. Reported in data_summary.md. |
| D-13 | 2026-11-24 | Filename-stem "duplicates" | Not real duplicates — they are distinct memes from the same source page. No action. | The `(N)` suffix removal naively conflates all tangaila/KAM/etc. files. Real duplication is captured by MD5 audit (I-06). |
| D-14 | 2026-11-25 | T3 encoder substitution | Replace `ai4bharat/indic-bert` with `csebuetnlp/banglabert` (key `indicbert` kept for file naming) | `ai4bharat/indic-bert` is a gated repo (`401 GatedRepoError`). `csebuetnlp/banglabert` is ungated, lightweight (32k vocab, ~110M params), outputs 768-dim embeddings, and is strong on Bengali NLP. |
| D-15 | 2026-11-25 | Token-length verification (`max_seq_len=128`) | Keep `max_seq_len=128` across all 3 encoders | Train token-length p50/p95/p99 (trunc@128): XLM-R 27/55/72 (0.03%), MuRIL 23/44/58 (0.03%), BanglaBERT 22/43/58 (0.07%). 128 covers >99.9% of samples. |
| D-16 | 2026-11-25 | Best text encoder + pooling | Select `T1b` (`xlm-roberta-base`, mean-pool + 2-layer MLP) as text branch for multimodal models | Highest validation macro-F1 (`0.4814` vs `T3b` `0.4550`, `T1` `0.4468`, `T3` `0.4461`, `T2b` `0.4086`, `T2` `0.3061`). Chosen strictly on validation macro-F1 without using test metrics. |
| D-17 | 2026-11-25 | Phase 3 partial execution | Run `T-ctrl`, `T1b` Text XAI, and report draft; leave `I1`, `M1`, `M3`, gate `z`, and disagreement XAI pending | Friend 2's `data/features/image_resnet50_{train,valid,test}.pt` are missing. Per hard rules, no placeholder features were created. |
| D-18 | 2026-11-25 | Text XAI faithfulness scope | Stratified `n=60` test subset (`seed=42`, `k=3` words removed, `n_steps=32`) | Live IG through XLM-R on CPU takes ~2s/sample; `n=60` gives a deterministic, class-stratified faithfulness measurement in ~2 min while preserving full 12-sample (6 correct + 6 incorrect) attribution outputs. |
