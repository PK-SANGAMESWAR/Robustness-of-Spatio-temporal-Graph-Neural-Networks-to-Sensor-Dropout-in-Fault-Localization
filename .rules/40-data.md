# 40 — Data, splits, and provenance

## The frozen corpus

`dataset_fullscale_50` — the definitive corpus. **Do not regenerate or overwrite it.**

- 825 scenarios (25 fault buses × 11 fault types × 3 resistances) × 50 runs × 40 windows
  = **1,650,000 windows** / 41,250 events.
- `dataset_fullscale_100` = 3.30 M windows, generated but its sweep is **parked**
  (see `00-execution.md`).
- Exactly **50 % no-fault** by construction.
- One window = `[N=25, S=20, F=6]` — 3 RMS voltages + 3 availability-mask channels. 26 classes.
- Graph: **25 nodes, 24 edges**, a connected substation-rooted tree over measured nodes only.
- **6 of the 25 fault buses are μPMU nodes → 19 fault locations are never directly measured.**

## Splits

Event-level **70/15/15**, verified **0 leakage** (`processing/splits.py::verify_no_leakage`) —
no window of an event may appear in two splits. 40.0 windows/event exactly.
Train 1,155,000 / val 248,000 / test 247,000 windows (6,175 test events).

Normalization touches **only voltage channels 0–2**, fit on train only. Never normalize the mask
channels — they are categorical instrumentation flags, not signals.

## Class balance is handled by the corpus, not the loss

**Unweighted cross-entropy is frozen** — `train.py` passes no `weight`. Imbalance is handled by
the balanced factorial design (50 % no-fault; 25 fault classes equal). Do not add class weights,
and do not describe the paper as using weighted or focal loss. (`common.py` *contains*
weighted/focal CE; it is not used by the frozen config.)

## Mask-diversity bugfix — provenance

Fixed 2026-08-13 (`6a00c4e` / `798406e`). `evaluate_at_dropout` passed a constant `mask_seed` to
`apply_sensor_dropout_batch`, which reseeds per call, so the same `batch_size` node subsets
repeated for every batch. The seed is now derived from `batch_idx`
(`mask_seed*10000 + batch_idx`).

**Every dropout number was regenerated.** Effect: shifts in the 2nd–3rd decimal only; no
conclusion changed. Pre-bugfix artefacts are preserved as `*.prebugfix.{csv,json,png}` — keep
them, never mix them with current numbers, never quote them.

## Dropout policies — where each lives

| Policy | Module | Notes |
|---|---|---|
| Random per-window (`zero_messages`) | `dropout/masking.py`, `dropout/evaluator.py` | The default sweep. `drop_edges` is computed but discarded — only `zero_messages` is live |
| Dropout-aware training | `dropout/training.py` | `DropoutAugmentedLoader` |
| Targeted (worst-case) | `dropout/targeted.py` | k nearest the fault by hop distance on the **full 123-bus** topology; deterministic given the label, so **no mask seeds** |
| Persistent (per-event) | `dropout/persistent.py` | Drawn once from `(event_id, k, mask_seed)`, held across the event's 40 windows |
| Neighbour-mean imputation | `scripts/run_imputation_baseline.py` | **Not a masking policy** — an inference-time baseline. Do not move it into `dropout/` |

Paper text describing dropout **must match `masking.py`**. If one changes, change the other in the
same edit.
