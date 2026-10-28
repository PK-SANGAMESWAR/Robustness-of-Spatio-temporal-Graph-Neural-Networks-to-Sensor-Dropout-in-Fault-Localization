# PROPOSED_CHANGES.md — corrections and gaps in the frozen GridSense sources

The frozen sources are `implementation-plan.md` and `rag-eval.md` (and `RAG-arch.png`). Do not edit them. Write each
problem here with: the source location, the problem, the evidence, the proposal, and a status (`proposed`,
`accepted`, `rejected`). An `accepted` entry has precedence over the source text (`claude-rag.md` § 2).

## P1 — The figure gives the wrong macro-F1 for the dropout-aware model
- Source: `RAG-arch.png`, "Layer 1 · Predict: dropout-aware STGNN 0.846 macro-F1".
- Evidence: 0.846 is the **standard** RGSAGE-Max at k = 0 (`results/RESULTS.md` §1). The dropout-aware RGSAGE-Max is
  0.839 at k = 0 and 0.674 at k = 10 (`results/RESULTS.md` §3–4; per seed 0.8515 / 0.8334 / 0.8314 in
  `results/dropout_aware/rgsage_max/seed_000{42,43,44}/metrics.json`).
- Proposal: change the label to "dropout-aware STGNN, 0.839 macro-F1 (0.674 with 10 of 25 μPMUs dark)".
- Status: proposed (2026-10-28).

## P2 — The dropout grid does not match the paper protocol
- Source: `rag-eval.md` §4 Layer 0 ("0, 10, 20, 30 and 40 %"; "use the same dropout method as the ICMLDE paper");
  `implementation-plan.md` §13.12 and §20 (same percent grid).
- Evidence: the paper drops k ∈ {0, 1, 2, 5, 10} of 25 μPMUs (0, 4, 8, 20, 40 %). 10 % and 30 % are 2.5 and 7.5
  sensors, which the protocol cannot express. The paper curves exist only at the k grid.
- Proposal: use k ∈ {0, 1, 2, 5, 10}. Author decision 2026-10-28 (`claude-rag.md` § 3).
- Status: accepted (2026-10-28).

## P3 — "Calibrated probability" is not defined for the soft vote
- Source: `implementation-plan.md` §11.2, §11.3; `rag-eval.md` §4 Layer 0 (ECE, Brier).
- Evidence: `fault_localization/models/voting.py` returns the sum of 25 node softmaxes; the sum is 25, not 1.
- Proposal: define the calibrated score explicitly (O2). Do not call the raw vote a probability.
- Resolution (author 2026-10-28): vote share = vote / 25; Platt or isotonic on the top-1 share; no temperature on
  node logits, so predictions stay identical to the paper; record the top-1 − top-2 margin.
- Status: accepted (2026-10-28).

## P4 — The calibration split has been used once already
- Source: `implementation-plan.md` §13.7 (calibration split used for temperature, τ_p, τ_k, conformal sets).
- Evidence: the repo has train / val / test only. `val` selected the checkpoint (best-validation epoch;
  `metrics.json` `selection_metric`). There is no unused calibration split.
- Proposal: use `val` for calibration and state the small optimistic bias as a limitation. Do not take windows from
  `test`. (`val_sample.txt`, 30,000 windows, is a subset of `val`, not a separate split.)
- Resolution (author 2026-10-28): split `val` by run into cal-A (fit) and cal-B (thresholds); bias in limitations.
- Status: accepted (2026-10-28).

## P5 — Leakage sanity check cannot separate near duplicates
- Source: `implementation-plan.md` §13.9 ("kNN accuracy should jump to near 100 %; if it does not, the split leaks").
- Evidence: the split divides runs, not scenarios. All 825 scenarios are in train and test; runs of one scenario differ
  in the load multiplier (0.5–1.3, `config/constants.py`) and the run seed (`metadata.csv`). A train neighbour of the same scenario is a near
  duplicate by design, so high kNN accuracy without test in the store is expected, not a sign of a bug.
- Proposal: keep the sanity check, and add the scenario-held-out split (`dataset_fullscale_50_scenario/splits/`) as an
  informational control. Report both.
- Status: proposed.

## P6 — "Event" is a window in this repo
- Source: `rag-eval.md` §5.3 (event golden set), `implementation-plan.md` §12 (event state).
- Evidence: the STGNN classifies 20-step windows; a run has 40 windows (w0000–w0019 label 0, then fault windows).
- Proposal: author decision 2026-10-28 — the unit is the window; one report per event; run-grouped LOO and
  bootstrap; event aggregation informational only. Report window (O3): window w has w − 20 post-fault samples, so
  w0020 (label ≠ 0) has **none** (`models/common.py:unlabelable_window_mask`); the index is the earliest w ≥ 21
  where `val` accuracy stops rising.
- Status: accepted (2026-10-28).

## P7 — The corpus is the model card, not procedures
- Source: `implementation-plan.md` §5.1 (feeder docs, curated notes, illustrative SOP).
- Proposal: author decision 2026-10-28 — Paper 1 and `results/RESULTS.md` as `model_card` documents. Other sources
  are open (O7). New rule: no aggregate metric as the confidence of the current prediction (rubric + safety item).
- Status: accepted (2026-10-28).

## P8 — `rag-eval.md` §11 file locations
- Source: `rag-eval.md` §11 (`eval/README.md`, `eval/EXPERIMENT_LOG.md`), §7 (`configs/eval.yaml`).
- Proposal: the track docs live in `rag-for-dropout/` (`EVAL.md`, `EXPERIMENT_LOG.md`, `configs/eval.yaml`); code in
  `src/gridsense/` (author decision 2026-10-28).
- Status: accepted (2026-10-28).
