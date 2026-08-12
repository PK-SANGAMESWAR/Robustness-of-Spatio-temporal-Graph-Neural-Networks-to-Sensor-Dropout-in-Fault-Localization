# CLAUDE.md — repo map & orientation (READ THIS FIRST to avoid re-exploring)

**Project:** conference paper that **adopts** Karabulut et al. (2026)'s STGNN fault-localization
methodology on the IEEE-123 feeder under partial μPMU observability and reports **our own**
numbers (NOT a reproduction — no "reproduce"/0.95-matching language; keep attribution), plus the
**sensor-dropout robustness** contribution: degradation curves + **dropout-aware training**
(the effective, architecture-agnostic mitigation) + **SDR-STGNN** (an availability-gated
model, kept as an honest ablation — it MATCHES but does not beat dropout-aware training).
Target venue ICMLDE 2026, deadline **15 Aug 2026**.

**Status (2026-08-12): all experiments DONE; the paper is a complete draft.** Every model is
trained at seeds 42/43/44, every degradation curve is 3-seed, and `paper/overleaf.tex` has all
sections written (abstract → conclusion + bibliography). What remains is author-only: Overleaf
page count vs. the 12-page cap, iThenticate/similarity check, acknowledgements + corresponding-
author details, final AI-disclosure wording. The 100-run sweep is still parked (do NOT launch
without being asked).

## Authoritative docs (read these for state/plan; don't re-derive)
- `PROGRESS.md` — full project journey + execution log (§14). Source of truth for the journey;
  for *current* status trust this file's Status line + `results/` on disk.
- `NEXT-STEPS.md` — the locked execution plan (Steps 1–6; Step 6 VOID). Supersedes older roadmaps.
- `PROVENANCE.md` — **append-only AI-provenance ledger. Every session that edits
  `paper/overleaf.tex` or manuscript assets MUST append an entry** (per `paper/.rules/60-integrity.md`
  Part A). Never overwrite prior entries.
- `RUN-SDR-50RUN-RUNBOOK.md` — runbook for the SDR contribution on the 50-run corpus.
- `WRONG-CLAIMS.md` — gaps between paper claims and what's actually run (#1 green-feeder = resolved/cut).
- `PROBLEM.md` — problem statement / scope. `QUESTIONS.md` — user's open questions.
- `RUN-100RUN-SWEEP.md` — **runbook: how to launch/monitor the full-scale (100-run/3.30M)
  sweep in a fresh session.** Follow it only if explicitly asked to run 100-run.
- `CURRENT-TRAINING.md` — **STALE** (2026-08-07 snapshot; the 8 missing model-seed dropout
  evaluations it lists have since been run). Don't treat it as current status.
- `README.md` — public GitHub readme. Repo is pushed to GitHub (private).
- `paper/PAPER-CLAUDE.md` + `paper/.rules/` — paper-writing conventions (Procedia/Vancouver,
  LaTeX, figures, page limit, integrity, quality bar). Read when editing `paper/overleaf.tex`.
- `IMP/paper-points.md`, `IMP/points-for-paper.md` — the author's own raw framing notes
  (honesty on dataset size, "same protocol, our own regeneration", no numeric replication claim).

## Source package: `fault_localization/`
- `config/constants.py` — **all dataset/protocol constants** (FAULT_BUSES, PMU_BUSES, 11
  FAULT_TYPES, FAULT_RESISTANCES, load range 0.5–1.3, windowing: PRE_FAULT_MS=40 /
  FAULT_DURATION_MS=20 / WINDOW_LENGTH=20 / WINDOWS_PER_EVENT=40, RUNS_PER_SCENARIO=100,
  NUM_SCENARIOS=825, NUM_CLASSES=26, class map; GREEN_FEEDER_* switch buses). Single source
  of truth for numbers.
- `models/` — `gru.py` (GRUBaseline), `graphsage.py` (RGSAGE mean/max), `gcn.py` (RGCN),
  `gatv2.py` (RGATv2), `common.py` (TemporalEncoder=shared 1-layer GRU-128; NodeClassifier
  BN→ReLU→dropout0.35→Linear-26; weighted/focal CE), `voting.py` (soft_vote = sum node
  softmax probs → argmax), `sdr_stgnn.py` (**SDRSTGNN** = RGSAGE-Max + availability-gated
  aggregation: dropped nodes emit no messages; gate keyed on original availability at every
  hop; k=0 ≡ RGSAGE-Max). All share GRU→[GNN]→head→soft-vote template. `num_layers` knob
  = spatial depth (frozen config uses 4).
- `dropout/` — the contribution's engine: `masking.py` (feature-masking; `drop_edges`
  computed but discarded → only `zero_messages` is live), `evaluator.py`
  (`evaluate_dropout_curve` over k, paired mask seeds; soft-vote now matches `voting.py`),
  `training.py` (`DropoutAugmentedLoader`).
- `experiments/` — `runner.py` (train/eval loop, builds model via `_model`, supports
  grad_clip_norm/warmup_epochs/gnn_layers/disable_edges/dropout_study), `cli.py` (train
  entrypoint + flags), `config.py` (yaml load + recursive merge + validate), `train.py`
  (`train_epoch`/`validate_epoch`), `evaluate.py` (`evaluate_loader`→graph+node metrics),
  `seed.py`, `reporting.py`.
- `data/` — `dataset.py` (`WindowGraphDataset` returns `{"x":[N,S,6],"y"}`; `make_dataloader`;
  normalizes only voltage channels 0–2), `manifest.py` (dataset fingerprint/verify).
- `simulation/` — OpenDSS sim: `feeder.py` (incl. `configure_green_feeder` — reconfigured
  topology, opens 60–160 / closes 54–94), `monitors.py` (Daily 1ms mode, voltage+mask
  extraction), `fault_injection.py`, `runner.py` (per-event sim, load uniform(0.5,1.3)).
- `processing/` — raw-sim → windows pipeline: `windowing.py`, `labels.py`, `splits.py`
  (**event-level 70/15/15** — verified 0 leakage; `verify_no_leakage`), `normalization.py`,
  `build_dataset.py`.
- `graph/` — measured-only graph construction (`measured_only.py` = Algorithm 1
  substation-rooted spanning tree; `edge_attributes.py`).
- `validation/` — `dataset_checks.py`, `verify_dataset.py`, `visualization.py`.
- `analysis/errors.py`. IEEE-123 DSS source files: `data/ieee123/*.DSS` + config yaml.

## Scripts (`scripts/`) — all `uv run python scripts/<x>.py`
- **Definitive 50-run sweep (DONE, 3 seeds):** `run_fullscale_final_sweep.py` (`SEEDS = [42,43,44]`)
  → `results/fullscale_final/` — all 5 baselines × seeds 42/43/44. Summary/figures:
  `build_fullscale_final_summary.py`, `build_fullscale_final_figures.py`. Metrics: `_metrics.py`.
- **SDR contribution (DONE):** `run_sdr_stgnn.py` (train SDR-STGNN, `--seed`),
  `run_dropout_aware_sweep.py` (dropout-aware; now takes `--models` **and `--seeds`**, resumable —
  any (model, seed) with an existing metrics.json is skipped), `run_dropout_study.py` (degradation
  curves; **`--seeds 42 43 44` aggregates training-seed error bars**), `build_sdr_result_figures.py`,
  `build_dropout_comparison.py`, `build_sdr_schematics.py`, `run_sdr_contribution.py` (orchestrator).
  De-risk: `smoke_test_dropout.py`.
- **100-run sweep (STAGED, parked):** `run_fullscale100_seed42_sweep.py` → would write
  `results/fullscale_final_100/`. Same frozen config, only DATA_ROOT differs. NOT launched.
- **Paper dataset assets:** `build_dataset_assets.py` → `paper/tables/dataset_info.tex` +
  `paper/figures/{measured_only_graph,fault_signature}.png` (all counts from real data;
  `--data-root`/`--runs-per-scenario` regenerate for 100-run).
- **Dataset generation:** `generate_trend_dataset.py`, `generate_pmu_metadata.py`.
- Older/diagnostic (context only): `run_trend_sweep.py`, `run_investigate_probes*.py`,
  `run_fullscale50_seed42_sweep.py`, `build_{trend,seed42,investigate,fullscale}_*.py`.

## Config (`config/`)
`experiment_base.yaml` (base: device cuda, gnn_layers, training, dropout_study block) +
per-model `{gru,rgcn,rgsage_mean,rgsage_max,rgatv2,sdr_stgnn}.yaml` (each `base:` experiment_base).
`config/dropout_aware/{gru,rgcn,rgsage_mean,rgsage_max,rgatv2}.yaml` are generated by
`run_dropout_aware_sweep.py` (dropout_aware_training=True) — all 5 now exist and are committed.
Frozen sweep config applied via CLI flags (`--gnn-layers 4
--epochs 30 --grad-clip-norm 1.0 --warmup-epochs 3`), not baked into the baseline yaml.

## Results (`results/`)
- `fullscale_final/<model>/seed_000{42,43,44}/` — **the definitive 50-run sweep**, all 5 baselines
  × 3 seeds (metrics.json, checkpoint_best.pt, predictions_test.csv, metrics_by_epoch.csv,
  confusion_matrix.csv, config.yaml). Aggregates: `benchmark_summary.csv` (mean±std over seeds,
  the paper's Table 3 source) + `benchmark_comparison.csv` (per-seed rows). Figures →
  `fullscale_final/figures/`.
- **SDR contribution:** `sdr_stgnn/sdr_stgnn/seed_000{42,43,44}/` (+ auto `dropout_curve.json`),
  `dropout_aware/<all 5 models>/seed_000{42,43,44}/`, and three 3-seed curve files —
  `dropout_study/degradation_curves.csv` (standard, 5 models), `dropout_study_aware/…` (dropout-aware,
  5 models), `dropout_study_sdr/…` (SDR-STGNN). Each row carries `f1_mean/f1_std/f1_per_train_seed`,
  `n_train_seeds=3`, `n_mask_seeds=3`, `policy=zero_messages`. Seed-42-only predecessors are kept as
  `*.bak.csv`. Result figures → `paper/figures/sdr_*.png`; summary `paper/figures/sdr_robustness_summary.csv`.
- Per-run metrics.json keys: `test_graph.f1` (soft-voted macro-F1), `test_node_diagnostic.f1`
  (per-window), `test_graph.per_class.{precision,recall}` (class 0 = no-fault),
  `best_validation_graph_f1`.
- Others are historical: `benchmark_v1.1_samplerfix` (tiny, OLD model structure — won't
  load), `fullscale_50_seed42` (pre-frozen depth-1), `diag_*` (depth/stability probes),
  `trend_*` (medium-A).

## Paper (`paper/`)
- `overleaf.tex` — **the live paper** (Elsevier CRC/Procedia), ~900 lines, **complete draft**:
  abstract, intro, related work (+`tab:related`), Materials & Methods (dataset +`tab:dataset`
  +`fig:graph`/`fig:sig`, baselines, deviations `tab:deviations`, dropout-eval protocol,
  dropout-aware training +`alg:aware`, SDR-STGNN), Setup, Results (`tab:baseline`,
  `tab:degradation`, `fig:degradation`, `tab:aware`, `fig:ablation`, architecture-agnostic
  subsection), Discussion (+limitations), Conclusion, bibliography (14 refs).
  Framing is reframed: dropout-aware training is the headline; SDR-STGNN is an ablation.
  The four wide tables use `table*` (full-width) to avoid two-column overflow.
- Revision docs: `paper-issue.md` (ICMLDE revision checklist — **executed**, with `paper-issue.tex`
  the pre-revision snapshot), `plan-paper.md` (the ~15pp→12pp trim plan + claim re-derivation).
- `figures/`, `tables/` — generated assets for Overleaf upload.
- `PAPER-CLAUDE.md` + `.rules/00-core,10-reading,20-research,30-writing,40-figures,50-pagelimit,
  60-integrity,70-quality` — paper conventions. Note: `50-pagelimit.md` still states 10 pages with
  the `Verified:` line unfilled, while the trim work targeted the **12-page** ICMLDE cap from
  `paper-issue.md` — confirm against the CFP before trusting either. `0*.md`, `RELATED-WORK.md`,
  `ALGORITHM-FORMULATION.md` — earlier markdown drafts (superseded by overleaf.tex).

## Key facts (verified against code — safe to reuse without re-checking)
- Dataset: 50-run corpus `dataset_fullscale_50` = 1.65M **windows** (825×50×40; event-level
  70/15/15, 0 leakage, 40.0 windows/event); 100-run `dataset_fullscale_100` = 3.30M. Exactly
  50% no-fault. Paper's reported 2.5M is NOT reconstructable from 825 grid — documented deviation.
- 6 of 25 fault buses are μPMU nodes → 19 fault locations unmeasured. Graph = 25-node
  24-edge connected tree. One window = `[N=25, S=20, F=6]` (3 RMS volts + 3 mask channels).
- Reproduction accepted at depth-4 (deviation from paper's 1 layer). **Unweighted CE frozen**
  (train.py passes no `weight`; imbalance handled by the balanced corpus — 50% no-fault, 25
  fault classes equal by factorial design — NOT by loss weighting; paper text fixed to match).
- **SDR-STGNN finding (interim seed-42):** robustness AUDC — standard RGSAGE-Max **0.41** →
  dropout-aware **0.92** → SDR-STGNN **0.89**. The gate MATCHES but does NOT beat dropout-aware
  training (it silences dropped nodes at every hop, blocking multi-hop routing through recovered
  nodes). **Report faithfully — do not claim the gate wins.**
- **Multi-seed (50-run):** SDR-STGNN + dropout-aware RGSAGE-Max trained at seeds 42/43/44
  (deviation from the runbook's single-seed; run via a scratchpad orchestrator, autonomous).
- **Green-feeder scenario: ❌ CUT (2026-08-04), out of scope.** Not a contribution booster
  (a second-axis stress test only) and would need a new green corpus — dropped before the
  deadline. No green dataset/runs/claims; `configure_green_feeder` stays as dead/future-work
  code. Paper no longer promises it. See `WRONG-CLAIMS.md` #1 / `NEXT-STEPS.md` Step 6 (VOID).
- Stabilization (grad-clip 1.0 + warmup 3) on EVERY run. GPU = T1000 8GB (NVML monitoring
  broken but training unaffected).

## Hard rules
Identical frozen data/splits across models; change one variable at a time; code passing
tests ≠ validated (confirm with a real run); dropout paper text must match `masking.py`;
**report claims faithfully — SDR-STGNN does not beat dropout-aware training, don't overclaim.**
The 50-run SDR chain runs autonomously to completion; STOP + report before the 100-run sweep.
