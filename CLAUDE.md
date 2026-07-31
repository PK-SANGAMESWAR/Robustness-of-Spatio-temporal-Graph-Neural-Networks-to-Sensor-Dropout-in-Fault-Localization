# CLAUDE.md — repo map & orientation (READ THIS FIRST to avoid re-exploring)

**Project:** conference paper reproducing Karabulut et al. (2026) STGNN fault-localization
baselines on the IEEE-123 feeder under partial μPMU observability, plus the novel
**sensor-dropout robustness** contribution (degradation curves + dropout-aware training).
Target venue ICMLDE 2026, deadline **15 Aug 2026**.

## Authoritative docs (read these for state/plan; don't re-derive)
- `PROGRESS.md` — full project journey + live execution log (§14). **Source of truth for status.**
- `NEXT-STEPS.md` — the locked execution plan (Steps 1–6). Supersedes older roadmaps.
- `PROBLEM.md` — problem statement / scope. `QUESTIONS.md` — user's open questions.
- `README.md` — public GitHub readme. Repo is pushed to GitHub (private).

## Source package: `fault_localization/`
- `config/constants.py` — **all dataset/protocol constants** (FAULT_BUSES, PMU_BUSES, 11
  FAULT_TYPES, FAULT_RESISTANCES, load range 0.5–1.3, windowing: PRE_FAULT_MS=40 /
  FAULT_DURATION_MS=20 / WINDOW_LENGTH=20 / WINDOWS_PER_EVENT=40, RUNS_PER_SCENARIO=100,
  NUM_SCENARIOS=825, NUM_CLASSES=26, class map). Single source of truth for numbers.
- `models/` — `gru.py` (GRUBaseline), `graphsage.py` (RGSAGE mean/max), `gcn.py` (RGCN),
  `gatv2.py` (RGATv2), `common.py` (TemporalEncoder=shared 1-layer GRU-128; NodeClassifier
  BN→ReLU→dropout0.35→Linear-26; weighted/focal CE), `voting.py` (soft_vote = sum node
  softmax probs → argmax). All share GRU→[GNN]→head→soft-vote template. `num_layers` knob
  = spatial depth (frozen config uses 4).
- `dropout/` — the contribution's engine: `masking.py` (feature-masking; `drop_edges`
  computed but discarded → only `zero_messages` is live), `evaluator.py`
  (`evaluate_dropout_curve` over k, paired mask seeds), `training.py`
  (`DropoutAugmentedLoader`).
- `experiments/` — `runner.py` (train/eval loop, builds model via `_model`, supports
  grad_clip_norm/warmup_epochs/gnn_layers/disable_edges/dropout_study), `cli.py` (train
  entrypoint + flags), `config.py` (yaml load + recursive merge + validate), `train.py`
  (`train_epoch`/`validate_epoch`), `evaluate.py` (`evaluate_loader`→graph+node metrics),
  `seed.py`, `reporting.py`.
- `data/` — `dataset.py` (`WindowGraphDataset` returns `{"x":[N,S,6],"y"}`; `make_dataloader`;
  normalizes only voltage channels 0–2), `manifest.py` (dataset fingerprint/verify).
- `simulation/` — OpenDSS sim: `feeder.py`, `monitors.py` (Daily 1ms mode, voltage+mask
  extraction), `fault_injection.py`, `runner.py` (per-event sim, load uniform(0.5,1.3)).
- `processing/` — raw-sim → windows pipeline: `windowing.py`, `labels.py`, `splits.py`
  (event-level 70/15/15), `normalization.py`, `build_dataset.py`.
- `graph/` — measured-only graph construction (`measured_only.py` = Algorithm 1
  substation-rooted spanning tree; `edge_attributes.py`).
- `validation/` — `dataset_checks.py`, `verify_dataset.py`, `visualization.py`.
- `analysis/errors.py`. IEEE-123 DSS source files: `data/ieee123/*.DSS` + config yaml.

## Scripts (`scripts/`) — all `uv run python scripts/<x>.py`
- **Definitive 50-run sweep (RUNNING):** `run_fullscale_final_sweep.py` → `results/fullscale_final/`.
  Summary/figures: `build_fullscale_final_summary.py`, `build_fullscale_final_figures.py`
  (5-figure set). Shared metric extraction: `_metrics.py`.
- **100-run sweep (STAGED, parked):** `run_fullscale100_seed42_sweep.py` → would write
  `results/fullscale_final_100/`. Same frozen config, only DATA_ROOT differs. NOT launched.
- **Contribution (Step 4, staged, gated):** `run_dropout_study.py` (4a curves),
  `run_dropout_aware_sweep.py` (4b retrain), `build_dropout_comparison.py` (overlay).
  De-risk check: `smoke_test_dropout.py` (passed).
- **Paper dataset assets:** `build_dataset_assets.py` → `paper/tables/dataset_info.tex` +
  `paper/figures/{measured_only_graph,fault_signature}.png` (all counts from real data;
  `--data-root`/`--runs-per-scenario` regenerate for 100-run).
- **Dataset generation:** `generate_trend_dataset.py`, `generate_pmu_metadata.py`.
- Older/diagnostic (context only): `run_trend_sweep.py`, `run_investigate_probes*.py`,
  `run_fullscale50_seed42_sweep.py`, `build_{trend,seed42,investigate,fullscale}_*.py`.

## Config (`config/`)
`experiment_base.yaml` (base: device cuda, gnn_layers, training, dropout_study block) +
per-model `{gru,rgcn,rgsage_mean,rgsage_max,rgatv2}.yaml` (each `base:` experiment_base).
Frozen sweep config is applied via CLI flags (`--gnn-layers 4 --epochs 30 --grad-clip-norm
1.0 --warmup-epochs 3`), not baked into yaml.

## Results (`results/`)
- `fullscale_final/<model>/seed_00042/` — **the definitive 50-run sweep** (metrics.json,
  checkpoint_best.pt, predictions_test.csv, metrics_by_epoch.csv, config.yaml). Figures →
  `fullscale_final/figures/`. Log → `fullscale_final_sweep.log`.
- Per-run metrics.json keys: `test_graph.f1` (soft-voted macro-F1), `test_node_diagnostic.f1`
  (per-window), `test_graph.per_class.{precision,recall}` (class 0 = no-fault),
  `best_validation_graph_f1`.
- Others are historical: `benchmark_v1.1_samplerfix` (tiny, OLD model structure — won't
  load), `fullscale_50_seed42` (pre-frozen depth-1), `diag_*` (depth/stability probes),
  `trend_*` (medium-A).

## Paper (`paper/`)
- `overleaf.tex` — **the live paper** (Elsevier CRC/Procedia). Written so far: abstract,
  intro, related-work table (§Related Work), Grid Simulation & Dataset (+ table + 2 figs),
  Baseline Architectures. Empty: dropout-eval/dropout-aware methods, setup, results,
  discussion, conclusion. Bibliography complete (13 entries, all cite-resolved).
- `figures/`, `tables/` — generated assets for Overleaf upload.
- `0*.md`, `RELATED-WORK.md` — earlier markdown drafts (superseded by overleaf.tex).

## Key facts (verified against code — safe to reuse without re-checking)
- Dataset: 50-run corpus `dataset_fullscale_50` = 1.65M windows (825×50×40); 100-run
  `dataset_fullscale_100` = 3.30M (both on disk, ready). Exactly 50% no-fault. Paper's
  reported 2.5M is NOT reconstructable from 825 grid — documented deviation.
- 6 of 25 fault buses are μPMU nodes → 19 fault locations unmeasured. Graph = 25-node
  24-edge connected tree.
- Reproduction accepted at depth-4 (deviation from paper's 1 layer). Weighted-CE frozen.
- Stabilization (grad-clip 1.0 + warmup 3) on EVERY run. GPU = T1000 8GB (NVML monitoring
  broken but training unaffected).

## Hard rules
Identical frozen data/splits across models; change one variable at a time; code passing
tests ≠ validated (confirm with a real run); dropout paper text must match `masking.py`;
STOP + report after NEXT-STEPS Steps 1/2/4 (don't auto-chain).
