# CLAUDE.md — repo map & orientation (READ THIS FIRST to avoid re-exploring)

**Project:** conference paper that **adopts** Karabulut et al. (2026)'s STGNN fault-localization
methodology on the IEEE-123 feeder under partial μPMU observability and reports **our own**
numbers (NOT a reproduction — no "reproduce"/0.95-matching language; keep attribution), plus the
**sensor-dropout robustness** contribution: degradation curves + **dropout-aware training**
(the effective, architecture-agnostic mitigation) + **SDR-STGNN** (an availability-gated
model, kept as an honest ablation — it MATCHES but does not beat dropout-aware training).
Target venue ICMLDE 2026, deadline **15 Aug 2026**.

**Status (2026-08-16, ICMLDE deadline 15 Aug has passed — confirm with the author whether this
is now a resubmission/extension target before doing deadline-driven work): all experiments DONE
and re-run post-bugfix; the paper is a complete draft.** Every model is trained at seeds
42/43/44, every degradation curve is 3-seed, imputation + targeted + persistent dropout
baselines have been added, and `paper/overleaf.tex` has all sections written (abstract →
conclusion + bibliography). Remaining polish: **page count (`paper/preview-local.pdf` = 13 pp
vs. the 12-page cap)**, iThenticate/similarity check. The 100-run sweep is still parked (do NOT
launch without being asked).

**Two late experiments (2026-08-15, commit `20b5b68`) are in the paper but were undocumented in
markdown until 2026-08-16 — see "Key facts" below:** targeted (worst-case) dropout and
persistent (per-event) dropout, both on the RGSAGE-Max backbone at seeds 42/43/44.

**A large doc cleanup happened in `0f04bc8`→`20b5b68`.** ~25 markdown files were deleted
(`PROGRESS.md`, `NEXT-STEPS.md`, `FIGURES-GUIDE.md`, all runbooks, `docs/`, `IMP/`,
`paper/ISSUES.md`, `paper/REVIEW.md`, `CHANGES.md`, …). They are recoverable from git history
but are **not** current state. The doc map below lists only what actually exists on disk.

**Mask-diversity bugfix (2026-08-13, commit `6a00c4e`/`798406e`) — every dropout number was
re-generated.** `evaluate_at_dropout` passed a constant `mask_seed` to
`apply_sensor_dropout_batch`, which reseeds per call, so the same `batch_size` node subsets
repeated for every batch; the seed is now derived from `batch_idx`. `evaluate_dropout_curve`
was also rewritten as a single pass over the loader (all (k, mask-offset) combos inside one
batch loop) and `apply_sensor_dropout_batch` vectorised — both are pure speedups, masks
unchanged. Effect on results: **shifts in the 2nd–3rd decimal only, no conclusion changed.**
Pre-bugfix CSVs/figures are preserved as `*.prebugfix.{csv,json,png}` — do not delete, and do
not mix them with current numbers.

## Authoritative docs — THE COMPLETE LIST (verified on disk 2026-08-16)

**`paper/overleaf.tex` is now the single source of truth for the science.** It is ahead of every
markdown file: it already documents the targeted and persistent dropout experiments with
p-values. When a doc and the `.tex` disagree, the `.tex` wins; when the `.tex` and a CSV in
`results/` disagree, the CSV wins.

Only these markdown files exist:
- `CLAUDE.md` — this file. Repo map + status + key facts.
- `IMPUTATION.md` — the reviewer-requested imputation baseline: what it is, how it was run, how
  it is wired into the paper. **Current and accurate.** Read before touching
  `run_imputation_baseline.py`.
- `PROBLEM.md` — problem statement / scope.
- `QUESTIONS.md` — the author's questions + answers (Q1 seeds: resolved; Q2 100-run: answered).
- `README.md` — public GitHub readme. Repo is pushed to GitHub (private).
- `PPT-GUIDE-REVIEW.md` — slide-by-slide content for the guide review presentation, with
  verified figure assignments and anticipated questions (added 2026-08-16).
- `results/RESULTS.md` — current headline results table + a clearly-marked historical appendix.
- `REVIEW.md` — **empty placeholder** created by the author, untracked. Not a real doc yet.
- Generated, not hand-written: `results/sdr_stgnn/sdr_stgnn/seed_000*/dropout_curve.md`,
  `.rules/CUDA.md`, `.pytest_cache/README.md`.

**Deleted — do not cite, do not resurrect unless asked** (recoverable via
`git show <sha>:<path>`): `PROGRESS.md`, `NEXT-STEPS.md`, `FIGURES-GUIDE.md`,
`CURRENT-TRAINING.md`, `RUN-SDR-50RUN-RUNBOOK.md`, `RUN-100RUN-SWEEP.md`,
`FULLSCALE-GENERATION.md`, `IMPLEMENT.md`, `APPROACH.md`, `model-run.md` (all in `0f04bc8`);
`docs/**`, `IMP/**` (`ed47a52`); `paper/ISSUES.md`, `paper/ISSUES2.md`, `paper/REVIEW.md`
(`9691654`); `CHANGES.md` (`20b5b68`); `PROVENANCE.md`, `WRONG-CLAIMS.md`,
`GREEN-FEEDER-PLAN.md`, `paper/PAPER-CLAUDE.md`, `paper/.rules/`, `paper/0*.md` (`6a00c4e`).

**The two paper-editing rules from the deleted `paper/ISSUES.md` still apply** and are restated
here so they survive: **edit, don't restructure** (large rewrites carry AI-plagiarism risk), and
**never fabricate a number**. The only open item from the deleted `paper/REVIEW.md` is the
**12-page cap** (currently 13 pp); its other items (em-dash standardisation, `\eqref` cross-refs,
≥300 dpi figures) are all DONE.

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
- `dropout/` — the contribution's engine: `masking.py` (feature-masking, vectorised scatter;
  `drop_edges` computed but discarded → only `zero_messages` is live), `evaluator.py`
  (`evaluate_dropout_curve` = one pass over the loader covering every (k, mask-offset) combo;
  per-batch mask seed `mask_seed*10000 + batch_idx`; soft-vote matches `voting.py`),
  `training.py` (`DropoutAugmentedLoader`), `targeted.py` (worst-case selection: k μPMUs nearest
  the fault bus by hop distance on the full 123-bus topology; deterministic given the label),
  `persistent.py` (per-event selection from `(event_id, k, mask_seed)`, held across the event's
  40 windows). Neighbour-mean **imputation is not a masking policy** — it lives in
  `scripts/run_imputation_baseline.py`.
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
- **Targeted (worst-case) dropout (DONE, 3 seeds, `20b5b68`):** `run_targeted_dropout_study.py`
  — drops the k μPMUs **topologically nearest the fault bus** (hop distance over the full
  123-bus topology, not the measured-only graph), so selection is a deterministic function of
  the label and there are **no mask seeds**; error bars are over training seeds only. No-fault
  windows have no fault bus and fall back to random selection, so the reported macro-F1 is a
  **conservative bound** on the true worst case. Summary: `build_targeted_summary.py`.
  → `results/dropout_study_targeted/`. Selection logic: `fault_localization/dropout/targeted.py`.
- **Persistent (per-event) dropout (DONE, 3 seeds, `20b5b68`):** `run_persistent_dropout_study.py`
  — draws the dropped set **once per event** and holds it across that event's 40 windows (how a
  real μPMU failure behaves) instead of redrawing per window. Everything else identical, so the
  curve difference isolates mask persistence. Summary: `build_persistent_summary.py`.
  → `results/dropout_study_persistent/`. Logic: `fault_localization/dropout/persistent.py`.
- **Imputation baseline (DONE, 3 seeds):** `run_imputation_baseline.py` — fills each dropped
  node's 3 voltage channels with the per-timestep mean of its *available* 1-hop neighbours and
  sets the mask to 1 (most favourable to imputation), then feeds the **standard-trained**
  RGSAGE-Max. Defaults: `--model rgsage_max --seeds 42 43 44 --k-levels 0 1 2 5 10`
  → `results/dropout_study_impute/imputation_summary.json`. See `IMPUTATION.md`.
- **Significance tests (DONE, no GPU):** `run_significance_tests.py` — paired two-sided
  t-tests over training seeds 42/43/44 (n=3, df=2) with 95% CIs + Holm-Bonferroni per family,
  built *only* from per-seed numbers already on disk; plus a paired **cluster bootstrap** over
  the 6175 test events (not windows — 40 correlated windows per event) at k=0 from
  `predictions_test.csv`. → `results/statistics/significance_tests.{json,csv}` +
  `paper/tables/significance.tex`. Wilcoxon is computed but **not** relied on (n=3 floor = 0.25).
- **Runtime/efficiency (DONE):** `run_runtime_benchmark.py` — inference latency/throughput/
  activation-memory/params measured on synthetic `[B,25,20,6]` batches (the real corpus is
  per-event `.npz`, so a loader-based timing would measure decompression); training
  sec/epoch + total hours + best-validation epoch read from each run's `metrics_by_epoch.csv`.
  → `results/runtime/{runtime_inference,runtime_training}.csv`, `runtime_summary.json`,
  `paper/tables/runtime.tex`.
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
  `n_train_seeds=3`, `n_mask_seeds=3`, `policy=zero_messages`, `err_kind=train_seed_std`.
  Seed-42-only predecessors are kept as `*.bak.csv`, pre-bugfix ones as `*.prebugfix.csv`.
  Result figures → `paper/figures/sdr_*.png` (+ `*.prebugfix.png`); summary
  `paper/figures/sdr_robustness_summary.csv`.
- **Imputation baseline:** `dropout_study_impute/imputation_summary.json` — per-k `f1_mean/std`,
  `per_seed_f1`, `audc_mean`, `ret10_mean`; `policy=neighbour_mean_impute`, seeds 42/43/44.
- **Targeted dropout:** `dropout_study_targeted/` — `targeted_curves_{standard,dropout_aware}.csv`
  (+ `*_per_seed_*.csv`, `targeted_summary.json` which also carries the paired `*_random` arm for
  the comparison, and `proximity_rank.npy` = the precomputed fault-bus→μPMU hop ranking).
  `policy=targeted_nearest_fault`, RGSAGE-Max only.
- **Persistent dropout:** `dropout_study_persistent/` — `persistent_curves_{standard,dropout_aware}.csv`
  (+ per-seed CSVs carrying `f1_per_mask_seed`, `persistent_summary.json` with the `*_random` arm).
  `policy=persistent_per_event`, RGSAGE-Max only.
- **Statistics:** `statistics/significance_tests.json` — every test carries `a_per_seed`/
  `b_per_seed`, `mean_diff`, `ci95_*`, `t_stat`, `p_ttest`, `p_holm`, `cohens_dz`, plus
  `event_level_bootstrap`. **Runtime:** `runtime/runtime_summary.json` (`derived` block holds
  the ratios the paper quotes).
- Per-run metrics.json keys: `test_graph.f1` (soft-voted macro-F1), `test_node_diagnostic.f1`
  (per-window), `test_graph.per_class.{precision,recall}` (class 0 = no-fault),
  `best_validation_graph_f1`.
- Others are historical: `benchmark_v1.1_samplerfix` (tiny, OLD model structure — won't
  load), `fullscale_50_seed42` (pre-frozen depth-1), `diag_*` (depth/stability probes),
  `trend_*` (medium-A).

## Paper (`paper/`)
- `overleaf.tex` — **the live paper** (Elsevier CRC/Procedia, `\documentclass[3p,times,procedia]`),
  ~880 lines, **complete draft**: abstract, intro, related work (+`tab:related`), Materials &
  Methods (dataset +`tab:dataset` +`fig:graph`, baselines, deviations `tab:deviations`,
  dropout-eval protocol, dropout-aware training +`alg:aware`, SDR-STGNN), Setup, Results
  (`tab:baseline`, `tab:degradation`, `fig:degradation`, `tab:aware` incl. the **imputation
  baseline** row, cross-architecture subsection), Discussion (robustness ranking
  + limitations), Conclusion, Acknowledgements, GenAI declaration, bibliography.
  Framing: dropout-aware training is the headline; SDR-STGNN is an ablation.
  The wide tables use `table*` (full-width) to avoid two-column overflow.
- **It does not compile locally** — `ecrc.sty` (Elsevier CRC) is not installed on this machine, so
  page count can only be checked on Overleaf. `preview-local.pdf` is the last exported PDF (13 pp).
- `figures/`, `tables/` — generated assets for Overleaf upload. `figures/*.prebugfix.png` are the
  pre-bugfix versions, kept for comparison only.
- **Page cap = 12** (ICMLDE, per `REVIEW.md`). The deleted `.rules/50-pagelimit.md` said 10; that
  file is gone — 12 is the number to work to.

## Key facts (verified against code — safe to reuse without re-checking)
- Dataset: 50-run corpus `dataset_fullscale_50` = 1.65M **windows** (825×50×40; event-level
  70/15/15, 0 leakage, 40.0 windows/event); 100-run `dataset_fullscale_100` = 3.30M. Exactly
  50% no-fault. Paper's reported 2.5M is NOT reconstructable from 825 grid — documented deviation.
- 6 of 25 fault buses are μPMU nodes → 19 fault locations unmeasured. Graph = 25-node
  24-edge connected tree. One window = `[N=25, S=20, F=6]` (3 RMS volts + 3 mask channels).
- Reproduction accepted at depth-4 (deviation from paper's 1 layer). **Unweighted CE frozen**
  (train.py passes no `weight`; imbalance handled by the balanced corpus — 50% no-fault, 25
  fault classes equal by factorial design — NOT by loss weighting; paper text fixed to match).
- **SDR-STGNN finding (final, 3 training seeds × 3 mask seeds, post-bugfix):** robustness AUDC —
  standard RGSAGE-Max **0.402** → neighbour-mean imputation **0.476** → dropout-aware **0.913**
  → SDR-STGNN **0.892**. Retention at k=10: **5.7% → 11.8% → 80.3% → 75.2%**. The gate MATCHES
  but does NOT beat dropout-aware training (it silences dropped nodes at every hop, blocking
  multi-hop routing through recovered nodes), and imputation gains little over zeroing because
  the reconstructed values still go to a model never trained to distrust them.
  **Report faithfully — do not claim the gate wins.** Source of truth:
  `paper/figures/sdr_robustness_summary.csv` + `results/dropout_study_impute/`.
- **Targeted (worst-case) dropout finding (final, 3 training seeds, RGSAGE-Max):** dropping the
  k μPMUs nearest the fault is **much more damaging where it matters** — standard RGSAGE-Max
  falls to **0.448±0.039 at k=1** vs 0.736±0.013 under random loss (p=0.010), a **3.6× larger**
  drop from clean; 0.361 vs 0.590 at k=2 (p=0.017). From k≥5 the two policies agree (the model
  has collapsed either way). Dropout-aware training **is also penalized** (AUDC 0.913→**0.747**,
  p=0.003; p≤0.014 at every k) — worst-case loss is a **distinct stressor, not one the mitigation
  absorbs** — but it still retains **+0.307 to +0.473** macro-F1 over standard under the same
  targeting (p≤0.008), and from k=2 up its *worst-case* score beats standard's *average-case*
  (0.695 vs 0.590 at k=2, p=0.042; 0.521 vs 0.048 at k=10, p=0.004). At k=1 the two are
  **indistinguishable** (+0.018, p=0.33) — don't claim a win there.
- **Persistent (per-event) dropout finding: a clean NULL.** Holding the dropped set across an
  event's 40 windows changes **nothing** — no point on either curve moves by more than **0.003**
  macro-F1 (standard AUDC 0.402→0.401, dropout-aware 0.913→0.913; Ret@10 5.7%→5.6% and
  80.3%→80.1%), and **no** difference survives Holm across the 12 paired comparisons
  (min p_Holm = 0.46). **Reason:** the decision rule is per-window — soft voting pools the 25
  nodes of *one* window, never the 40 windows of an event, so a window's prediction depends only
  on its own mask, whose marginal distribution is identical under both schemes. **This
  equivalence does NOT transfer to an event-level rule** — per-window resampling would flatter
  that by supplying 40 differently-masked views of a failure that presents only once.
- **Multi-seed (50-run):** SDR-STGNN + all 5 dropout-aware models trained at seeds 42/43/44
  (deviation from the runbook's single-seed; run via a scratchpad orchestrator, autonomous).
- **What is / is not statistically significant at n=3** (`results/statistics/`): the mitigation
  claims ARE — dropout-aware vs standard at k=10 (Δ=+0.626, p_Holm=0.003), Ret.@10 (+74.6 pp),
  AUDC (+0.511), and vs imputation at k=10 (+0.574, p_Holm=0.008); the AUDC gain holds for all
  5 architectures (p_Holm ≤ 0.027). The nulls test as null, which is what the paper claims:
  clean-F1 cost −0.008 (p=0.181) and DA-over-SDR AUDC +0.021 (p=0.098) — i.e. the gate
  statistically MATCHES. **NOT significant:** the accuracy/robustness inversion (RGCN over
  RGSAGE-Max at k=10, Δ=+0.116, p=0.076 — same sign in all 3 seeds, 2.4–6.7×, but the gap size
  varies), the k=0 architecture ranking (best pair p=0.023, nothing survives Holm), and
  imputation over zeroing at k=10 (+0.051, p=0.102). Don't upgrade these to "significant".
- **Cost facts (`results/runtime/`):** dropout-aware = **identical** params (95,258) and latency
  (0.639 vs 0.640 ms @B=1; 6038 windows/s both) → the "no inference cost" claim is now measured,
  not asserted; its only cost is **+2.7% training** sec/epoch. SDR-STGNN's gate adds 0 params but
  **+18% latency** for no robustness gain. Activation memory 145 MiB for ALL models (the shared
  GRU-128 encoder dominates). Best-validation epoch 27.7–28.7 of 30 for RGCN/RGSAGE-*, so those
  models were **still improving** at the budget's end — report accuracies as a floor.
- **Mask channels are static, not all-ones** (verified on `dataset_fullscale_50` test split):
  they encode per-μPMU phase instrumentation — only **14 of 25** nodes are three-phase, the rest
  present patterns like (1,0,0) — and are identical in every window and timestep. Dropout zeroes
  all 6 channels of a dropped node, so its mask becomes (0,0,0), a pattern that node never shows
  in standard training. This is the paper's mechanism for why dropout-aware training works; don't
  restate it as "the mask is always 1".
- **Green-feeder scenario: ❌ CUT (2026-08-04), out of scope.** Not a contribution booster
  (a second-axis stress test only) and would need a new green corpus — dropped before the
  deadline. No green dataset/runs/claims; `configure_green_feeder` stays as dead/future-work
  code. Paper no longer promises it (it survives only as a future-work clause: "cross-topology
  generalization to reconfigured feeders"). The `NEXT-STEPS.md` Step 6, `WRONG-CLAIMS.md`, and
  `GREEN-FEEDER-PLAN.md` that documented this were all deleted — this bullet is now the record.
- Stabilization (grad-clip 1.0 + warmup 3) on EVERY run. GPU = T1000 8GB (NVML monitoring
  broken but training unaffected).

## Hard rules
Identical frozen data/splits across models; change one variable at a time; code passing
tests ≠ validated (confirm with a real run); dropout paper text must match `masking.py`;
**report claims faithfully — SDR-STGNN does not beat dropout-aware training, don't overclaim.**
The 50-run SDR chain runs autonomously to completion; STOP + report before the 100-run sweep.
When editing `overleaf.tex`: **edit, don't restructure** (rule inherited from the deleted
`paper/ISSUES.md`), never invent a number, and re-check any quoted result against the CSV/JSON
on disk. `paper/overleaf.tex` does not compile locally (`ecrc.sty` missing) — page count can
only be checked on Overleaf.
