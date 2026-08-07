# PROJECT_STATUS.md — Fault Localization STGNN (Handoff for Claude Code)

## 1. What this project is
Building a **conference paper** based on Karabulut et al. (2026), *"Robustness of
Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable
Distribution Grids."* Task: reproduce the paper's baselines, then add a **novel
contribution: μPMU sensor-dropout robustness** (degradation curves + dropout-aware
training).

## 2. Domain in one paragraph
Fault **localization** = find WHERE a fault occurs on the IEEE 123-bus distribution
feeder using only 25 sparse μPMU voltage sensors (partial observability). Data is
simulated in OpenDSS via OpenDSSDirect.py. 26 classes = 25 fault locations + 1 no-fault.

## 3. Dataset protocol (from paper)
- IEEE 123-bus feeder, OpenDSS/OpenDSSDirect.py
- 25 fault locations, 25 μPMUs, 11 fault types, 3 fault resistances (0.1/1/10 Ω)
- Load scaling U(0.5,1.3), 1 ms sampling, 20 ms sliding windows (40 per event)
- Feature format: F=6 per node/timestep = [V1,V2,V3 voltages] + [m1,m2,m3 availability mask]
- 70/15/15 event-level split, train-only normalization
- Baselines: GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2 (all spatio-temporal: GRU + GNN)

## 4. Current dataset state
- Running on a TINY diagnostic set: 825 scenarios × 2 runs = 1,650 events / 66,000 windows.
- Full-scale (~2.5M windows) NOT generated yet — DELIBERATE. Scale up only after the
  GNN mechanism is confirmed on the tiny set.

## 5. Journey / what has been DONE
- [x] Fixed broken simulator (was static snapshots → now real 1 ms time-stepped, Daily mode)
- [x] Fixed graph: Algorithm 1 substation-rooted spanning tree, acyclic (nx.is_forest asserted)
- [x] PMU phase types from metadata YAML (not inferred)
- [x] F=6 features + availability mask channel
- [x] Fixed RGATv2 attention (shared W on concat pair, dropout on α coefficients)
- [x] Green feeder config code (open 60–160, close 54–94)
- [x] Sensor-dropout module written (masking, evaluator, dropout-aware training) — NOT run yet
- [x] **GATE 1 PASSED** (validated on tiny set):
      - Binary fault detection: 97.08% acc, 0.97 F1 → signal is learnable
      - Sag physics correct: 0.72/0.30/0.036 p.u. mean sag @ 0.1/1/10 Ω
      - GRU detects faults but CANNOT localize beyond 1-hop of a μPMU (spatial blindness)
- [x] **STEP 1 DONE** (imbalance isolation, 1-seed diagnostic GRU):
      - Inverse-freq Weighted CE preserves detection (Class0 recall 87%, prec 74%)
      - Focal Loss (γ=2) collapses to 100% Class0 recall → REJECTED
      - Fixing imbalance does NOT fix localization → limitation is ARCHITECTURAL
      - **Weighted-CE is FROZEN as the loss for all GNN runs**

## 6. STEP 2 (GNN baseline sweep, tiny set) — DONE, but too close to chance to trust
Goal: train RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2 (seeds 42/43/44, weighted-CE, tiny set).
Question: does GNN localization macro-F1 beat the GRU chance floor (~0.038)?

The in-place-op crash in GraphSAGE (previously blocking rgsage_max/mean and rgcn) was
fixed. All 4 GNNs completed 3 seeds each. Result (results/benchmark_v1.1_samplerfix/):

| model | graph_f1 mean ± std (3 seeds) |
|---|---|
| gru (1 seed) | 0.0417 |
| rgcn | 0.0496 ± 0.0016 |
| rgsage_mean | 0.0549 ± 0.0026 |
| rgsage_max | 0.0584 ± 0.0042 |
| rgatv2 | 0.0589 ± 0.0016 |

Technically "GNN beats GRU floor" — but the margin (0.05–0.06 vs. ~0.038 chance) is too
small to call this a real mechanism confirmation. See docs/MEDIUM-DATASET.md for why this
gate was upgraded to a real trend/ablation test before trusting it.

## 7. MEDIUM-DATASET TREND TEST — DONE, verdict: GO (ablation-confirmed)
Ran per docs/MEDIUM-DATASET.md. Generated medium-A (20 runs/scenario, 16,500 events,
660,000 windows, full 26-class coverage), sharing one checkpointed raw-sim pool with the
tiny set (no duplicate simulation — scripts/generate_trend_dataset.py).

Instead of running the full 5×3 sweep on medium-A (~18h) blind, ran the cheapest decisive
check first: **rgsage_max on medium-A, edges-on vs. edges-off** (empty edge_index,
self-loops only retained — isolates spatial message passing from raw parameter count),
1 seed (config/rgsage_max.yaml unchanged; `--disable-edges` ablation flag added to the
runner/CLI for this).

| | edges ON | edges OFF | ratio |
|---|---|---|---|
| graph_macro_f1 | 0.3426 | 0.1240 | 2.8x |
| perwindow_macro_f1 (node diag) | 0.2265 | 0.0817 | 2.8x |
| f1 @ 0.1Ω | 0.4706 | 0.1521 | 3.1x |
| f1 @ 1Ω | 0.3519 | 0.1320 | 2.7x |
| f1 @ 10Ω | 0.1535 | 0.0489 | 3.1x |
| no-fault precision | 0.9944 | 0.9223 | — |
| no-fault recall | 1.0000 | 1.0000 | — |

The gap holds across all three resistance levels (not just the easy one), and both numbers
are far above the tiny-set's ~0.05 chance floor. **Conclusion: tiny-set flatness was data
starvation, not a structural graph bug — spatial message passing genuinely contributes.**

**Decision: GO**, made on this ablation + the single medium-A data point (0.3426, well
past the doc's 0.15 GO bar). The full medium-A 5-model × 3-seed sweep (~18h) and medium-B
(~37h) were deliberately **not** run — a conscious call to move forward without the full
trend curve / error bars at this scale, to avoid spending that compute now. Revisit later
(scripts are resumable: `uv run python scripts/run_trend_sweep.py --tag medium_A`) if
reviewers want the fuller medium-A table or the third (medium-B) trend point.

Infra: `scripts/generate_trend_dataset.py`, `scripts/run_trend_sweep.py`,
`scripts/build_trend_summary.py` → `results/trend_test/trend_summary.csv` +
`results/trend_test/pivot_graph_f1.csv`.

## 8. FULL-SCALE REPRODUCTION (50 runs/scenario) — DONE, verdict: INVESTIGATE

Generated `dataset_fullscale_50` (825 scenarios × 50 runs = 41,250 events / 1,650,000
windows; extensible prefix design so 51–100 can be appended without regenerating 1–50).
Pre-flight passed: event/window counts exact, all 26 classes, 50.0% no-fault balance,
graph is a valid 25-node forest.

GPU note: `nvidia-smi`/NVML has been broken all along (`Driver/library version mismatch`
— kernel module 535 loaded vs. userspace 580, needs a reboot neither of us has triggered).
This never affected training — confirmed independently via a raw matmul benchmark and via
`torch.cuda.memory_allocated()` during real runs — only live utilization monitoring is
unavailable.

Ran all 5 models, seed 42 only, frozen paper protocol (GRU 11 / GNN 15 epochs, weighted CE,
1 GNN layer, 50 runs) → `results/fullscale_50_seed42/`:

| model | graph_macro_f1 | f1@0.1Ω | f1@1Ω | f1@10Ω | nofault prec/recall |
|---|---|---|---|---|---|
| gru | 0.144 | 0.224 | 0.103 | 0.050 | 0.872 / 1.000 |
| rgcn | 0.426 | 0.625 | 0.431 | 0.147 | 0.964 / 1.000 |
| rgsage_mean | 0.424 | 0.619 | 0.443 | 0.138 | 0.965 / 1.000 |
| rgsage_max | 0.439 | 0.584 | 0.457 | 0.199 | 0.969 / 1.000 |
| rgatv2 | 0.430 | 0.592 | 0.458 | 0.171 | 0.998 / 1.000 |

Best-GNN 0.439 — rising from medium-A's 0.343 but well under the 0.55 INVESTIGATE floor,
and far under the paper's 0.86–0.95. **Gate verdict: INVESTIGATE.** Notable: all 4 GNNs
cluster tightly (0.42–0.44) despite different architectures — pointed away from an
architecture-specific bug and toward a shared ceiling. No-fault recall 1.000 across the
board (detection fully preserved).

## 9. INVESTIGATE diagnostics — ceiling was 1-hop GNN depth, not underfitting

Training-curve review (epoch-by-epoch): train loss still falling at epoch 15 for 3/4 GNNs;
rgsage_max/rgatv2 still rising at their final epoch → underfitting signal, not a plateau.

Ran depth ablations on rgsage_max (single model/seed, no data regen), `--gnn-layers`
override added (RGSAGE only, initially):

| config | graph_macro_f1 |
|---|---|
| depth=1, 15 epochs (baseline) | 0.439 |
| depth=2, 15 epochs | 0.571 |
| depth=3, 15 epochs | 0.751 |
| depth=4, 15 epochs | 0.806 |
| edges-off, depth=1, 15 epochs (ablation @ 50 runs) | 0.253 (confirms graph mechanism still real at scale) |
| **depth=4 + 30 epochs (combined)** | **0.834** |

Monotonic improvement with depth, no over-smoothing observed through depth 4. **Verdict:
the ceiling is GNN receptive-field depth (1-hop is architecturally too shallow for faults
>1 hop from a μPMU under partial observability), not underfitting or a broken graph.**

**Paper-architecture check (important):** fetched the arXiv HTML version
(arxiv.org/html/2604.20403v1) — the paper's Fig. 1(b) and text describe only **1 GNN
layer**. So depth-4 = 0.834 reproduces the *number* using a *different architecture* than
the paper's. Must be stated explicitly in the paper, not glossed over.

**Depth-1-vs-paper follow-up (data-starvation vs. architectural-limit test):**
- depth=1, 30 epochs, 50 runs: plateaus at **0.509** (best val 0.518 @ epoch 29) — more
  training alone does not close the gap to depth-4.
- Generated `dataset_fullscale_100` (825×100 = 82,500 events / 3,300,000 windows) to test
  depth-1 at more data — raw sim is cheap (~0.018s/event measured directly, ~fast wall-clock
  build) and fits disk easily; this was a real, executed build, not just an estimate.
- depth=1, 30 epochs, 100 runs (unstabilized): promising through epoch 9 (val-F1 0.510,
  matching the 50-run run's *final* value in 1/3 the epochs) — then **diverged hard at
  epoch 10** (train loss climbing, val-F1 crashed to 0.03–0.05, never recovered by epoch 30;
  official test f1=0.506 from the epoch-9 checkpoint). **Inconclusive** — cannot separate
  "depth-1 has a real ceiling" from "it broke before it could show whether more data helps."

**Training-instability discovery:** the same collapse pattern also hit the depth-4+30-epoch
run mid-training (val-F1 crashed 0.841→0.251 at epochs 25–26, recovered by 30 since
best-checkpoint selection is unaffected). Added **opt-in** `grad_clip_norm` /
`warmup_epochs` (CLI: `--grad-clip-norm`, `--warmup-epochs`; default off, zero effect on
prior runs) to `train_epoch`/`runner.py`. **Stabilization verified**: `diag_depth4_epochs30_stable`
(grad-clip=1.0, warmup=3 epochs) cleared the full epoch 10–26 danger zone cleanly — official
result **best val-F1 0.855, test macro-F1 0.853**.

Extended `num_layers` support from RGSAGE-only to **RGCN and RGATv2** too (needed since the
locked config below requires depth=4 across all GNN variants) — verified via existing
GATv2 exact-equivalence regression tests (updated for the new per-layer `ModuleList`
structure, all 91 tests pass) plus fresh shape/gradient smoke tests. 5-epoch smoke tests at
depth=4 with stabilization: **RGCN clean from epoch 1** (test f1=0.357 in 5 epochs);
**RGATv2 slow to start during LR warmup** (flat ~0.06 for 2 epochs) **then accelerates
cleanly** (test f1=0.219 in 5 epochs) — no divergence in either.

## 10. DEADLINE MODE — venue locked, scope frozen

Venue: **ICMLDE 2026** (5th Intl. Conf. on Machine Learning and Data Engineering, Dehradun,
India, 27–28 Nov 2026). **Submission deadline 15 Aug 2026** — general ML/data-engineering
venue, no dedicated power-systems track, so the paper framing should lead with the general
robustness-to-missing-sensor-input angle. 3–10 pages, MS-CMT submission portal, own
LaTeX/Word template (exact download link not yet resolved).

**Locked decisions (do not revisit without an explicit new call):**
- Reproduction is ACCEPTED at depth-4 rgsage_max = 0.834. Do not chase the exact 0.86.
- Frozen config for the definitive sweep: GNN depth=4 (GRU has no spatial layer), epochs=30,
  50 runs/scenario, default feeder, unweighted CE, grad-clip-norm=1.0,
  warmup-epochs=3 — applied to **every** run.
- **Permanently cut:** depth-1@100 investigation (including a stabilized rerun — trip-wire
  monitor was needed once already to kill a stray relaunch of it), any further use of the
  100-run dataset, depth-5, any further depth/epoch/scale probes, exact-0.86 chasing.
- The paper's contribution is the sensor-dropout study on the existing models — NOT a new
  architecture. Depth-4 is documented as a reproduction finding/deviation, not sold as a
  novel architecture contribution.

`NEXT-STEPS.md` is now the authoritative execution plan (supersedes the roadmap in §12
below where they conflict): Step 1 (stability gate) done; Step 2 is the seed-42-only
5-model sweep, RGATv2/RGCN scheduled *early* (not last) since they've only been
smoke-tested at 5 epochs, not the full 30; Step 3 is a sanity gate on that baseline; Step 4
is the sensor-dropout study (THE contribution); Step 5 backfills seeds 43/44 later.

**Current state (EXECUTING — see §14 for the live execution log):**
`scripts/run_fullscale_final_sweep.py` matches NEXT-STEPS.md Step 2 (seed 42 only, order
`rgatv2, rgcn, gru, rgsage_mean, rgsage_max`) and `scripts/build_fullscale_final_summary.py`
/ `build_fullscale_final_figures.py` are ready. The already-completed, validated
`diag_depth4_epochs30_stable` run was copied into
`results/fullscale_final/rgsage_max/seed_00042/` so it won't be re-run (saves ~7h). **Sweep
was LAUNCHED 2026-07-30 and is running (§14).**

**Dropout mechanism audited and confirmed (resolved, read-only — no code changed):**
a dropped μPMU stays a node in the graph with voltage (0–2) AND availability-mask (3–5)
channels zeroed (`masking.py::apply_sensor_dropout`, lines 89–101) — pure feature-masking,
not node/edge removal. The `"drop_edges"` policy *computes* a filtered `edge_index`
(lines 103–114) but it is **discarded at both call sites** (`evaluator.py::evaluate_at_dropout`
line 86 and `training.py::DropoutAugmentedLoader.__iter__` line 81 both call/queue
`model(x_dropped)` with no `edge_index` argument, so every model falls back to its full,
undropped `self.edge_index`) — so `"drop_edges"` currently has **zero effect on any
reported result** regardless of which policy string is passed; only `"zero_messages"`
behavior is ever functionally reachable, which also happens to be the default in all three
places (`evaluator.py:37,106`, `training.py:52`).

**Decision: do NOT fix the drop_edges wiring.** Feature-masking (graph intact) is judged
the more realistic sensor-dropout model and is what actually runs correctly — describe (b)
in the paper, note `drop_edges` as an unexercised/future-work code path, not a second
condition that was tested.

Also verified: for a live (non-dropped) μPMU, mask channels are genuinely 1 per wired phase
(`monitors.py::extract_bus_voltages_with_mask`, lines 73–81 — mask starts at zero, set to
1.0 per phase present in `Bus.Nodes()`), confirmed untouched by normalization
(`dataset.py`'s `WindowGraphDataset.__getitem__` only normalizes channels 0–2). So live=1 /
dropped=0 is a real, learnable contrast — the "explicit availability signal" framing in the
paper draft holds.

**Paper drafting (in parallel, ICMLDE-agnostic prose so far):** `paper/01_introduction.md`
and `paper/02_setup_and_methods.md` drafted, pulling directly from `PROBLEM.md` /
`KNOWN_ISSUES.md` framing; citations beyond the base paper and exact final numbers left as
explicit `[CITE]`/`[X]` placeholders rather than invented.

## 11. Reporting requirements (paper-ready output) For every model report, over seeds 42/43/44 (mean ± std): - Per-window macro-F1 AND soft-voted macro-F1 (compare like-for-like across models) - Per-resistance breakdown (0.1 / 1 / 10 Ω) - No-fault precision/recall (confirm detection preserved) ## 12. Roadmap 1. ~~Fix in-place bug → complete GNN sweep on tiny set.~~ DONE. 2. ~~Medium-dataset trend/ablation gate.~~ DONE — GO. 3. ~~Scale up to full dataset (50 runs/scenario) → benchmark against paper 0.86–0.95.~~ DONE — INVESTIGATE → resolved via depth ablation (§9). 4. ~~Stabilize depth-4 training (grad-clip + warmup).~~ DONE — verified (§9). 5. Full seed-42 5-model sweep at frozen depth-4/epochs-30 config (NEXT-STEPS.md Step 2) — staged, awaiting go-ahead. 6. Sensor-dropout study (THE NOVELTY): degradation curves + dropout-aware training (NEXT-STEPS.md Step 4). 7. Backfill seeds 43/44 (NEXT-STEPS.md Step 5). 8. Green feeder experiments — optional, first to cut if time is short. 9. Write the paper (deadline: 15 Aug 2026, ICMLDE 2026). ## 13. Hard rules (do not violate) - All models on identical frozen data/splits. - Report ≥3 seeds with mean ± std (seed 42 first is a deliberate deadline-mode exception — 43/44 backfilled later, not skipped). - Change one variable at a time. - Code passing tests ≠ scientifically valid — always confirm with a real training run. - Same frozen config + stabilization (grad-clip 1.0, warmup 3) on EVERY run, no exceptions. - Dropout mechanism in the paper text must match `masking.py` exactly.
Immediate priority: launch the seed-42, 5-model definitive sweep (`scripts/run_fullscale_final_sweep.py`,
NEXT-STEPS.md Step 2) once given the go-ahead, report the baseline table + no-fault recall
check, then proceed to the sensor-dropout study — the paper's actual contribution — before
the 15 Aug 2026 ICMLDE deadline.

## 14. Deadline-mode execution log (started 2026-07-30)

### Step 1 — verified for reuse (DONE)
Confirmed the reused `rgsage_max`/seed-42 run
(`results/fullscale_final/rgsage_max/seed_00042/`) already logs everything the Step 2
table needs — no re-extraction required: soft-voted graph macro-F1 (`test_graph.f1` =
**0.853**), per-window macro-F1 (`test_node_diagnostic.f1` = 0.723), full per-class
precision/recall (no-fault = class 0: prec 0.999 / recall 1.000), and
`predictions_test.csv` carries `resistance` + `truth` so per-resistance F1 is derived by
`scripts/_metrics.py`. It slots straight into `build_fullscale_final_summary.py`;
`row_for_run` returns `None` for incomplete dirs, so partial runs never corrupt the table.

### Step 2 — baseline sweep LAUNCHED + running
`run_fullscale_final_sweep.py` launched 2026-07-30 in the background (nohup;
log `results/fullscale_final_sweep.log`). Config confirmed at launch: seed 42 only,
depth-4, 30 epochs, grad-clip 1.0 + warmup 3 on every run, order
`rgatv2 → rgcn → gru → rgsage_mean → rgsage_max` (rgsage_max auto-skips as already-done;
rgatv2/rgcn early to surface any depth-4 instability fast). GPU: T1000 8GB (the NVML
warning is the known cosmetic monitoring issue; training unaffected). A stale/interrupted
`rgatv2` seed dir (config only, no metrics) was removed before launch.
- **Timing:** ~25 min/epoch at depth-4 on the 1.65M-window set → ~12.5h/model; realistic
  full-sweep wall-clock **~2 days** (a bit longer than the plan's 1.5-day estimate; 4 models
  to train since rgsage_max is reused).
- **rgatv2 health (as of last check, epoch 22/30):** cleared the epoch 10–26 danger zone
  cleanly — val-F1 climbed 0.06 (ep1, LR warmup) → 0.47 (ep16) → 0.59 (ep21), loss falling
  monotonically. Stabilization holding, no divergence. (Slow warmup start is exactly the
  behavior seen in the 5-epoch smoke test.)
- **STILL PENDING:** rgcn, gru, rgsage_mean must complete; then build the Step 2 table
  (`build_fullscale_final_summary.py` + `_figures.py`), run the Step 3 sanity gate, and
  **STOP for review** before any Step 4 work. Per-model danger-zone curves to be audited on
  completion; flag any run with a loss spike / val-F1 collapse.

### Step 2 parallel task — dropout code de-risked (DONE, PASS)
`scripts/smoke_test_dropout.py` (CPU, tiny data) exercises both Step-4 code paths that had
never been run: `evaluate_dropout_curve()` over k∈{0,1,2,5,10} with the multi-mask-seed loop,
and `DropoutAugmentedLoader` through a real `train_epoch`. Both **PASS** — added an assertion
that masked batches actually zero k full nodes per sample (feature-masking, not pass-through),
grad-clip applied, no NaN. (F1≈0 is expected — untrained model; this is a crash/shape check.)
Step 4 will not crash on first real run.

### Step 4 harness — built + dry-run PASSED (staged, NOT run — gated on Step 2 + Step 3)
Three CPU-only scripts, **no edits to `cli.py`/`runner.py`** (zero risk to the running sweep;
dropout-aware runs use generated config files, not code changes):
- `scripts/run_dropout_study.py` — Step 4a: loops k × 5 models, loads each `checkpoint_best.pt`,
  evaluates with **paired masks** (mask seed `10000·k+offset` is model-independent and the test
  loader is `shuffle=False`, so the SAME k nodes drop for every model at a given (k,offset) — a
  true paired study by construction). Policy locked to `zero_messages` (`drop_edges` is
  CLI-rejected per the audit). Emits `degradation_curves.csv` + overlay figure.
- `scripts/run_dropout_aware_sweep.py` — Step 4b: retrains (seed 42) with `DropoutAugmentedLoader`,
  same frozen config + stabilization, `benchmark_version=dropout_aware`; crash-isolated/resumable.
  Generates `config/dropout_aware/<model>.yaml` (new files, regenerated per run). `--dry-run` does
  CPU/tiny/1-epoch.
- `scripts/build_dropout_comparison.py` — Step 4b report: merges standard vs dropout-aware curves →
  `dropout_aware_comparison.csv` (with `recovery` = Δ) + overlaid figure (standard dashed / aware solid).

Dry-run (CPU, tiny data, fresh 1-epoch current-structure checkpoints for rgsage_max+gru, both
regimes) ran the full pipeline end-to-end: correct table shapes (10 rows = models × k), valid
1350×825 PNGs, and the delivered scaffold's `metrics.json` shows `dropout_aware_training=True`
(masking path confirmed engaged). Note: the OLD `benchmark_v1.1_samplerfix` tiny checkpoints
can't be loaded by the current code (they predate the per-layer `ModuleList` refactor, §9) — the
current `fullscale_final` checkpoints are current-structure and load fine, so 4a works on the
real models.

Figures reuse the palette/style of `build_fullscale_final_figures.py` for cross-paper consistency.

### Step 2 — 50-run sweep COMPLETE (2026-08-01), Step 3 gate = baseline OK ✅
Full sweep done in 1916.7 min (~32 h), 0 failures. Final table (`results/fullscale_final/`,
seed 42, depth-4, 30 epochs, stabilized; summary + 5 figures built):

| model | soft-voted F1 | per-window F1 | f1@0.1Ω | f1@1Ω | f1@10Ω | nofault P/R |
|---|---|---|---|---|---|---|
| rgsage_max | **0.853** | 0.723 | 0.951 | 0.917 | 0.665 | 0.999 / 1.000 |
| rgsage_mean | 0.811 | 0.689 | 0.924 | 0.846 | 0.633 | 0.993 / 1.000 |
| rgcn | 0.729 | 0.619 | 0.904 | 0.807 | 0.460 | 1.000 / 1.000 |
| rgatv2 | 0.610 | 0.537 | 0.702 | 0.666 | 0.409 | 0.991 / 1.000 |
| gru | 0.296 | 0.153 | 0.461 | 0.278 | 0.092 | 0.941 / 1.000 |

- **Step 3 verdict: baseline OK** — no divergence, best GNN sane (0.853), min no-fault recall
  0.9998 (detection preserved). rgatv2's ep29–30 spike was absorbed by best-checkpoint.
- Findings: clean ordering rgsage_max > rgsage_mean > rgcn > rgatv2 ≫ gru; per-resistance
  monotonic (10Ω hardest, backs the high-impedance narrative); **rgatv2 underperforms** vs the
  base paper (where it was top) — likely depth-4 attention instability, worth a paper sentence.
- **STOPPED for review per plan.** GPU now FREE. Did NOT auto-start dropout study or 100-run.
- Next: user will launch 100-run in a fresh session via `RUN-100RUN-SWEEP.md`; contribution
  (Step 4 dropout study) still gated/parked.

### Full-scale (100-run / 3.30M) decision — user wants it; PARKED, not launched
User asked for full-scale training. Confirmed 3.30M = `dataset_fullscale_100` (already on disk),
vs our 50-run 1.65M. Architecture is IDENTICAL (same models/depth-4/epochs/stabilization — only
DATA_ROOT + benchmark_version differ). Chosen path: **finish 50-run first (fallback), THEN launch
100-run.** `scripts/run_fullscale100_seed42_sweep.py` staged (all 5 models train fresh — no reuse;
→ `results/fullscale_final_100/`). NOT auto-chained — launch manually when 50-run done. Cost:
~4–5 days for seed-42 baseline (2× data). Timeline risk vs Aug 15 flagged to user.

### Paper writing (paper/overleaf.tex) — sections filled + verified against code
- **Related Work table** fixed: added `\usepackage{booktabs}` + `\yes/\no/\pt` defs, corrected
  Nguyen row (2022, 1D-CNN+GCN not RNN), added all 8 missing `\bibitem`s (web-verified). No
  undefined citations remain (13 total).
- **Grid Simulation & Dataset** written (all claims code-verified) + auto-generated dataset table
  (`scripts/build_dataset_assets.py` → `paper/tables/dataset_info.tex`) + 2 figures from REAL data
  (`paper/figures/measured_only_graph.png`, `fault_signature.png`; deepest-sag 0.83/0.22/0.014 p.u.
  @ 0.1/1/10 Ω). Dataset-scale numbers (50-run/1.65M) live ONLY at overleaf.tex lines ~300/317/321
  + the table — regenerate table via `build_dataset_assets.py --data-root dataset_fullscale_100
  --runs-per-scenario 100` if switching to 100%.
- **Baseline Architectures** written from the model code (GRU-128 shared temporal encoder; RGCN
  sym-norm; RGSAGE concat mean/max; RGATv2 4-head shared-W-on-pair; BN→ReLU→drop0.35→Linear-26
  head; soft-vote = sum node probs; weighted-CE; depth-4 flagged as the only architectural
  deviation). Now cites Hamilton2017 + Brody2022 (previously uncited).
- Still empty in overleaf.tex: dropout-eval + dropout-aware methods, experimental setup, all
  results subsections, discussion, conclusion.

### Git — repo published
Initialized + pushed to GitHub (private): `PK-SANGAMESWAR/Robustness-...-Fault-Localization`.
Comprehensive `.gitignore` (excludes .venv, all dataset*/ dirs, *.pt, predictions_test.csv,
manifests, BasePaper.pdf); source + paper + docs + small result artifacts (figures/metrics/
summaries) tracked. Merged GitHub's MIT LICENSE, wrote real README.md. Pushed via VS Code
(sandbox has no git creds). `.git` pruned 2.6G→5M.

### Orientation
`CLAUDE.md` now holds a full repo map (file locations + roles) — read it first each session to
avoid re-exploring.