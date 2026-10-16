# CLAUDE.md — repo map, rules, current state

Read first. Checked against disk **2026-10-16**. Numbers live in `results/RESULTS.md`; this file: rules, map, pointers.

## Project
Three linked deliverables on the IEEE-123 feeder, 25 μPMUs, partial observability:
1. **Paper 1** — adopts Karabulut et al. (2026)'s STGNN fault-localization methodology and reports **our own** numbers,
   plus the sensor-dropout contribution: degradation curves, **dropout-aware training** (the headline mitigation),
   **SDR-STGNN** (availability-gated ablation; matches, does not beat, dropout-aware training). Submitted as **Paper ID
   1170** (ICMLDE 5.0 → Procedia Computer Science); reviews returned.
2. **PC-STGNN** — Phase 2: the Paper-1 backbone with a fixed physics operator fused in. Plan/spec:
   `PC-STGNN/00_INDEX.md` (read order, gate map, five guiding rules).
3. **M.Tech thesis** — `thesis-for-college/`; rules in `thesis-for-college/thesis-rules.md`.

## Current state (2026-10-16)
- **Revision 1170**: decision email 2026-10-10, deadline **2026-10-17**. Work lives in worktree
  `/home/ubuntu/Fault-Localization-rev1170` (branch `revision-1170`, last commit `49fb80c`, 2026-10-11). Plan/log:
  `resubmit/1170_revision_plan.md`, `resubmit/CHANGELOG-1170.md`. Whether the revision was submitted is not recorded
  on disk — ask. Do not do revision work on `main`.
- **PC-STGNN Gate 4** (arms R0ⁿ vs R5, 30 epochs, seeds 42/43/44) is **running** (`scripts/run_s4_4_gate4.py`,
  restarted 2026-10-12 after the 30-epoch amendment). Live: `results/s4_4/gate4_progress.json` (5 of 6 runs scored at
  last check; R5 seed 44 training). Settings: `PC-STGNN/06_STAGE_E_pcstgnn.md` § "S4.4 — Frozen settings" + amendments.
  Quote no Gate 4 number before all 6 runs are scored; never use `results/s4_4_pilot_20ep/`.
- `main` has large uncommitted PC-STGNN work (source, tests, `operator/`, configs). Do not reset, clean, or commit it unless asked.

## Hard rules
`.rules/` was deleted in `f4b7e7b` (2026-09-01); these bullets replace it.
- **Framing:** "adopt the methodology, report our own numbers". No "reproduce"/0.95-matching language. Keep every
  `\cite{Karabulut2026}` credit for the dataset protocol and baselines.
- **Never invent a number.** Re-check any quoted result on disk. Precedence: CSV/JSON > `paper/overleaf.tex` > docs.
- **One variable at a time.** Identical frozen data, splits, epochs, optimiser, checkpoint rule.
- **n = 3 discipline:** paired t-tests over seeds 42/43/44, Holm per family. Never upgrade a trend (list below).
- **Paper edits: edit, don't restructure** (AI-plagiarism risk). Dropout text must match `dropout/masking.py`.
- **Code passing tests ≠ validated** — confirm with a real run. Check `metrics_by_epoch.csv` for the best-val epoch
  before trusting a new model's headline F1.
- **Frozen corpus** `dataset_fullscale_50/` is never modified. Pre-bugfix (`*.prebugfix.{csv,json,png}`) and
  seed-42-only `*.bak.csv` outputs are for comparison only — never mix with current numbers, never delete.
- **Sealed OOD test set** `data/ood_test/` (hash in `data/ood_test.sha256`): do not open without the author's
  go-ahead; record any opening in `PC-STGNN/14_PREREGISTRATION.md` §6 "Sealed-set log". `data/ood_pilot/` is a
  throwaway Gate-2 set — never report its numbers.
- **PC-STGNN wording:** "physics-consistent", never "PINN"; the operator is fixed, never learned or fitted on test
  events (`PC-STGNN/00_INDEX.md`).
- **Stop and ask before:** the 100-run sweep (`run_fullscale100_seed42_sweep.py`, parked), any new GPU run not in an
  approved plan, opening a sealed set, or committing.

### Claims that may not be upgraded (details + sources: `results/RESULTS.md`)
- SDR-STGNN **matches**, does not beat, dropout-aware training (AUDC 0.892 vs 0.913; p=0.098).
- Imputation over zeroing at k=10 (+0.051, p=0.102): **not significant**.
- Accuracy/robustness inversion (RGCN over RGSAGE-Max at k=10, +0.116, p=0.076; same sign in all 3 seeds, 2.4–6.7×):
  **trend only**. k=0 architecture ranking: nothing survives Holm.
- Targeted dropout: dropout-aware is **also penalized** (p≤0.014 at every k) but keeps +0.307 to +0.473 over standard
  under the same targeting (p≤0.008); k=10 worst-case 0.521 vs standard average-case 0.048 (p=0.004); **no win at
  k=1** (+0.018, p=0.33). A conservative bound (no-fault windows fall back to random selection).
- Persistent dropout is a **null** because the decision rule is per-window; this does **not** transfer to an
  event-level rule.
- Noise, contiguous, absolute-AUDC, classical-baseline caveats: `results/RESULTS.md` §8.
- Accuracies are a **floor** (best-val epoch 27.7–28.7 of 30 — still improving).
- FaultQuery seed-42 `f1=0.5691` is an **epoch-1 checkpoint of a diverged run**, not a result; ZoneRefine seed 42
  (test 0.803) has not cleared its gate.

## Commands
- Env: `uv` (Python ≥3.12, torch 2.5.1+cu121). Run `uv run python scripts/<x>.py`. Tests: `uv run pytest tests/`
  (242 tests; `-m "not slow"` skips 2).
- Frozen training flags (CLI, not in the yaml): `--gnn-layers 4 --epochs 30 --grad-clip-norm 1.0 --warmup-epochs 3`.
  Entry point: `fault_localization/experiments/cli.py`.
- Paper preview: `bash scripts/build_paper_preview.sh` (prints page count; vendored `ecrc.sty` + blank logos in
  `paper/texlib/`; preview only — submit from Overleaf). Paper 1 was 17 pp at the last count (2026-08-17); the page
  cap was never verified — confirm with the author before cutting.
- GPU: one T1000 8 GB, no parallel runs. NVML is broken (`nvidia-smi` fails); training is fine.

## Repo map
**`fault_localization/`**
- `config/constants.py` — single source for every dataset/protocol constant (fault/PMU buses, 11 fault types,
  resistances, load 0.5–1.3, windowing, 825 scenarios, 26 classes).
- `models/` — Paper 1: `gru.py`, `gcn.py` (RGCN), `graphsage.py` (RGSAGE mean/max), `gatv2.py`, `sdr_stgnn.py`; shared
  `common.py` (GRU-128 encoder, node head, losses), `voting.py` (soft vote = sum of node softmaxes). All: GRU → GNN →
  head → soft vote; `num_layers` = depth. Phase 2: `pc_stgnn.py`, `physics_loss.py` (P3 loss arm), `pignn.py` (P1/P2
  grafts), `fault_query.py`, `zone_refine.py`.
- `physics/` — fixed, zero-parameter operator (`01_SPEC.md`): `grids.py` (hypothesis axes), `signature_solver.py` +
  `table.py` (build `operator/mu.npy`, `mu0.npy`), `observation.py`, `residual.py` (ρ), `cache.py` (precomputed ρ;
  writes `cache/rtilde_{split}.npy`), `identifiability.py`.
- `dropout/` — `masking.py` (only `zero_messages` is live), `evaluator.py` (one pass over all (k, mask-offset) combos;
  per-batch seed `mask_seed*10000 + batch_idx`), `training.py` (`DropoutAugmentedLoader`), `targeted.py`,
  `persistent.py`, `correlated.py`, `noise.py`, `silent.py` (mask stays 1, value wrong).
- `graph/` — `measured_only.py` (Algorithm 1 tree), `edge_attributes.py`, `fault_distance.py`, `sensitivity.py`.
- `experiments/` — `runner.py`, `cli.py`, `config.py`, `train.py`, `evaluate.py`, `seed.py`, `reporting.py`.
  `data/` — `dataset.py` (`{"x":[N,S,6],"y"}`, normalizes voltage channels 0–2 only), `manifest.py`. `simulation/`
  (OpenDSS), `processing/` (event-level 70/15/15 splits, `verify_no_leakage`), `validation/`, `analysis/`.

**`scripts/`** (~96), grouped by prefix:
- Paper 1 sweeps: `run_fullscale_final_sweep.py` → `results/fullscale_final/`; `run_dropout_aware_sweep.py`,
  `run_sdr_stgnn.py`, `run_dropout_study.py --seeds 42 43 44`.
- Dropout policies: `run_{targeted,persistent,correlated}_dropout_study.py`, `run_noise_robustness_study.py`,
  `run_imputation_baseline.py` (see `IMPUTATION.md`).
- Reviewer arms: `run_classical_baseline.py`, `run_localization_metrics.py`, `run_reviewer_retrains.py`,
  `build_scenario_heldout_split.py`. Stats/cost: `run_significance_tests.py`, `run_new_policy_significance.py`,
  `build_absolute_audc.py`, `run_runtime_benchmark.py`.
- Phase 2: `build_signature_table.py`, `check_operator_*.py`, `run_identifiability_analysis.py`, `run_physics_*`,
  `generate_ood_sets.py`, `run_pcstgnn_*`, `run_s4_4_gate4.py`, `score_s4_4_gate4_run.py`, `run_pignn_screen.py`,
  `run_fault_query.py`, `run_zone_refine.py`.
- Figures/tables: `build_*`. Older diagnostics: `run_trend_sweep.py`, `run_investigate_probes*.py`.

**`config/`** — `experiment_base.yaml` + one yaml per model; `dropout_aware/` (generated); `reviewer/`, `s4_4/` (Gate 4
arms), `pcstgnn_*/` (tuning/diagnostics).

**`results/`** — `fullscale_final/` (definitive 5 baselines × 3 seeds; `benchmark_summary.csv` = Table 3),
`dropout_aware/`, `sdr_stgnn/`, `dropout_study*/` (curves per policy), `classical_baseline/`, `statistics/`, `runtime/`,
`pcstgnn*/`, `s4_4/`, `pignn_*`, `physics_baseline/`. Per-run `metrics.json`: `test_graph.f1` (soft-voted macro-F1),
`test_node_diagnostic.f1`, `best_validation_graph_f1`. Historical (do not quote): `benchmark_v1.1_samplerfix`,
`fullscale_50_seed42`, `diag_*`, `trend_*`.

**Other top level** — `paper/` (`overleaf.tex` = Paper 1 source; submitted version is commit `e9453df`; `f4b7e7b`
later added 74 lines for the scenario side-experiment; `figures/`, `tables/`), `resubmit/` (untracked copy of the
worktree's revision files, identical on 2026-10-16; the worktree is authoritative), `operator/` (physics table),
`dataset_fullscale_{50,100}/`, `dataset_fullscale_50_scenario/` (held-out splits; `processed/` is a symlink to the
frozen corpus), `data/` (OOD sets), `cache/` (ρ cache), `PC-STGNN/`, `thesis-for-college/`, `PPT2/` + `PPT-*.md`
(slide content), `docs/archive/` (stale docs), `fault-loc.tex` (another Paper 1 LaTeX copy from `f4b7e7b`; differs from `overleaf.tex`).

## Docs map
| Doc | Use |
|---|---|
| `results/RESULTS.md` | All Paper 1 headline numbers + §8 reviewer-hardening findings |
| `PC-STGNN/00_INDEX.md` → `01_SPEC.md`, stages `02–09`, `10–14`; `PC-STGNN/REVIEW-2026-09-22.md`, `PC-STGNN-PLAN.md` | Phase 2 spec, gates, metrics, pre-registration; plan review and implementation plan |
| `RETIRED_CLAIMS.md` | Claims killed by measurement — check before reusing an old claim |
| `IMPUTATION.md` | Neighbour-mean imputation baseline (read before touching its script) |
| `FAULTQUERY-ARCHITECTURE.md`, `PIGNN-SURVEY.md` | Phase 2 architecture options |
| `PROBLEM.md`, `QUESTIONS.md`, `README.md` | Problem statement, author Q&A, public readme |
| `REVIEW.md`, `REVIEW-new.md`, `PPT-{DATASETS,METHODOLOGY,REVIEW-PHASE2}.md`, `PPT2/` | College review rubrics; Phase 2 slide content |
| `docs/archive/`: `NEXT.md` (stale handoff), `NEW-STRATEGY.md` (superseded by `PC-STGNN/`), `DONE.md` (author review of the PC-STGNN plan), Review 1 slide docs (`PPT-{GUIDE-REVIEW,SPEAKER-NOTES,INTRO-SLIDES,EVAL-STRATEGY}.md`, `REVIEW1-TALK-SCRIPT.md`) | History only (archived 2026-10-16) |
| Deleted docs (`PROGRESS.md`, runbooks, the old `docs/**` tree, `.rules/`, …) | In git history only — do not cite or resurrect them unless asked |

## Key facts (verified; safe to reuse)
- Corpus `dataset_fullscale_50`: 825 scenarios × 50 runs × 40 windows = 1.65M windows; event-level 70/15/15, 0 leakage;
  exactly 50% no-fault. `dataset_fullscale_100` = 3.30M (its sweep is parked). The base paper's 2.5M is not
  reconstructable from the 825 grid (documented deviation).
- **The split shares all 825 scenarios between train and test** (it splits runs, not scenarios): accuracy measures
  generalization to unseen load conditions of known configurations.
- 6 of 25 fault buses are μPMU nodes (19 fault locations unmeasured). Graph = 25-node, 24-edge tree. Window =
  `[N=25, S=20, F=6]` (3 RMS voltages + 3 mask channels).
- **Mask channels are static, not all-ones:** they encode per-μPMU phase instrumentation (14 of 25 nodes three-phase).
  Dropout zeroes all 6 channels → mask (0,0,0), a pattern never seen in standard training. This is the mechanism
  behind dropout-aware training.
- Frozen config: depth 4 (deviation from the base paper's 1 layer), unweighted CE (balance comes from the corpus, not
  loss weights), grad-clip 1.0 + 3 warm-up epochs on every run.
- Every model is trained at seeds 42/43/44. Mask-diversity bugfix (`6a00c4e`/`798406e`, 2026-08-13): all dropout
  numbers regenerated; shifts in the 2nd–3rd decimal only.
- Trivial always-"no fault" classifier: 26-class macro-F1 **0.0256**. Green-feeder scenario: **cut** (2026-08-04);
  `configure_green_feeder` is dead code; future work only.
- **2026-08-17 reviewer-hardening pass** (noise, contiguous, classical baseline, absolute AUDC, scenario-held-out
  split, "never size from a subset", the named learned-imputation gap): moved to `results/RESULTS.md` §8.
