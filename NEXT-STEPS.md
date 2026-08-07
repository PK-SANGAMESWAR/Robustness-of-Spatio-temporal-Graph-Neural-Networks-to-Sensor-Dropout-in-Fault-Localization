# NEXT_STEPS.md — Deadline-Mode Execution Plan (target: Aug 15)

## Context / locked decisions (do NOT revisit)
- Reproduction is DONE. Stability fix (grad-clip 1.0 + LR warmup 3) is VERIFIED:
  diag_depth4_epochs30_stable (rgsage_max, seed 42) cleared the epoch 10–26 danger
  zone cleanly, val-F1 ~0.855. The fix works.
- Frozen config for ALL runs: GNN depth = 4, epochs = 30, 50 runs/scenario (~1.65M
  windows — do not echo the paper's "~2.5M windows" figure, that was the paper's
  full 100-run corpus, not ours), default feeder, unweighted CE,
  grad-clip-norm = 1.0, warmup-epochs = 3.
- CUT permanently (never run): depth-1@100 investigation, 100-run dataset, depth-5,
  any further depth/epoch/scale probes, exact-0.86 chasing.
- **Measurement noise: DROPPED entirely.** PROBLEM.md §5 mentions it as optional;
  no noise-injection code exists and none will be added. Not in scope for this paper.
- **Green feeder: CUT (decision 2026-08-04).** Dropped from scope entirely. It is a
  stress test on a second robustness axis (topology change), not an accuracy/contribution
  booster for the conference paper — the base paper's own STGNNs merely *degrade* under
  reconfiguration. It does not strengthen the headline dropout-aware-training story enough
  to justify the extra green-dataset simulation before the deadline. Do NOT run it; do NOT
  claim it. The abstract/intro no longer promise it and the empty Results subsection has
  been removed from `paper/overleaf.tex`. Step 6 below is VOID.
- We are shipping a SCOPED paper. The contribution is the SENSOR-DROPOUT STUDY applied
  to the existing models — NOT a brand-new network architecture.

---

## STEP 1 — Stability run ✅ DONE
- Completed 30 epochs cleanly, danger zone (10–26) held, test macro-F1 = 0.853
  (precision 0.875, recall 0.847). Stabilization confirmed.
- REUSE this run as the rgsage_max / seed-42 entry in Step 2 — do NOT re-run it.
  First verify it logged the same metrics Step 2 expects (per-resistance F1,
  no-fault precision/recall, soft-voted + per-window F1); re-extract from checkpoint
  if any are missing.

## STEP 2 — Baseline sweep, SEED 42 ONLY (fast baseline)
Launch the full-scale baseline sweep, but ONE SEED first (not 3), to get a complete
baseline table in ~1.5 days instead of ~4–5 days.

- models: gru, rgcn, rgsage_mean, rgsage_max, rgatv2
- seed: 42 ONLY
- config: depth=4 (all GNNs), epochs=30, 50 runs, default feeder, unweighted CE
- stabilization: grad-clip-norm=1.0 + warmup-epochs=3 MUST be applied to EVERY run
  (verify these are in the sweep script/config, not just a CLI probe)
- resumable + crash-isolated: one model diverging must not kill the others
- ORDERING: do NOT schedule rgatv2 last — put it early so any depth-4 attention
  instability surfaces early, not after 30h. (RGATv2 + RGCN at depth-4 for 30 epochs
  were only smoke-tested at 5 epochs, so they have not passed the epoch-25 danger zone.)

Output → results/fullscale_final/ (benchmark_summary.csv + comparison.csv + figures).

### STEP 2 parallel task (CPU, while sweep runs)
Smoke-test the never-run dropout code on ONE model + tiny data:
- evaluate_dropout_curve() over k∈{0,1,2,5,10} runs without crashing
- DropoutAugmentedLoader masks nodes during training without crashing
Fix any bugs NOW so Step 4 doesn't crash on first real run. Report pass/fail.
This de-risks the contribution while the GPU is busy — no time cost.

### STEP 2 report + STOP
Report the 5-model seed-42 baseline table:
- soft-voted graph macro-F1, per-window macro-F1
- per-resistance macro-F1 (0.1 / 1 / 10 Ω)
- no-fault precision + recall
Then STOP. Do NOT auto-start the dropout study. Flag ANY run whose train loss spiked
or val-F1 collapsed (possible divergence) so we can catch a corrupted row.

## STEP 3 — Sanity-check the baseline (gate before building on it)
Before any contribution work, confirm:
- all 5 models trained to completion, no divergence
- best GNN macro-F1 is in a sane range (roughly the ~0.83 ballpark we saw)
- no-fault recall stayed high (detection preserved)
If a model diverged or a number looks broken, FIX that first — do not build the dropout
study on a broken baseline. Report the verdict: baseline OK / needs-fix.

## STEP 4 — Sensor-dropout study (THE CONTRIBUTION)
Run ONLY after Step 3 confirms the baseline is clean. Uses existing models — no new
architecture. Two parts:

### 4a. Inference-time dropout degradation curves
- For each of the 5 trained (seed-42) models, evaluate with k μPMUs dropped at inference,
  k ∈ {0, 1, 2, 5, 10} of 25.
- MECHANISM RESOLVED (audited): dropout = feature masking (option ii). Dropped nodes
  stay in the graph; voltage channels [0:3] AND availability-mask channels [3:6] are
  zeroed. Live μPMUs carry mask=1 per present phase (verified in monitors.py), giving
  an explicit "unavailable" signal. Edges are NOT removed; message passing uses the
  full graph. Default policy "zero_messages". The drop_edges path exists but is NOT
  wired through — do NOT use it. Paper text already matches this; no re-audit needed.
- Use deterministic mask seeds so the SAME k nodes are dropped for every model at a
  given (k, mask-seed) — paired comparison.
- Report macro-F1 vs k for each model → degradation curves.
- Output → results/dropout_study/degradation_curves.csv + figure.

### 4b. Dropout-aware training
- Retrain the models (seed 42) with random node masking DURING training
  (variable number of nodes masked per batch), same frozen config + stabilization.
- Re-run the 4a dropout evaluation on these dropout-aware models.
- Report: how much accuracy at each k does dropout-aware training recover vs standard
  training? → overlaid degradation curves (standard vs dropout-aware).
- Output → results/dropout_study/dropout_aware_comparison.csv + figure.

### STEP 4 report + STOP
Report both figures + a summary table (k × model × standard-vs-dropout-aware macro-F1).
Then STOP for review before any further scope (green feeder, extra seeds).

## STEP 5 — Backfill seeds 43 + 44 (rigor, do LAST / in background)
Only after the contribution results exist:
- Re-run STEP 2 baseline sweep for seeds 43 and 44 (same frozen config).
- If time permits, also run 4a/4b for seeds 43/44.
- Recompute all reported numbers as mean ± std over the 3 seeds.
- This adds error bars for the paper; it is NOT a blocker for earlier steps.

## STEP 6 — Green feeder dropout study — ❌ CUT (2026-08-04), VOID
Removed from scope. It was a second-axis robustness stress test (topology change), not a
contribution booster, and needed a new green simulation corpus we won't spend before the
deadline. No green dataset, no green runs, no green claims. `configure_green_feeder` stays
in the code as dead/future-work only. Nothing to do here.

---

## Hard rules
- Same frozen config + stabilization (grad-clip 1.0, warmup 3) on EVERY run.
- All models on identical data/splits; dropout mask seeds paired across models.
- STOP and report after Steps 1, 2, 4 — do not chain into the next phase automatically.
- Contribution = dropout study on existing models. Do NOT build a new architecture.
- No 100-run, no depth-5, no exact-0.86 chasing.
- Dropout mechanism in the paper text must match masking.py exactly.