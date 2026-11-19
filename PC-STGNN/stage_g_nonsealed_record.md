# Stage G: results on the non-sealed splits

## Record `[Recorded 2026-11-18]`

Source: `results/stage_g/analysis/summary.json` (written 2026-11-17 by `scripts/summarize_stage_g.py`) and `results/stage_g/queue_progress.json`.
The queue ran 46 steps and ended 2026-11-16T14:52:38Z. Step 46 (silent failures on the test split) was skipped: G-TF2 on the test split is outside its
pre-registration. **The sealed OOD set `data/ood_test/` was not opened by this queue.** The sealed-set log row in `14_PREREGISTRATION.md` § 6 is still empty.
Seeds 42/43/44, test split (247,000 windows per scored cell). n = 3: signs are reported as signs; p values are paired t over seeds.

- **R5 vs R0ⁿ, noise grid (k ∈ {0,1,2,5,10} × sigma ∈ {0, 5e-4, 2e-3, 1e-2}: 20 cells).** R5 is higher in all 3 seeds in all 20 cells.
  k=0, sigma=0: 0.894 vs 0.833 (raw p 0.013). k=10, sigma=0: 0.851 vs 0.647 (raw p 0.0011; Holm over the post hoc 7-cell family 0.0077).
  Not significant by raw p (> 0.05), same sign in all seeds: k=0, 1, 2 at sigma=1e-2 (p 0.121, 0.102, 0.086).
- **R5 vs R0ⁿ, loss policies at k=10.** Targeted 0.741 vs 0.496 (p 0.0032). Persistent 0.851 vs 0.646 (p 0.0011). Contiguous 0.838 vs 0.648 (p 0.0014).
  All 3 seeds positive in every policy and every k of {1,2,5,10}. With the 20 noise cells this gives 32 conditions, all sign-consistent.
- **R2 (learned decoder) vs R5.** Below R5 in all 3 seeds in all 20 cells (k=0, sigma=0: 0.851 vs 0.894).
- **R6ⁱ (shuffled table, inference only) vs R5.** Far below R5 in all 3 seeds in all 8 cells (k=0, sigma=0: 0.079 vs 0.894; k=10: 0.063 vs 0.851).
- **Model mismatch, validation split (R5 above R0ⁿ in N of 21 seed-by-cell comparisons).** Table +1 tap: 15 of 21, not graceful. Table −1 tap: 18 of 21, not graceful.
  Impedance ±5 %: 21 of 21, graceful. Retrain of R5 on the impedance-perturbed table (seed 42 only, validation): 0.894 at k=0, sigma=0, vs 0.888 for the nominal weights on that table.
- **B2a (seed 42, validation).** 0.709 at k=0, sigma=0, below R0ⁿ (0.846) and R5 (0.895).
- **Not done here:** H3 and Gate 6 (need the sealed set), the frozen model list, the grounded/ungrounded split result, and any test-split result for G-TF2.

## Tables and figures built from these results `[Recorded 2026-11-18]`

Code: `fault_localization/stage_g/tables.py`, `cost.py`; scripts `build_stage_g_tables.py`, `build_stage_g_t12.py`, `build_stage_g_figures.py`;
tests `tests/test_stage_g_tables.py`, `tests/test_stage_g_cost.py`. Output: `results/stage_g/tables/` (T5, T7, T10, T11, T12, T13, `manifest.json` with the
sha256 of every input) and `results/stage_g/figures/` (F-G1, F-G4, F-G5, F-G6 with `figures_data.json`). No sealed file was read.

- **Metric-consistency check (12_CHECKLISTS § 0).** The training-loop validation F1 of R0ⁿ equals the scorer's k=0, σ=0 value for all 3 seeds (|Δ| < 1e-9). The
  Paper-1 R0 row rebuilt by `tables.py` gives k=0 0.839 and AUDC 0.913, as `results/RESULTS.md` § 3.
- **Evaluator note for T5.** R0 rows come from the Paper-1 evaluator. At k > 0 its mask stream differs from the Stage G scorer (same distribution). The table says so.
- **B2ᵃ is an epoch-1 checkpoint of an unstable run.** `results/stage_g/train/b2a/.../metrics_by_epoch.csv`: best validation F1 0.709 at epoch 1; epochs 2–30 vary between
  0.46 and 0.67; train loss about 100 against 2.6 for R5 (the R̂ term dominates). The B2ᵃ rows (T5, T7, F-G1) are labelled "epoch-1 checkpoint". They do not show how well
  a regressed R̂ can work. What "collapses on OOD R" means for A4 is still open (14, Amendment 5), so A4 is not decided here.
- **T12 (cost), CPU part.** R5 has 1 more parameter than R0ⁿ (95,259 vs 95,258). Backbone 53.6 MFLOPs per window; the residual adds 26.3 MFLOPs (counted from
  `physics/residual.py`, 4 × table entries). Training time (sum of `train_seconds`): R0ⁿ 7.17 h, R5 21.51 h. Latency is **not measured**: the sealed run is using the GPU
  and the CPU, and `12_CHECKLISTS.md` § 0 forbids timing on a busy machine. `build_stage_g_t12.py --latency` refuses to run while the machine is not quiet.
- **Figure colour rule.** One colour per model in all figures. F-G6 plots tables, not models, so it uses other colours and marker shapes.
- **T13 (test split, event-level paired cluster bootstrap, 10,000 resamples, `RandomState(20261102)`; `results/stage_g/tables/T13_statistics_test.md`).**
  Sign 3/3 in all 24 cells (8 for each of R5 vs R0ⁿ, R5 vs R6ⁱ, R5 vs R2) and every CI excludes zero, so all cells meet the pre-registered "supported" rule. Seed-level
  evidence is weaker: raw paired-t p > 0.05 at H4 σ=1e-2 (ΔF1 +0.028, p 0.121) and A2 σ=1e-2 (+0.008, p 0.081); Holm over the post hoc families {H1–H4} (8 cells) leaves H1
  (10 Ω, +0.099, Holm p 0.055), H2 σ=1e-2 (0.055) and three H4 cells above 0.05. The bootstrap CI resamples test events with the 3 models fixed, so it does not include
  training-seed variance. Those cells are trends. **H6 (control):** R5 − R0ⁿ = +0.061 at k=0 (R0ⁿ seed-std 0.010): outside "within 1 seed-std", because R5 is higher. R5 does not lose
  clean accuracy; the control as worded is not met. A1 (R5 vs R6ⁱ): ΔF1 +0.52 to +0.81. A2 (R5 vs R2): +0.008 to +0.211.
- **S7.1 error attribution (test split, σ=0, 3 seeds × 3 offsets; `scripts/run_error_attribution.py`, `results/stage_g/analysis/error_attribution.json`, F-H1).**
  Errors are mapped onto the identifiability matrix `D_global_k{k}.npy` (`results/pcstgnn/audit/s2_1_identifiability`; D < 1 = indistinguishable; w = 20 windows excluded).
  At k=0, 5 location pairs have D < 1 (1.5 % of ordered pairs). **47–48 % of R0ⁿ's errors, 52–54 % of R5's and 50–54 % of R2's fall in those 5 pairs** (every seed). At k=10 (6 pairs, 1.9 %):
  R0ⁿ 17–19 %, R5 34–35 %, R2 17–18 %. The pairs at k=0 are the bus pairs (25,29), (47,51), (62,65), (89,93), (97,99) in `FAULT_BUSES` terms, class pairs (5,6), (9,10), (14,15), (20,21), (22,23).
  The target sentence of S7.1 ("residual errors concentrate in pairs that are physically indistinguishable") holds for about half of the errors at k=0, not for all. Reading: R5 makes fewer
  errors (error rate 0.042 vs 0.069–0.079 at k=0) and what is left sits more in the D < 1 pairs. For R5 at k=0, windows whose true class is in a D < 1 pair are wrong 13 % of the time, other windows 2 %.
  Limits: D is an average over random sensor subsets with the nominal table and no measurement noise; the test masks are different random subsets of the same distribution; one operating point (σ=0).
