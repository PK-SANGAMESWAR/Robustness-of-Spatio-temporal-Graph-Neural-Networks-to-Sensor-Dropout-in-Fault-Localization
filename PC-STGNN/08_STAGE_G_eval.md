# Stage G — Full Evaluation and Robustness (Weeks 15–19)

**Purpose.** Produce every results table, and answer the leakage objection with an experiment.
**Compute.** Mostly inference — minutes per model per condition. Plus ~1.5 h CPU and one ≈ 7 GPU-h
retrain.

## S6.1 — Master sweep

Every model × every condition. Shared seeds throughout — mask seeds, corruption seeds, noise
seeds identical across models.

| Condition | Settings | Existing results to reuse |
|---|---|---|
| Random dropout | k ∈ {0,1,2,5,10} | `results/dropout_study{,_aware,_sdr}/` |
| Persistent | per-event, same k for all 40 windows | `results/dropout_study_persistent/` (null for standard and DA) |
| Structured | contiguous connected subgraph | `results/dropout_study_correlated/` (milder than scattered; incidence confound noted) |
| Targeted | k nearest the fault | `results/dropout_study_targeted/` |
| Silent failures | 4 types × s ∈ {1,2,3} | new |
| Measurement noise | σ ∈ {5e-4, 2e-3, 1e-2} p.u., σ₀ set to match in the residual | `results/dropout_study_noise/` (standard + DA) |
| OOD resistance | {3,5,20,30} Ω × k ∈ {0,5,10} — open the sealed set here, once | new |

**Models.** B1, B1ᵃ (matched filter), B2 (PSM), logreg/RF, both impute-then-infer rows, all 5
baselines standard, all 5 dropout-aware, SDR-STGNN, R1–R7, B2ᵃ, G0–G3. `[Corrected 2026-09-22]`
Only the new models need the existing conditions re-run; the Paper-1 numbers are reused as-is
(same mask seeds).

**Check.** Before opening the sealed OOD set, verify its hash against `data/ood_test.sha256`.
Record the date it was opened in `14_PREREGISTRATION.md`.

**Tables produced.** T5 (all models × k) · T6 (baseline ladder) · T8 (OOD) · T9 (silent failures)
· T10 (structured + noise).
**Figures produced.** F-G1 (F1 vs k, all models) · F-G2 (F1 by resistance) · F-G3 (OOD
extrapolation — the headline) · F-G4 (noise robustness) · F-G5 (structured vs random vs
persistent dropout).

## S6.2 — Model-mismatch study — answers the leakage objection with evidence

The sharpest criticism you will face is not "you used test events" — `12_CHECKLISTS.md` handles
that. It is:

> The table comes from the same simulator that generated the data. PC-STGNN therefore has the
> exact generative model of its own test set. A real utility has an approximate feeder model.
> This will not transfer.

No paragraph answers that. An experiment does. On this corpus the objection is sharper still:
the load is a scalar and the data are noise-free, so the nominal table is nearly an exact
generative model (see Stage C priors).

**Do.** Rebuild the signature table from a perturbed feeder model. Leave the corpus and the trained
models untouched.

| Perturbation | Levels |
|---|---|
| Line impedances, random per segment | ±2 %, ±5 %, ±10 % |
| Load composition (constant-P vs a ZIP mix) | one alternative |
| One unmodelled lateral, or a small unmodelled DER | one scenario |

**Check.** Gain over R0 as a function of mismatch level, for PSM (B2), fusion-only (S2.6 on R0)
and R5.

**Cost.** ≈ 30 min of solves per perturbed table (70 k solves). The fusion-only variant needs no
retraining. One retrain of R5 at ±5 % confirms the trained variant behaves the same way
(≈ 7 GPU-h).

**Either outcome is publishable.** Graceful degradation → "the operator tolerates realistic
feeder-model error", exactly what a power-systems reviewer wants to read. Collapse at ±2 % → a
real limitation, found in a week rather than after submission.

**Table produced.** T11 — model-mismatch robustness.
**Figure produced.** F-G6 — ΔF1 vs perturbation level.

## S6.3 — Statistics

**Do.**
- Mean ± std over the 3 seeds for everything that has 3 seeds; 1-seed rows are marked as such.
- Paired bootstrap over test **events** (not windows — 40 correlated windows per event) for the
  pre-registered comparisons only: R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2. Reuse
  `scripts/run_significance_tests.py` (paired t over seeds + Holm + event-level cluster bootstrap).
- Report 95 % CIs, not just p-values.

**Interpretation.** With 6,175 test events the bootstrap resolves small differences easily. Seed
variance, not significance, is the binding constraint. Report both and lead with the seed spread.

**Table produced.** T13 — statistical tests summary.

## S6.4 — Computational cost accounting

**Do.** For every model record: parameter count, FLOPs per window, training GPU-hours, inference
latency per window (including the residual and, for G2, the LOO pass and second forward). Extend
`scripts/run_runtime_benchmark.py` (already measures the Paper-1 models on synthetic
`[B,25,20,6]` batches).

**Why.** Reviewers in this area ask. It also lets you state plainly that the physics path adds
three scalars and ≈ 1.3× forward cost for gate variants only.

**Table produced.** T12 — computational cost.
