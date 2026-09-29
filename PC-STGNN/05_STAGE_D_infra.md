# Stage D — Infrastructure and Stronger Baselines (Weeks 5–9, parallel)

**Purpose.** Build every evaluation condition and close the baseline gaps a reviewer would raise
even about Paper 1. Start this during Stage C — it is needed under every Gate-2 branch.
**Compute.** `[Corrected 2026-09-22]` CPU only. The 12 dropout-aware baseline runs already exist.

## S3.1 — Failure injectors

**Do.** One data-loader hook extending the Paper-1 masking. No new simulation for any of it.

| Injector | Settings | Status |
|---|---|---|
| Random dropout | k ∈ {0,1,2,5,10} (Paper 1 protocol) | exists — `dropout/masking.py` |
| Persistent | the same k sensors missing for all 40 windows of an event | exists — `dropout/persistent.py`; Paper 1 found it a null (per-window decision rule) |
| Structured | contiguous connected subgraph of the measured-only tree | exists — `dropout/correlated.py`; Paper 1 found it *milder* than scattered, with a known per-node incidence confound |
| Targeted | k µPMUs nearest the fault bus | exists — `dropout/targeted.py` |
| Measurement noise | σ ∈ {5e-4, 2e-3, 1e-2} p.u., mask-gated, applied in normalised space | exists — `dropout/noise.py` (keep this grid so the standard/DA curves need no re-run) |
| **Silent (m = 1, value wrong)** | stuck · bias ±2 %, ±5 % · Gaussian σ = 0.02 p.u. · swapped phase channels — on s ∈ {1,2,3} sensors | **new** — `dropout/silent.py` |

**Definition of "stuck"** `[Corrected 2026-09-22]`: freeze the sensor at its value in the
**first timestep of the window** (input-only; no event metadata). For windows 21–39 that is the
pre-fault level, which is the realistic failure — a sensor that stopped updating before the fault.

**Check.** Unit tests:
1. With a fixed seed the injector is deterministic and reproducible across models.
2. k = 0 with no corruption returns the batch byte-identical to the un-injected tensor.
3. Silent injectors leave m = 1 (verify explicitly — this is the whole point).
4. Silent injectors never touch an uninstrumented phase (m = 0 stays 0, V stays 0).

**Pass.** All four tests green.

**Note.** Stuck-at-pre-fault is the most realistic silent failure. Weight the discussion toward it.

**Note on noise.** The corpus is noise-free, which makes a 0.01–0.02 p.u. sag at 10 Ω trivially
visible. Paper 1 already measured the cost: standard RGSAGE-Max 0.846 → 0.786 / 0.579 / 0.345 at
σ = 5e-4 / 2e-3 / 1e-2, and dropout-aware training does not help. σ = 0.01 is the IEEE C37.118
1 % TVE cap. This is the most likely location for a physics gain — and σ₀ must be set in the
residual denominator to the injected level.

## S3.2 — Dropout-aware baselines

`[Corrected 2026-09-22]` **DONE.** All five architectures at seeds 42/43/44 are in
`results/dropout_aware/`, with 3-seed curves in `results/dropout_study_aware/`. Each beats its
standard counterpart on AUDC (p_Holm ≤ 0.027, `results/statistics/`). Nothing to run; cite.

## S3.3 — Impute-then-infer baselines

**Do.** Two rows:
1. Neighbour-mean imputation — **exists** (`scripts/run_imputation_baseline.py`, AUDC 0.476,
   Ret@10 11.8 %).
2. Physics imputation — fill dropped sensors with μ at the PSM-best hypothesis, then run the
   unmodified standard-trained RGSAGE-Max. New, no training.

**Check.** Full sweep, same mask seeds.

**Why.** Closes the "impute then infer" family from the literature (physics-guided ST-GNN
measurement recovery) and shows whether a *physically correct* fill beats a graph-mean fill when
the downstream model was never trained to distrust either.

`[Corrected 2026-09-29]` **Row 2 DONE.** `scripts/run_physics_imputation_baseline.py` — fills each
dropped node with μ at `argmin_f rho_f(A)`'s profiled `(tau,R,L)`, normalised back through the
loader's stored mean/std and broadcast across the window's 20 timesteps (the same quasi-static
approximation `physics/observation.py` already makes). Same `mask_seed = 10000*k + offset` /
per-batch-seed scheme as row 1, so the two rows are paired on identical dropped sets. Full sweep,
3 training seeds, standard-trained RGSAGE-Max, k in {0,1,2,5,10}:

| k | physics-impute F1 | neighbour-mean F1 |
|---|---|---|
| 0 | 0.846 ± 0.007 | 0.846 ± 0.007 (identical — k=0 is a no-op in both) |
| 1 | 0.775 ± 0.020 | 0.739 ± 0.017 |
| 2 | 0.701 ± 0.044 | 0.637 ± 0.028 |
| 5 | 0.479 ± 0.100 | 0.336 ± 0.043 |
| 10 | 0.225 ± 0.114 | 0.099 ± 0.033 |

AUDC (relative) 0.601 ± 0.098 vs 0.476 ± 0.042; AUDC (absolute) 0.508 ± 0.079 vs 0.403 ± 0.032;
Ret@10 26.6 % ± 13.8 % vs 11.8 % ± 4.0 %. The physically-consistent fill beats the graph-mean fill
at every k > 0 and more than doubles Ret@10, at the cost of much larger seed variance at high k (a
wrong PSM hypothesis under heavy dropout can pick a badly wrong fill, unlike the graph-mean's
bounded worst case). `results/dropout_study_impute/physics_imputation_summary.json`.

`[Corrected 2026-09-29]` **Batch-size fairness bug found and fixed post-hoc.** The first run above
used `--batch-size 256` (for gram-method GPU throughput); the neighbour-mean baseline's documented
command uses the default 32. Since the mask seed is `mask_seed*10000 + batch_idx`, this shifted
which windows landed in which batch, so the "paired on identical dropped sets" claim above was
**not actually true beyond the first 32-window batch** — verified directly: 64/100 of the first
100 windows got a *different* dropped node between the two batch sizes. Rerunning at
`--batch-size 32` (matching the baseline) and spot-checking two independent batch-size-32
dataloaders under the shared seed formula confirms 0/128 mismatches — the fix holds. The corrected
numbers are the ones in the table above; the delta from the flawed bsz=256 run was in the 4th
decimal at every k (e.g. k=10: 0.2249 vs 0.2257; AUDC absolute unchanged at 0.508), so the
aggregate comparison itself was never in doubt — only the literal per-window pairing claim was
wrong. Flawed run preserved for the record at `/tmp/physics_imputation_summary.bsz256.json`
(not in the repo).

## S3.4 — Residual computation layer

**Do.** Implement r̃(A), r̃(D), r̃ᵢ^LOO as batched tensor ops: broadcast the
[26, 11, 15, 17, 75] μ tensor against [B, 75] observations (in p.u. — de-normalise with the
loader's stored mean/std, `data/dataset.py:30–31`), mask, sum, min over the (τ, R, L) axes →
[B, 26]. LOO adds one pass: [B, 25 sensors] by removing each sensor's term before the min.

**Check.**
1. Correctness against a slow reference loop on 100 samples — exact to float tolerance.
   `[Extended 2026-09-23]` 100 samples is fine for an *exactness* check (a mismatch shows on the
   first sample, not the hundredth), but it says nothing about **degenerate inputs**, which are
   not rare events — they are normal operating points. Add them as explicit cases:
   - **|D| = 0** — no synthetic drops. This is *every* batch at k = 0, i.e. the λ₂ term's
     commonest input, and the spec's original `r̃_f(D) = r_f(D)/|D|` would have divided by zero on
     all of them. The shifted/halved ρ of `01_SPEC.md` §4 removed the division; the test pins it.
   - **|A| = 0** — every channel dropped. ρ must be flat and `P_phys` uniform, never NaN.
   - **|A| = 1** — a single surviving channel, the thinnest non-empty evidence.
   - **exact ties** — two locations with identical signatures: ρ has two zeros, the margin is
     0, and PSM picks one. Legal, and must not NaN.
   All five are in `tests/test_physics_residual.py` and pass.
2. Speed: under ~5 ms per batch of 32 on the T1000 (the min is over 2,805 hypotheses × 75
   channels per candidate; LOO multiplies by 25).

**Branch.** If slow: precompute and cache r̃ for the entire corpus from the raw `.npz`. It depends
only on inputs, so this is possible — and it is the single most useful optimisation available,
enabled purely by the architectural rule in `01_SPEC.md` §5. Note that under synthetic dropout
r̃(A) changes with the mask, so the cache is per (window, mask seed, k), or the layer stays online
for training and the cache is used for evaluation only.

**Output.** `cache/rtilde_{train,val,test}.npy`

`[Corrected 2026-09-29]` **DONE — slow, cache built.** `scripts/run_residual_speed_benchmark.py`
measured `method="direct"` (float32, the only form accurate enough for a differentiable training
loop — see `residual.py`'s module docstring) at **44.7 ms/batch-of-32** on the T1000, 9x over
budget, and confirmed compute-bound rather than a batching artefact (`chunk_f` 1→26 changed nothing
— the actual table is 26 x 3366 x 75, not the 2,805-hypothesis count above, which predates the
2026-09-25 R-grid fix). `method="gram"` (float64) is only ~2x faster per window on this GPU's weak
fp64 throughput, not the "~800x" its docstring's general claim implies on faster hardware, and
still exceeds 5 ms at batch 32. Numbers: `results/pcstgnn/audit/s3_4_residual_speed/speed_benchmark.json`.
Built the branch's cache: `fault_localization/physics/cache.py::build_rtilde_cache`, keyed by
`(window_id, k, mask_seed)` via a sha256-derived seed independent of batch assembly (Python's
`hash()` on `str` is process-salted and unusable here) — verified against the direct-method
reference to ~1e-5 float tolerance before the full build. Uses `method="gram"` (exact, and the
build is offline so its float64/speed tradeoff is irrelevant). Covers k in {0,1,2,5,10} x 3 mask
seeds (k=0 replicated across the offset axis to stay rectangular) x all windows of all 3 splits:
`cache/rtilde_test.npy` (247,000 windows), `rtilde_val.npy` (248,000), `rtilde_train.npy`
(1,155,000) — shapes and window counts verified against the splits, no NaNs. This is a new,
independent seed source for training/caching use; it is **not** a drop-in replacement for the
paper's `apply_sensor_dropout_batch` evaluation-protocol seeding, which is deliberately
batch-order-dependent.

## GATE 3 — Infrastructure correctness

`[Corrected 2026-09-29]` **All five closed.**

- [x] Injectors deterministic and shared-seeded across all models — verified programmatically
      (all 6 injectors: masking/random, correlated, targeted, persistent, noise, silent return
      byte-identical output across two calls at the same seed) and architecturally (every runner
      — `run_targeted_dropout_study.py`, `run_persistent_dropout_study.py`,
      `run_correlated_dropout_study.py`, `run_noise_robustness_study.py` — applies the injector
      once per `(k, offset, batch_idx)` *outside* the `for model_name in args.models` loop, so
      every model sees the identical corrupted batch at a given seed)
- [x] Silent injectors verified to leave m = 1 and to skip uninstrumented phases —
      `tests/test_silent.py::test_mask_channels_are_never_written`,
      `test_uninstrumented_phases_are_untouched`, `test_deterministic_under_a_fixed_seed` (68/68
      tests pass across `test_silent.py` + `test_physics_residual.py` + `test_gate_checks.py`)
- [x] Residual matches the reference loop — `tests/test_physics_residual.py`, all pass including
      the five degenerate-input cases (|D|=0, |A|=0, |A|=1, exact ties, planted-corruption LOO)
- [x] Residual under 5 ms/batch, or cached — not under 5 ms (44.7 ms measured, S3.4); cache built
      and verified instead: `cache/rtilde_{test,val,train}.npy`
- [x] Physics-imputation baseline evaluated on the full sweep — S3.3 row 2, 3 seeds, full test
      split, `results/dropout_study_impute/physics_imputation_summary.json`
