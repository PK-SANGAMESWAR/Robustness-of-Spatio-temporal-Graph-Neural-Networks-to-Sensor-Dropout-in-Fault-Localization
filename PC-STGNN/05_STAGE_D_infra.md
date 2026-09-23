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

## GATE 3 — Infrastructure correctness

- [ ] Injectors deterministic and shared-seeded across all models
- [ ] Silent injectors verified to leave m = 1 and to skip uninstrumented phases
- [ ] Residual matches the reference loop
- [ ] Residual under 5 ms/batch, or cached
- [ ] Physics-imputation baseline evaluated on the full sweep
