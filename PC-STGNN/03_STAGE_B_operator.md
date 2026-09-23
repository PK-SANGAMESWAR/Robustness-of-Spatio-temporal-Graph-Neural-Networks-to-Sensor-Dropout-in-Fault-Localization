# Stage B — Build and Validate the Physics Operator (Weeks 2–4)

**Purpose.** Answer one question: is a magnitude-only signature table a faithful description of
this corpus? If not, nothing downstream works — and you learn it for a few hours of CPU.
**Compute.** CPU only. ~70 k OpenDSS snapshot solves.

## Context you need

Operator spec: `01_SPEC.md` §2. Axes 25 f (+ f = 0) × 11 τ × 15 R (0.05–30 Ω log-spaced) ×
17 L (0.5–1.3 linear). `[Corrected 2026-09-22]` The corpus load is one scalar per event
(stored as `meta["load_multiplier"]`), so the 50 runs of a scenario lie on the curve μ(L); that
is what makes S1.5 an exact test of the load axis.

## S1.1 — Network matrices

**Do.** Extract the three-phase Y-bus from the `.dss` model; factorise for the Z-bus entries
needed for the transfer sensitivity |Z_pf| / |Z_ff|. (`graph/sensitivity.py` already builds a
first-order path-impedance template and caches it beside the feeder config — start there.)

**Check.** Y-bus symmetric. Reconstruct one known branch current from a solved power flow via
I = Y·V.

**Pass.** Reconstructed current within 0.1 % of OpenDSS's own reported value.

**Use.** Paper's physics section and the identifiability discussion only. It is not used in the
residual — state this explicitly in the paper, because reviewers familiar with
admittance-weighted adjacency (Nguyen 2023, PPGN) will assume otherwise.

**Figure produced.** F-B1 — transfer-sensitivity map |Z_pf|/|Z_ff| over the feeder (paper,
physics section).

## S1.2 — L-grid convergence

`[Corrected 2026-09-22]` Replaces the N_L convergence study of the first draft (no ensemble any
more).

**Do.** For 3 representative (f, τ, R) — near-sensor / far-lateral / 10 Ω — build μ(L) at
L-steps 0.1, 0.05, 0.025 (and one finer reference grid). Interpolate the coarse grids linearly and
compare against the fine one.

**Check.** Max interpolation error in p.u. at each step size, on the deepest-sagging channels.

**Pass.** `[Corrected 2026-09-23, superseding the 2026-09-23 note below]` This step **decides**
the L grid. It selects on **usable ρ dynamic range, not maximum fidelity** — and those point in
opposite directions here.

Why "finest that runs" is the wrong rule for this operator: S0.4 established that the table is
built by the corpus's own generator, so a finer L grid drives r_{f_true} → 0, widens the top-2 ρ
margin, and makes `softmax(−ρ)` more nearly one-hot. That is *more* degeneracy, not less — it
pushes β toward trusting physics completely and the network's contribution toward zero (see
`04_STAGE_C_gonogo.md` §2.0). Fidelity is free here; dynamic range is the scarce quantity.

**Selection rule.** Build the top-2 ρ margin distribution (`11_METRICS.md` §2) on the training
split at each candidate step and choose the grid that keeps the margin in a workable band — a
physics prior that is informative but not saturating — subject to the interpolation error
staying below the **physical** σ_ε floor (µPMU accuracy class, `01_SPEC.md` §4). Report the
margin distribution for every candidate step alongside the choice, so the selection is auditable
and not a preference. Record the chosen value in `operator/grid.json`; `01_SPEC.md` §2 defers to
it, and σ_ε is calibrated from the discretisation error of the **chosen** grid (S1.6).

*(Superseded: the earlier rule "choose the coarsest step whose interpolation error is < 1e-3 p.u.,
and since both candidates are affordable there is no reason to accept a marginal pass at the
coarser step". Affordability was never the constraint — 0.05 is 17 points and ≈ 2.9 min serial,
0.025 is 33 points and ≈ 5.6 min (S0.5). The constraint is that fidelity beyond the physical
floor buys nothing and costs ρ dynamic range.)*

**Branch.** Neither step passes → linear interpolation in L inside the residual (voltages are
near-linear in L).

**Figure produced.** F-B2 — L-grid interpolation error vs step (appendix).

## S1.3 — Build the table

**Do.** Generate μ[f,τ,R,L] and μ₀[L] over the full grid. Parallelise across 4–6 processes, one
OpenDSS engine each. Use load L × 0.99953 (S0.4).

**Check.**
1. No NaNs.
2. No non-converged solves (log and re-solve any).
3. Monotonicity sanity: sag depth at the nearest sensor decreases monotonically with R for each
   (f, τ, L).

**Pass.** All three clean.

**Branch.** Non-monotonic sag depth usually means a regulator or cap-bank control interaction.
Decide once and match the corpus: either freeze controls in snapshot mode exactly as the corpus
Daily-mode run had them, or allow them and document it. Re-run S0.4 after any change.

**Output.** `operator/mu.npy` [25,11,15,17,25,3], `operator/mu0.npy` [17,25,3],
`operator/build_log.csv`

**Table produced.** T3 — signature table configuration (axes, grid, size, build cost).

## S1.4 — Bias check on μ

**Do.** For every training event compute e_ip = s_ip − μ[f_true, τ_true, R_true, L_true, i, p]
with L_true the stored multiplier (nearest grid point, or interpolated).

**Check.** Median and IQR of |e| by fault type, by R, and by sensor.

**Pass.** `[Corrected 2026-09-23 — this is now a construction check, not a fidelity test.]`
S0.4 measured the operator against the corpus at **1.02e-12 p.u.**: the table is not an
approximation of the generator, it *is* the generator re-run. So this check will return ~1e-12 by
construction, and the original criterion below can no longer fail for any reason that has to do
with physics.

```
median |e| / sag < 0.3      i.e. roughly  |e| < 0.005 p.u.      [retained, but now vacuous]
```

Read it accordingly: **a hit here is a bug report, not a fidelity measurement.** Anything above
the discretisation scale means the build path diverged from the corpus's — a control state
(regulator tap / cap bank) that differs between the table build and the Daily-mode corpus, a
wrong load freeze factor, or a stale `.dss`. Hunt that; never absorb it into σ_ε.

This is a good engineering outcome and a bad scientific one, and the honest response is to move
the falsifiable content of Gate 1 to the checks that still have any. It is also the reason
`01_SPEC.md` §4's third branch — "last resort: calibrate μ from training-split event means" — is
now **dead**: there is nothing to calibrate against, and invoking it would fit the operator to
data it already reproduces exactly.

**Branch, in order.**
1. Check the L-grid step (S1.2) and the 0.99953 factor.
2. Look for a systematic offset — a constant bias means a source or regulator setpoint mismatch
   between snapshot and Daily mode. Fix it, do not absorb it into σ_ε.
3. Last resort: calibrate μ from training-split event means per (f,τ,R). Describe honestly as a
   model-derived template calibrated on training data, and weaken the "no data used" claim.

**Figure produced.** F-B3 — distribution of |e| by R and fault type (box/violin, paper).
**Metrics.** See `11_METRICS.md` §4.

## S1.5 — Load-axis check — the decisive test

`[Corrected 2026-09-22]` Replaces the dispersion-slope test. Since the stored load is a scalar
and the table has it as an axis, the test is direct.

**Do.** For every training event, profile (τ, R, L) at the **true** location f_true on the full
instrumented set A and read L*. Compare with the stored `load_multiplier`.

**Check.** Distribution of |L* − L_true| in grid steps; the fraction of events where τ* = τ_true
and R* is within one grid step of R_true.

**Pass.** |L* − L_true| ≤ 1 grid step (0.05) on > 90 % of events; τ* = τ_true on > 90 %.

**Branch on failure.** The profiled optimum is landing on the wrong point of the (τ, R, L)
manifold, i.e. the table is not a faithful description even at the true location.
1. Re-read the generator — confirm the multiplier really is applied uniformly to every load
   (`monitors.py:_scale_loads_from_base`) and that no per-bus loadshape is active in Daily mode.
2. Include any other per-run randomness in the table (there should be none — the corpus is
   noise-free).
3. If the failures concentrate at 10 Ω / far laterals, that is the identifiability limit and
   belongs in S2.1, not here.

**Figure produced.** F-B4 `[Corrected 2026-09-23]` — **L* and τ* recovery**: the L* vs stored-L
1:1 scatter coloured by R, alongside the τ* confusion over the 11 types. This goes in the paper.
It converts "we assumed a load model" into "we tested our load model", and the τ* panel shows the
same for the type axis, which is profiled on exactly the same footing.

## S1.6 — Residual floor σ_ε

`[Corrected 2026-09-22]` Replaces the correlation check (moot without an ensemble).

**Do.** Set σ_ε per channel from the S0.4 snapshot-vs-Daily error (a priori, no training data).
`[Corrected 2026-09-23]` Because ρ is shifted by its own minimum (`01_SPEC.md` §4), ρ_{f_true} is
0 whenever PSM is correct and carries no scale information. Calibrate on the **unshifted**
per-channel average instead — r̄_{f_true} = r_{f_true}(A)/|A| — and report its distribution on the
training split, alongside the ρ **margin** ρ_(2) − ρ_(1), which is the quantity β and λ actually
see.

**Check.** r̄ at the true hypothesis is O(1) — not ≫ 1 (floor too small, residual over-confident)
and not ≪ 1 (floor too large, residual uninformative).

**Pass.** Median r̄_{f_true} ∈ [0.3, 3].

**Branch.** Outside that range → rescale σ_ε by one global constant, chosen on the training split,
and document it as the single calibrated scalar of the operator.

**Figure produced.** F-B5 — r̄ at the true hypothesis, by R, and the ρ margin by R (appendix).

## S1.7 — R-grid sufficiency

**Do.** Two sub-tests, both free.
1. Discretisation: rebuild the residual using only every second R grid point (7 values); re-run
   PSM (Stage C S2.4) and compare F1.
2. Interpolation: for the corpus's own R ∈ {0.1, 1, 10}, check where the profiled argmin_R lands.

**Check.** ΔF1 from halving the grid; distribution of argmin_R error in grid steps.

**Pass.** Discretisation costs < 0.01 macro-F1; argmin_R within one grid step on the large
majority of events.

**Branch.** Either fails → densify to 25 log-spaced points (≈ 117 k solves with the L axis, still
< 1 h). Report the chosen grid and this test.

**Figure produced.** F-B6 — PSM F1 vs grid density, and argmin_R accuracy (appendix).

## GATE 1 — Fidelity

`[Corrected 2026-09-23]` S1.4 is no longer a gate criterion with content — it passes by
construction (see S1.4). Gate 1's real criteria are the three checks that can still fail, plus
the new ρ-margin band that S1.2 selects on.

Proceed only if:

- [ ] S1.4 bias is at the discretisation scale — **a construction check**: above it ⇒ the build
      path diverged from the corpus, stop and find the divergence
- [ ] S1.5 L* within one grid step on > 90 % of events, τ* correct on > 90 %
      — **the decisive test of the load axis**, and an exact one, since the true L is stored
- [ ] S1.6 residual floor calibrated, median r̄_{f_true} ∈ [0.3, 3]
- [ ] S1.7 discretisation loss < 0.01 macro-F1
- [ ] **ρ top-2 margin sits in the band S1.2 selected on**, reported for every candidate L step
      (`11_METRICS.md` §2) — the check that the physics prior is informative without being
      saturating

**If Gate 1 fails after all branches.** Magnitude-only physics is too weak for this corpus. Fork to
the voltage-angle route: re-log the same events (seeds are stored, S0.2) with voltage angles at the
25 µPMUs — `monitors.py:78–80` already computes them — and rebuild the operator on complex
quantities. Take this fork here, at week 4, not at week 16.
