# Checklists

## 0. Sampling — how every acceptance criterion is measured `[Added 2026-09-23]`

**Every acceptance criterion states its sample size. Any criterion that gates a decision runs on
the full population — or on a pre-registered random sample with a stated confidence bound.
Convenience samples are for smoke tests only.**

This is a rule because it has already failed three times, in three different shapes — a sample
that missed the population, a grid that silently corrupted one axis, and two stages using
different scopes for the "same" number without either one stating its scope:

| Criterion | Failure mode | What the full check said |
|---|---|---|
| S0.4 "snapshot ≡ daily, so the fast build path is validated" | **5-event sample** | The standing-circuit path disagrees with the corpus on **43.3 %** of 1,000 events. It would have silently corrupted the operator |
| S0.3 "settling ≤ 1.4e-5 p.u." | **400-event sample** | 1.98e-5 for 41,247 of 41,250 — and **three** control-driven outliers at 6.4–7.1e-3 that the sample never saw |
| S2.4's "PSM ≥ STGNN, confirmed" (`04_STAGE_C_gonogo.md`, 2026-09-25→29) | **cross-stage scope mismatch** — S1.7's 0.8977 (event-level, single settled step, 25-class, fault-only) was compared directly against a window-level 26-class network number, no shared protocol stated | Measured side by side (`scripts/check_s2_4_metric_consistency.py`): the gap was **entirely the metric** (~0.13, present equally on both splits); the real train→val generalisation gap was **+0.004**, and the corrected in-distribution reading is the opposite sign (network wins by ~0.08) |

Neither sampled statistic was *wrong*; both were **sampled**, and both were then written into the
spec as though they were population facts. S2.4's case is a distinct third shape — both numbers
were exact, full-population measurements, individually correct, and still not comparable, because
neither carried its evaluation protocol (population granularity, class count, aggregation level)
into the sentence that compared them. The failure mode is the same each time in one respect: a
check passes, its scope is dropped in the write-up, and a decision is taken on it.

So, concretely:

- [ ] Every check reports `n`, and how the `n` was drawn, in its own output JSON
- [ ] Decision-gating checks (anything feeding a Gate) run on the full population where that is
      affordable — and on this corpus it almost always is: the whole 41,250-event corpus scans in
      15 s, and a full L-grid table solves in minutes
- [ ] Where full population is genuinely infeasible, the sample is **pre-registered** (scheme and
      seed recorded before the run) and the result carries a **confidence bound**, not a point
      estimate
- [ ] A check that passed on a convenience sample may be cited as a smoke test and **may not be
      cited as evidence** for a gate
- [ ] **No runtime, latency, or throughput measurement is taken while anything else is running.**
      T12 and the `run_runtime_benchmark.py` extensions need a quiet box or the numbers are
      worthless. F1 and every other accuracy number is deterministic under contention and is
      unaffected, so only the cost measurements carry this constraint (`11_METRICS.md` §7)
- [ ] `[Added 2026-09-29]` **Any number that crosses from one stage's write-up into a comparison
      or a Gate decision in another stage states its evaluation protocol inline** — population
      granularity (event vs window), class count, and aggregation level (per-window vs
      soft-voted-per-event) — not just its split and its value. Two numbers with the same name
      (e.g. "PSM macro-F1", "STGNN F1 at k=0") computed by two different scripts are not assumed
      comparable until this is checked
- [ ] `[Added 2026-09-29]` Before a cross-stage or cross-model comparison is written into a Gate
      table, run (or point to) a **metric-consistency check** — recompute at least one of the two
      numbers under the other's exact protocol, on a shared split — the same discipline this
      section already requires for sample scope, applied to metric scope. See
      `scripts/check_s2_4_metric_consistency.py` for the pattern (a 2x2 over {metric} x {split}
      that decomposes a suspicious gap into its real components rather than asserting an
      explanation for it)

## A. Leakage and fairness — verify before every table

- [ ] The true R, fault type, or load multiplier is never supplied to the model or the residual
      at test time (the stored `load_multiplier` is used only in S1.4/S1.5 diagnostics)
- [ ] σ_ε, β, λ₁, λ₂, the gate parameters α, c, and the R and L grids are all fixed using
      train/validation only
- [ ] The residual is computed from the input tensor alone — no window index, event id, or
      inception offset (`01_SPEC.md` §3, §5)
- [ ] The sealed OOD set was opened once, in Stage G. The Gate-2 pilot used different seeds and
      is not reported
- [ ] The sealed set was never used for tuning, checkpoint selection, or any decision
- [ ] If any model receives extra input, all baselines receive the same input
- [ ] Mask seeds, corruption seeds, and noise seeds are shared across all models
- [ ] Checkpoint selection is validation F1 at k = 0 for every model, as in Paper 1; λ selection
      on validation F1 averaged over k is stated separately
- [ ] The paper states the table comes from the nominal feeder model, and backs it with the S6.2
      mismatch study
- [ ] The Y-bus is stated as used for analysis only, not inside the residual
- [ ] Any "first" claim is narrowed against Karabulut et al. (arXiv 2510.03571 — sensor
      configuration and topology generalisation for detection) and PPGN
- [ ] Pre-bugfix Paper-1 files (`*.prebugfix.*`) are never mixed with current numbers

### The two-reference pairing trap `[Added 2026-09-23]`

There are now **two** reference models, and confusing them would credit physics with a gain that
noise augmentation alone produced:

- **R0** — the Paper-1 dropout-aware checkpoints, trained noise-free, re-evaluated inside Paper
  2's pipeline (S0.1). The Paper-1 anchor and the S4.1 null-test target.
- **R0ⁿ** — the same recipe **plus noise augmentation**, no physics. The paired reference for
  every physics arm (`06_STAGE_E_pcstgnn.md` S4.3ᵇ).

- [ ] Every physics ablation pairs against **R0ⁿ**, never against R0
- [ ] **R0ⁿ − R0 is itself a reported number** — what noise augmentation alone buys — and it is a
      row in T7 *before* any physics arm
- [ ] T7 carries **both** columns explicitly, so the decomposition into "noise augmentation" and
      "physics" is visible rather than asserted
- [ ] Any cross-paper comparison states which of the two it is paired against

## B. Reproducibility

- [ ] Paper 1 tables reproduce exactly (S0.1)
- [ ] Snapshot, Daily and reuse modes agree with the corpus on **1,000 events** (S0.4 as
      corrected — the original 5-event version of this line passed while the build path was
      wrong for 43 % of the corpus; see §0)
- [ ] The table is built with a **recompile per solve**; `build_table` is never run with a
      standing circuit (`mode="reuse"` with `reuse_limit > 1`)
- [ ] λ₁ = λ₂ = β = 0 reproduces R0 exactly (S4.1)
- [ ] Residual layer matches the slow reference loop (S3.4)
- [ ] Injectors deterministic under a fixed seed, across models (S3.1)
- [ ] `data/ood_test.sha256` verified before the set is opened
- [ ] `14_PREREGISTRATION.md` committed before any test-split result was viewed; commit hash
      recorded in the paper

## C. Correctness traps specific to this method

- [ ] Gradient audit passed (S4.2). A reconstruction loss at the arg-min hypothesis has zero
      gradient — do not use one
- [ ] Silent injectors leave m = 1. Verify explicitly; if they zero the mask you are testing
      ordinary dropout
- [ ] Silent injectors never touch uninstrumented phases (V = 0, m = 0)
- [ ] LOO residual refits on A \ {i}, not on A — otherwise the corrupted sensor explains itself
- [ ] The gate scales the input embedding and vote weight, not messages at every hop
- [ ] The observation vector is the last timestep, de-normalised to p.u. with the loader's stored
      stats (`data/dataset.py:30–31`); the table is in p.u.
- [ ] Windows with zero post-inception samples (w = 20) are excluded from L₁/L₂
- [ ] σ₀ in the residual denominator is set to the injected noise level under the noise
      experiments, not left at 0
- [ ] The no-fault hypothesis f = 0 is in the residual (μ₀[L]); PSM can predict class 0
- [ ] The snapshot solve uses L × 0.99953 (the frozen pre-fault modulation)
- [ ] r̃ cache entries are keyed by (window, mask seed, k) — r̃(A) changes with the mask

## D. Claim discipline

- [ ] "Physics-consistent", never "PINN" or "physics-informed PINN"
- [ ] No claim of KCL, KVL, or power-flow residuals — the data cannot support them
- [ ] No claim of physics-informed temporal dynamics — OpenDSS Daily mode is quasi-static, there
      are no transients inside a window
- [ ] Resistance invariance stated as approximate, under the adopted fault model and operating
      conditions
- [ ] The signature table is disclosed as privileged model knowledge, with the mismatch study as
      evidence it degrades gracefully
- [ ] The load model is disclosed as a single global multiplier (the corpus's own), not a per-bus
      load model
- [ ] The contribution is framed as physics-consistent decision making on a spatio-temporal graph
      representation, not a new GNN layer
- [ ] If PSM ≥ STGNN on the clean corpus, say so in the abstract; the contribution is then where
      learning still helps, not "the GNN beats physics"
