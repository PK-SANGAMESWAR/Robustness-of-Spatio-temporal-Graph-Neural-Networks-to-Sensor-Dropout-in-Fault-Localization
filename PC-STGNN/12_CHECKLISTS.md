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
- [x] The residual is computed from the input tensor alone — no window index, event id, or `[Ticked 2026-11-19: by signature: PCSTGNN.forward_pc(x, x_clean, drop_mask, ...) takes no window id, event id or inception offset.]`
      inception offset (`01_SPEC.md` §3, §5)
- [x] The sealed OOD set was opened once, in Stage G. The Gate-2 pilot used different seeds and
      is not reported `[Ticked 2026-11-19: one STARTED.json and one COMPLETED.json in results/stage_g/ood; the pipeline refuses a second run; neither paper uses pilot numbers.]`
- [x] The sealed set was never used for tuning, checkpoint selection, or any decision `[Ticked 2026-11-19: all 10 checkpoints are 5.9 days older than the opening and match the sha256 in config/stage_g/frozen_models.json; the model list was frozen before the run; data/ood_test is unmodified.]`
- [ ] If any model receives extra input, all baselines receive the same input
- [x] Mask seeds, corruption seeds, and noise seeds are shared across all models `[Ticked 2026-11-19: by code reading: fault_localization/stage_g/scoring.py builds seeds from (k, offset, batch index) and (seed index, offset, event index) only; no model argument enters.]`
- [ ] Checkpoint selection is validation F1 at k = 0 for every model, as in Paper 1; λ selection
      on validation F1 averaged over k is stated separately
- [ ] The paper states the table comes from the nominal feeder model, and backs it with the S6.2
      mismatch study
- [ ] The Y-bus is stated as used for analysis only, not inside the residual
- [ ] Any "first" claim is narrowed against Karabulut et al. (arXiv 2510.03571 — sensor
      configuration and topology generalisation for detection) and PPGN
- [x] Pre-bugfix Paper-1 files (`*.prebugfix.*`) are never mixed with current numbers `[Ticked 2026-11-19: paper1_repro.json: 4 prebugfix files exist, current_tables_reference_prebugfix is empty.]`

### The two-reference pairing trap `[Added 2026-09-23]`

There are now **two** reference models, and confusing them would credit physics with a gain that
noise augmentation alone produced:

- **R0** — the Paper-1 dropout-aware checkpoints, trained noise-free, re-evaluated inside Paper
  2's pipeline (S0.1). The Paper-1 anchor and the S4.1 null-test target.
- **R0ⁿ** — the same recipe **plus noise augmentation**, no physics. The paired reference for
  every physics arm (`06_STAGE_E_pcstgnn.md` S4.3ᵇ).

- [x] Every physics ablation pairs against **R0ⁿ**, never against R0 `[Ticked 2026-11-19: T13_statistics_test.md pairs R5 against R0ⁿ (Amendment 9).]`
- [x] **R0ⁿ − R0 is itself a reported number** — what noise augmentation alone buys — and it is a `[Ticked 2026-11-19: T7_ablation_val.md has the row 'R0ⁿ − R0 (noise augmentation alone)' before any physics arm.]`
      row in T7 *before* any physics arm
- [x] T7 carries **both** columns explicitly, so the decomposition into "noise augmentation" and `[Ticked 2026-11-19: T7_ablation_val.md lists R0 and R0ⁿ as separate rows with the difference row.]`
      "physics" is visible rather than asserted
- [ ] Any cross-paper comparison states which of the two it is paired against

## B. Reproducibility

- [x] Paper 1 tables reproduce exactly (S0.1) `[Ticked 2026-11-19: results/pcstgnn/audit/paper1_repro.json: pass true at printed precision 1e-4; it checks stored files (reeval_run false), so it is a check of the stored numbers, not a retrain.]`
- [x] Snapshot, Daily and reuse modes agree with the corpus on **1,000 events** (S0.4 as `[Ticked 2026-11-19: results/pcstgnn/audit/build_path_1000/build_path.json, 1,000 events: snapshot, daily and reuse_limit_1 are identical (median 1.04e-10 p.u., p95 6.2e-9, max 7.3e-3); 1.8 % of events exceed 1e-3 p.u., which the settling check traces to tap/control hysteresis; the never-recompile path fails (43.5 % above 1e-3), as the box warns.]`
      corrected — the original 5-event version of this line passed while the build path was
      wrong for 43 % of the corpus; see §0)
- [x] The table is built with a **recompile per solve**; `build_table` is never run with a `[Ticked 2026-11-19: operator/grid.json: solve_mode daily, reuse_limit 1, 84,167 solves, 0 non-converged.]`
      standing circuit (`mode="reuse"` with `reuse_limit > 1`)
- [x] λ₁ = λ₂ = β = 0 reproduces R0 exactly (S4.1) `[Ticked 2026-11-19: tests/test_pc_stgnn.py::test_all_switches_off_is_exactly_rgsage and ::test_fusion_at_beta_zero_is_also_exactly_rgsage pass (75 passed over four physics test files).]`
- [x] Residual layer matches the slow reference loop (S3.4) `[Ticked 2026-11-19: tests/test_physics_residual.py::test_matches_the_reference_loop passes; speed in results/pcstgnn/audit/s3_4_residual_speed/speed_benchmark.json.]`
- [x] Injectors deterministic under a fixed seed, across models (S3.1) `[Ticked 2026-11-19: tests/test_silent.py::test_deterministic_under_a_fixed_seed passes for every kind; an injector takes a batch and a seed, not a model, so models that share the seed get the same corruption.]`
- [x] `data/ood_test.sha256` verified before the set is opened `[Ticked 2026-11-19: STARTED.json hash_computed equals hash_recorded (2a61cff0...).]`
- [ ] `14_PREREGISTRATION.md` committed before any test-split result was viewed; commit hash
      recorded in the paper

## C. Correctness traps specific to this method

- [x] Gradient audit passed (S4.2). A reconstruction loss at the arg-min hypothesis has zero `[Ticked 2026-11-19: results/pcstgnn/audit/s4_2_gradient/gradient_audit.json; PASSED 2026-09-30 in 06_STAGE_E_pcstgnn.md S4.2.]`
      gradient — do not use one
- [x] Silent injectors leave m = 1. Verify explicitly; if they zero the mask you are testing `[Ticked 2026-11-19: tests/test_silent.py::test_mask_channels_are_never_written passes.]`
      ordinary dropout
- [x] Silent injectors never touch uninstrumented phases (V = 0, m = 0) `[Ticked 2026-11-19: tests/test_silent.py::test_uninstrumented_phases_are_untouched passes.]`
- [x] LOO residual refits on A \ {i}, not on A — otherwise the corrupted sensor explains itself `[Ticked 2026-11-19: tests/test_physics_residual.py::test_loo_beats_in_sample_on_a_planted_corrupt_sensor and ::test_in_sample_can_hide_a_corruption_entirely pass.]`
- [ ] The gate scales the input embedding and vote weight, not messages at every hop `[N/A 2026-11-20: this item is about the trained PCRG gate, which was not built (14, Amendment 5: PCRG G0–G3 replaced by G-TF2; A5/H5 not tested). G-TF2 has no learned gate: a flagged sensor is zeroed (six channels) at the input, the same path as dropout (`stage_f_gtf2_preregistration.md` § 3). F-F4 (gate-value distribution) does not apply. Box left unticked on purpose.]`
- [x] The observation vector is the last timestep, de-normalised to p.u. with the loader's stored `[Ticked 2026-11-19: tests/test_physics_cache_observation.py::test_observation_is_last_timestep_denormalised_with_mask_availability passes.]`
      stats (`data/dataset.py:30–31`); the table is in p.u.
- [x] Windows with zero post-inception samples (w = 20) are excluded from L₁/L₂ `[Ticked 2026-11-19: tests/test_pc_stgnn.py::test_loss_mask_excludes_the_unlabelable_windows passes (w0020 mask 0).]`
- [x] σ₀ in the residual denominator is set to the injected noise level under the noise `[Ticked 2026-11-19: tests/test_noise_augmentation.py (per-window sigma0 equals the injected grid value; scalar rows match) passes.]`
      experiments, not left at 0
- [x] The no-fault hypothesis f = 0 is in the residual (μ₀[L]); PSM can predict class 0 `[Ticked 2026-11-19: tests/test_physics_residual.py::test_flatten_table_keeps_the_no_fault_row passes.]`
- [x] The snapshot solve uses L × 0.99953 (the frozen pre-fault modulation) `[Ticked 2026-11-19: tests/test_signature_solver.py::test_prefault_freeze_is_the_step_39_value passes (0.99953).]`
- [x] r̃ cache entries are keyed by (window, mask seed, k) — r̃(A) changes with the mask `[Ticked 2026-11-19: tests/test_physics_cache_observation.py::test_cache_seed_is_keyed_by_window_k_and_mask_seed_and_is_stable passes.]`

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
