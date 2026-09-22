# Stage A — Verify Your Assets (Week 1)

**Purpose.** Nothing here produces a result. All of it prevents a month of wasted work.
**Compute.** CPU only, plus one inference sweep. No GPU training.

## Context you need

IEEE 123-bus corpus, 41,250 events / 1.65 M windows, 70/15/15 event-level split. Checkpoints
exist for GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2 (standard **and** dropout-aware, all five)
and SDR-STGNN at seeds 42/43/44. Dropout sweep k ∈ {0,1,2,5,10} with stored mask seeds
(`results/dropout_study*/`).

## S0.1 — Reproduce Paper 1

**Do.** Load all checkpoints; re-run the dropout sweep with the stored mask seeds
(`scripts/run_dropout_study.py --seeds 42 43 44`).

**Check.** Every number in Paper 1's `tab:baseline`, `tab:degradation` and `tab:aware`
reproduces to the printed precision. `[Corrected 2026-09-22]` The paper has no "Table 6"; the
CSV sources of truth are `results/fullscale_final/benchmark_summary.csv` and
`results/dropout_study{,_aware,_sdr}/degradation_curves.csv`.

**Pass.** Exact match.

**Branch on failure.** Stop the project. Every comparison in this paper is paired against these
numbers. Common causes: non-deterministic dataloader ordering, a changed normalisation
statistic, cuDNN non-determinism in the GRU, or mixing `*.prebugfix.csv` with current numbers.
Fix before anything else.

**Output.** `audit/paper1_repro.csv`

## S0.2 — Metadata audit

**Do.** For all 41,250 events confirm the `.npz` `meta` carries: `fault_bus`, `fault_type`,
`resistance`, `run_id`, `load_multiplier`, `seed`, `feeder_config`
(`simulation/runner.py:156–162`). Re-run the generator on 5 events with the stored seed.

**Check.** 100 % coverage. Records reproduce bit-for-bit.

`[Corrected 2026-09-22]` The load field is a **scalar** (one global multiplier per event), not a
per-bus vector — see `01_SPEC.md` §1–2. This is not a failure branch any more; it is the reason
L is a profiled axis. The seed is `deterministic_seed(fault_bus, fault_type, resistance, run_id)`,
so the partial-regeneration escape hatch (re-logging voltage angles for the same events —
`monitors.py:78–80` computes and discards them) is **open**.

**Output.** `audit/metadata_coverage.md`

## S0.3 — Window census and quasi-static check

`[Corrected 2026-09-22]` The census is analytic: window w covers steps [w, w+20), inception is
at step 40, and `labels.py:79` labels windows 0–19 no-fault and 20–39 fault. Fault window w has
w−20 post-inception samples, so **5 % of fault windows (w = 20) have zero and 25 % (w = 21–25)
have 1–5.** The "> 25 % zero → add a column" branch does not trigger.

**Do.** (a) Confirm the counts above on the test split. (b) For 100 random fault events,
confirm that the post-inception samples within a window are identical to numerical precision
(the fault loop in `runner.py` never rescales loads), which is what makes the last-timestep
observation vector of `01_SPEC.md` §3 exact.

**Check.** (a) matches; (b) max within-window post-fault spread < 1e-6 p.u.

**Policy** (state in the paper).
- 0 post-inception samples → physics term disabled (r̃ prefers f = 0 by construction), excluded
  from L₁/L₂.
- 1–5 samples → identical s (quasi-static), no special-casing; report their F1 separately in the
  appendix because the *network* sees mostly pre-fault content there.

**Figure produced.** F-A1 — window census histogram (appendix).
**Output.** `audit/window_census.csv`

## S0.4 — Snapshot-mode equivalence

**Do.** Take a known corpus event. Re-solve in OpenDSS snapshot mode with the existing
fault-injection code, at load `load_multiplier × 0.99953` (the pre-fault sinusoid frozen at
step 39, `monitors.py:196`) and the same (f, τ, R).

**Check.** max |V_snapshot − V_corpus_postfault| across all **instrumented** channels
(|A| < 75: absent phases are 0.0 with m = 0).

**Pass.** < 1e-3 p.u. Repeat for 5 events spanning 0.1 / 1 / 10 Ω and ≥ 3 fault types; all must
pass. Record the achieved error — it sets σ_ε in `01_SPEC.md` §4.

**Branch on failure.** The fault model differs between modes. Usual causes: the fault element
applied before rather than after the first solve; `Solve mode=snap` not re-initialising the
control iteration; different `MaxControlIter`; regulator/cap-bank state carried over from Daily
mode. Fix before building the table — the entire operator inherits this error.

**Output.** `audit/snapshot_equivalence.csv`

## S0.5 — Timing calibration

**Do.** Time 100 snapshot solves and 100 Daily-mode event simulations (60 steps each) on this
machine. Training time is already measured: RGSAGE-Max 30 epochs = 6.7 h standard / 6.9 h
dropout-aware, RGATv2 12.3 h (`results/runtime/runtime_training.csv`).

**Check.** Record both solve rates.

**Pass.** No threshold — rescale `13_BUDGET_TIMELINE.md` linearly.

**Branch.** `[Corrected 2026-09-22]` One full run already exceeds 6 h, so the reduced-seed rule
is the default: 3 seeds for R0, R2, R5, R6, G0, G2; 1 seed for R1, R3, R4, R7, G1, G3, B2ᵃ unless
Gate 4 passes and budget remains. State this in the paper's experimental setup.

**Output.** `audit/timing.md`

## GATE 0

Proceed only if:

- [ ] S0.1 reproduces the Paper-1 tables exactly
- [ ] S0.4 matches to 1e-3 p.u. on all 5 events, σ_ε recorded
- [ ] S0.3 census confirmed, quasi-static check passed, policy fixed
- [ ] S0.5 solve rates recorded and `13_BUDGET_TIMELINE.md` rescaled
