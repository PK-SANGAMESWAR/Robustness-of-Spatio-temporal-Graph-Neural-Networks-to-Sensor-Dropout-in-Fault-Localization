# Stage A — Verify Your Assets (Week 1)

**Purpose.** Nothing here produces a result. All of it prevents a month of wasted work.
**Compute.** CPU only, plus one inference sweep. No GPU training.

## Context you need

IEEE 123-bus corpus, 41,250 events / 1.65 M windows, 70/15/15 event-level split. Checkpoints
exist for GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2 (standard **and** dropout-aware, all five)
and SDR-STGNN at seeds 42/43/44. Dropout sweep k ∈ {0,1,2,5,10} with stored mask seeds
(`results/dropout_study*/`).

## S0.0 — Load-model check `[Added 2026-09-23]`

**Do.** Settle in writing whether the corpus load is one global scalar or a per-bus vector: read
`meta["load_multiplier"]` from all 41,250 `.npz` files and record its type and coverage, and quote
the two lines of `paper/overleaf.tex` that describe the load draw.

**Check.** The field is a Python float in 100 % of events (not a list/array), and the Paper-1 draft
on disk says "per run", never "per bus".

**Pass.** Both. This is already known to hold (`../PC-STGNN-PLAN.md` §0) — the script exists to
turn a read-only verification into a dated artefact, because the entire L axis of `01_SPEC.md` §2
rests on it. If the submitted copy of Paper 1 ever claimed a per-bus load, Paper 2 must not repeat
it.

**Output.** `audit/load_model.md`

## S0.1 — Reproduce Paper 1

**Do.** Load all checkpoints; re-run the dropout sweep with the stored mask seeds
(`scripts/run_dropout_study.py --seeds 42 43 44`).

**Check.** Every number in Paper 1's `tab:baseline`, `tab:degradation` and `tab:aware`
reproduces to the printed precision. `[Corrected 2026-09-22]` The paper has no "Table 6"; the
CSV sources of truth are `results/fullscale_final/benchmark_summary.csv` and
`results/dropout_study{,_aware,_sdr}/degradation_curves.csv`.

**Pass.** `[Corrected 2026-09-23]` Exact match at k = 0. At k > 0 exact match is **not
achievable** and never was, for a reason that is now measured rather than suspected: the mask
realisation is a pure function of (k, mask offset, `batch_idx`, **batch size**) — `evaluator.py`
derives the per-batch seed as `mask_seed * 10000 + batch_idx` — and **nothing on disk records the
batch size the stored Paper-1 curves were produced with**. Re-evaluating at batch 32 and batch 64
gives curves differing from each other by up to 9.3e-4 macro-F1, the same order as either differs
from the stored one. The stored curve is simply a third, equally valid, mask realisation.

**Resolution — re-evaluate, do not tolerate.** Accepting a ~1e-3 drift would import an error bar
into every paired comparison Paper 2 makes. Instead:

1. **R0 and every Paper-1 arm are re-evaluated fresh inside Paper 2's own pipeline** (33 runs:
   5 standard × 3 seeds, 5 dropout-aware × 3 seeds, SDR-STGNN × 3; ≈ 5.7 h of inference, no
   training, so it is permitted before Gate 2). Those numbers — not the stored CSVs — are the
   reference for every paired comparison in Paper 2, which is then internally consistent by
   construction. Output: `audit/reeval/{standard,dropout_aware,sdr}/degradation_curves.csv`.
2. Paper 1's published values are cited separately, with a one-line footnote recording the
   mask-realisation tolerance (~1e-3 macro-F1, ≈ 5 % of the training-seed std at the same k).
   No published number moves.
3. **Batch size is recorded in every curve CSV from the first Paper-2 run**
   (`run_dropout_study.py` now emits a `batch_size` column), so this is never rediscovered.

**Branch on failure.** Stop the project — but only for the failures this gate was written for: a
changed metric, a changed normalisation statistic, non-deterministic evaluation, or
`*.prebugfix.csv` mixed with current numbers. All four are positively excluded as of 2026-09-23:
k = 0 reproduces exactly, the same command run twice agrees to 0.00e+00, the three free layers
(derived / aggregate / curves) match to ≤ 4.0e-7, and the pre-bugfix files are quarantined and
unreferenced. A mask-realisation difference at k > 0 is **not** this branch.

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

**Check.** (a) matches; (b) `[Corrected 2026-09-23]` post-inception samples are constant to
**< 1e-6 p.u. from step 42 onward**; the first two post-inception steps (40, 41) carry a
**settling** of **≤ 2e-5 p.u. for 41,247 of the 41,250 events**, with **three documented
control-driven exceptions**.

Measured over the **whole corpus** (not a sample): max settled spread from step 42 onward
**1.11e-7 p.u.**; max |V[41] − V[40]| **1.98e-5 p.u.** excluding three events, p99.9 1.60e-5,
median 1.67e-6; max |V[42] − V[41]| 6.99e-6.

**The three exceptions are a real finding, not noise.** `55_BC_1p0_0007`, `7_CA_10p0_0002` and
`51_ABC_1p0_0000` settle at **6.4–7.1e-3 p.u.**, ~300× the bulk. All three are **ungrounded**
faults (BC, CA, ABC) and in each the voltage *recovers* between steps 40 and 41 (1.0137 → 1.0208
p.u. at node 16 of `55_BC_1p0_0007`) instead of settling downward — a **control device operating
in response to the fault**, taking one extra solve, not a numerical artefact. This is almost
certainly the same mechanism as the 3.3 % non-monotone sag cells of S1.3
(`03_STAGE_B_operator.md`), so one control-state decision covers both; check these three buses
there. For those events the w = 21/22 windows — **6 windows out of 1.65 M** — observe a level up
to 7e-3 p.u. off the settled one, which is *not* comfortably below σ_ε. Far too rare to affect
training or any reported metric; enumerated in `SETTLING_OUTLIERS` so a fourth fails loudly
rather than being averaged in.

**An earlier 400-event sample reported 1.44e-5 and missed this tail entirely.** That is the
argument for the assertion below in one line: the sampled statistic was not wrong, it was
sampled.

The original flat `< 1e-6` threshold is not met and is restated rather than quietly relaxed. Two
requirements come with the restatement:

- **It is an assertion in `audit_window_census.py`, not prose** — three of them: the settled
  level is flat to < 1e-6 from step 42; the bulk settling stays under `SETTLING_BOUND_PU` = 2e-5;
  and **no event outside the documented set of three exceeds it**. Pass `--sigma-eps` once S1.2
  has set it and a fourth assertion requires the settling to stay an order of magnitude below the
  residual floor. This matters because s_ip is the last timestep and windows w = 21–22 have only
  1–2 post-inception samples, so they observe steps 40 and 41 directly. Scan the full corpus with
  `--n-quasi-static 0` (15 s); the default 100-event sample will not see the three outliers.
- **Call it solver settling, never a "transient".** It is the number of solves OpenDSS needs to
  reach the settled post-fault level after the fault element is energised — a property of the
  solver, not of the feeder. Describing it as a transient invites "so there *are* dynamics, why
  not model them?", and this project has deliberately decided not to claim physics-informed
  temporal dynamics (§ `01_SPEC.md` §3).

**Consequence for the operator.** The table must **not** be built from a single fault solve: it
would sit on the unsettled level while nearly every observation sits on the settled one — a
small, systematic, location-dependent bias that S1.4 would misread as operator error.
`physics/signature_solver.py` takes `n_fault_steps` (default **3**, converged to 2.8e-8).

**Policy** (state in the paper).
- 0 post-inception samples → physics term disabled (r̃ prefers f = 0 by construction), excluded
  from L₁/L₂.
- 1–5 samples → identical s (quasi-static), no special-casing; report their F1 separately in the
  appendix because the *network* sees mostly pre-fault content there.

### Unlabelable-window ceiling `[Added 2026-09-23]`

The w = 20 windows are not merely awkward: they carry a fault label while containing **zero**
post-inception samples, so they are physically indistinguishable from a no-fault window of the
same event. No model — physics, learned, or hybrid — can beat chance on them. That is a hard
ceiling on every absolute macro-F1 in both papers, and it has never been quantified.

**Do.** On the test split: (a) compute each existing model's macro-F1 restricted to the w = 20
subset, and (b) compute the implied 26-class macro-F1 ceiling for a model that is otherwise
perfect and scores at chance on that subset.

**Check.** (a) is ≈ chance. **Anything meaningfully above chance on w = 20 is evidence of label
leakage** — investigate before reporting anything else. (b) is reported as the ceiling.

**Use.** Reframes every absolute number in Paper 1 and Paper 2, and feeds the error attribution
F-H1 in `09_STAGE_H_writing.md`. The metric is defined in `11_METRICS.md` §1.

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

**Freeze assertion** `[Added 2026-09-23]`. The script must **assert**, not assume, that the corpus
fault steps sit at `L × 0.99953`: recompute `1 + 0.003·sin(2π·39/40)` from the code, multiply by
the event's stored `load_multiplier`, and confirm the fault-period solve at that load reproduces
the recorded post-fault level while the step-40 value `L × 1.0` does not. The fault loop of
`simulation/runner.py` never rescales loads, so the last rescale is pre-fault step 39 — the
step-40 value of the sinusoid is never applied. `PREFAULT_FREEZE` in `physics/grids.py` carries
this constant and a docstring pointing here.

**Output.** `audit/snapshot_equivalence.csv`

## S0.5 — Timing calibration

**Do.** Time 100 snapshot solves and 100 Daily-mode event simulations (60 steps each) on this
machine. Training time is already measured: RGSAGE-Max 30 epochs = 6.7 h standard / 6.9 h
dropout-aware, RGATv2 12.3 h (`results/runtime/runtime_training.csv`).

**Check.** Record both solve rates.

**Pass.** No threshold — rescale `13_BUDGET_TIMELINE.md` linearly.

**Branch.** `[Corrected 2026-09-22]` One full run already exceeds 6 h, so the reduced-seed rule
is the default: 3 seeds for R0, R2, **R4**, R5, R6, G0, G2; 1 seed for R1, R3, **R7**, G1, G3, B2ᵃ
unless Gate 4 passes and budget remains. State this in the paper's experimental setup.

`[Corrected 2026-09-23]` **Seed swap R4 ↔ R7.** λ₂ is the novel loss term and R4 is the only arm that isolates it, so R4 gets 3 seeds; R7 (size-matched wider backbone) answers a one-line objection with a one-row answer and gets 1.

**Output.** `audit/timing.md`

## GATE 0

Proceed only if:

- [ ] S0.1 reproduces the Paper-1 tables exactly
- [ ] S0.4 matches to 1e-3 p.u. on all 5 events, σ_ε recorded
- [ ] S0.3 census confirmed, quasi-static check passed, policy fixed
- [ ] S0.5 solve rates recorded and `13_BUDGET_TIMELINE.md` rescaled
- [ ] S0.0 load model recorded as a scalar, in writing `[Added 2026-09-23]`
- [ ] S0.3 unlabelable-window ceiling computed; no model beats chance on w = 20 `[Added 2026-09-23]`
- [ ] S0.4 freeze assertion (`L × 0.99953`) passes `[Added 2026-09-23]`
