# Stage A — audit note (Gate 0)

**Date:** 2026-09-23 · **Scope:** `PC-STGNN/02_STAGE_A_audit.md` S0.0–S0.5 · **Compute:** CPU only,
no GPU training. Scripts: `scripts/audit_{load_model,paper1_repro,metadata,window_census,
snapshot_equivalence,timing}.py`. Raw outputs alongside this file.

## Verdict

| Step | Check | Result |
|---|---|---|
| S0.0 | Corpus load is one global scalar | **PASS** |
| S0.1 | Paper-1 numbers reproduce | **PASS** on the three free layers and **exactly at k = 0**; at k > 0 agrees to ≤ 6.7e-04, which is mask-realisation noise (see below). **Ruling: re-evaluate rather than tolerate** — the full 33-arm re-eval is the reference set for Paper 2 |
| S0.2 | Metadata coverage + bit-identical regeneration | **PASS** |
| S0.3 | Window census, quasi-static, unlabelable ceiling | **census PASS · no leakage · ceiling 0.9744 · quasi-static restated and asserted** — the assertion then exposed **3 control-driven settling outliers** a 400-event sample had missed |
| S0.4 | Snapshot/Daily equivalence, freeze assertion | **PASS**, and far better than required |
| S0.5 | Timing calibration | **PASS** — the CPU budget is ~9x too pessimistic |

**Gate 0 is passable. All three open items were ruled on by the author on 2026-09-23, together
with a fourth and larger consequence — that the operator is the corpus's own generator — which
reshapes where the project's evidence lives. See § "Decisions taken".** None of the four is a
failure of the method. Each is a place where a measurement came back far better, or far
different, than the spec assumed — and where quietly relaxing the written criterion to make it
pass would have been the wrong way to resolve it.

## S0.0 — The load is a scalar

All **41,250** events store `load_multiplier` as a Python `float`; zero are lists or arrays, zero
are missing, and there are **41,250 distinct values** (one independent draw per event) spanning
0.500048–1.299998. The Paper-1 draft on disk says "`\mathcal{U}(0.5, 1.3)` per run" and "differ
only in the load multiplier drawn per run"; **zero lines claim a per-bus load**.

Decision 1 of `PC-STGNN-PLAN.md` §1 is confirmed: L is a profiled scalar axis, the 50 runs of a
scenario lie on the 1-D curve μ(L), and S1.5 is an exact test against stored ground truth.

## S0.1 — Paper-1 reproduction

| Layer | Result |
|---|---|
| `derived` — macro-F1 recomputed from each run's own `predictions_test.csv` vs its `metrics.json` | **33/33 match**, worst abs diff **1.1e-16** |
| `aggregate` — `benchmark_summary.csv` recomputed from per-seed `metrics.json` | **5/5 models match**, worst abs diff **5.6e-17** |
| `curves` — every `degradation_curves.csv` row vs its own `f1_per_train_seed` | **55/55 match**, worst abs diff **4.0e-7** (the CSV rounds per-seed values to 6 dp) |
| pre-bugfix quarantine | 4 `*.prebugfix.*` files on disk; **none referenced by a current table** |

The stored tables are therefore internally consistent to machine precision, and the metric
definition used in Paper 1 is reproducible from the prediction dumps.

### The `reeval` layer — a real, small discrepancy at k > 0

RGSAGE-Max seed 42 was re-evaluated end to end from its checkpoint (10 min 18 s per
(model, seed); the full 33-run version is ≈ 5.7 h).

| k | stored (seed 42) | re-evaluated | abs diff |
|---|---|---|---|
| 0 | 0.852620 | 0.852620 | **2.2e-07** (the CSV rounds per-seed values to 6 dp — this is an exact match) |
| 1 | 0.722050 | 0.722585 | 5.4e-04 |
| 2 | 0.562632 | 0.561961 | 6.7e-04 |
| 5 | 0.209940 | 0.210196 | 2.6e-04 |
| 10 | 0.050694 | 0.050818 | 1.2e-04 |

Three facts localise this precisely:

1. **k = 0 reproduces exactly.** That level applies no masks, so the checkpoint, the data
   pipeline, the soft vote and the metric definition are all confirmed reproducible.
2. **The evaluation is bit-identical run to run.** The same command run twice agreed to
   **0.00e+00** at every k, and the model's forward pass is bit-identical on repeat call and on
   fresh load (`cudnn.benchmark` is off). GPU non-determinism and the "non-deterministic
   dataloader ordering" branch of S0.1 are both **ruled out**.
3. **Therefore the difference lives entirely in the mask draw.** Masks are a pure function of
   `(k, mask offset, batch_idx, batch size)` — `evaluator.py` derives the per-batch seed as
   `mask_seed * 10000 + batch_idx` — so a different batch size gives a different, equally valid,
   still model-paired mask realisation. **Nothing on disk records the batch size the stored
   curves were produced with.**

This was then confirmed directly by re-running at a second batch size:

| k | stored (seed 42) | batch 32 | batch 64 |
|---|---|---|---|
| 0 | 0.852620 | 0.852620 | 0.852620 |
| 1 | 0.722050 | 0.722585 | 0.723008 |
| 2 | 0.562632 | 0.561961 | 0.561977 |
| 5 | 0.209940 | 0.210196 | 0.209266 |
| 10 | 0.050694 | 0.050818 | 0.049947 |

k = 0 is identical across all three because it applies no masks. At k > 0 the two batch sizes
differ from **each other** by up to **9.3e-04** — the same order as either differs from the
stored curve. The stored curve is simply a third mask realisation. So the discrepancy is
quantified and bounded: **mask-realisation sensitivity is ~1e-3 macro-F1**, and that is the size
of the disagreement, not a defect in the pipeline.

**Size of the effect.** At most 6.7e-4 macro-F1, against a training-seed std of ~1.3e-2 at the
same k — roughly 5 % of one error bar. It is invisible in every comparison the papers make. But
at the three decimals the paper prints, k = 1 rounds to 0.722 stored and 0.723 re-evaluated, so
it is not *nothing* either, and "exact match" as S0.1 words it is not satisfied at k > 0.

**Recommendation.** Do not treat this as the stop-the-project branch: that branch exists for a
changed metric, a changed normalisation, or a non-reproducible pipeline, and all three are now
positively excluded. Three concrete actions instead:

1. **Restate S0.1's tolerance at k > 0.** "Exact match" is not achievable for a masked level
   unless the mask configuration is recorded, and it never was. The honest criterion is
   *exact at k = 0, and within mask-realisation noise (~1e-3 macro-F1) at k > 0* — with the
   noise floor measured, as above, rather than assumed.
2. **Record the mask configuration from now on.** The curve CSVs should carry the batch size (and
   loader settings) alongside `n_mask_seeds`, so a future re-evaluation is reproducible to the
   bit. Paper 2's own sweeps must do this from their first run.
3. **Run the full `--reeval` before Gate 0 is signed off** (≈ 5.7 h of otherwise-idle GPU) to
   confirm the ≤ 1e-3 bound holds for all 11 model/arm combinations, not only RGSAGE-Max.

None of this moves a published number: 6.7e-04 is ~5 % of the training-seed std at the same k, so
every comparison in Paper 1 and every paired comparison planned for Paper 2 is unaffected.

The archived variants were checked too and are *not* the reference: `degradation_curves.bak.csv`
and `.prebugfix.csv` are identical to each other and differ from the current curve by up to
5.6e-2 — two orders of magnitude more than anything above, which is exactly the bugfix's
documented effect and a good reminder of why they stay quarantined.

## S0.2 — Metadata and regeneration

All seven keys present in 41,250/41,250 events; every record `[25, 60, 6]`; **0** disagreements
between the stored `seed` and `deterministic_seed(bus, type, R, run_id)`. Five events re-simulated
from their stored seed reproduced **bit-for-bit** (max abs diff exactly 0.0, load multiplier
exact).

The voltage-angle fallback of Gate 1 is therefore **open**: the same events can be re-logged with
angles, which `monitors.py:78–80` already computes and discards.

## S0.3 — Window census, quasi-static, and the F1 ceiling

**Census — exact.** 123,500 fault-labelled test windows; **5.00 %** have zero post-inception
samples (w = 20) and **25.00 %** have 1–5 (w = 21–25), matching the analytic prediction to the
digit. The "> 25 % → add a column" branch does not trigger. Figure `F-A1_window_census.png`.

**Quasi-static — misses its stated threshold, for an identifiable reason, plus a tail the first
pass missed.** `[Revised 2026-09-23 after re-measuring on all 41,250 events.]` Max within-event
spread of the post-inception samples is **7.09e-3 p.u.**, against the spec's `< 1e-6`. The cause
is not drift but **solver settling** — the solves the iteration needs after the fault element is
energised. Call it settling, never a "transient": this project does not claim physics-informed
temporal dynamics, and the word invites the obvious follow-up question.

Measured over the **whole corpus**, not a sample:

| quantity | value |
|---|---|
| settled spread, step 42 onward | **1.11e-7 p.u.** |
| max \|V[41] − V[40]\|, excluding 3 events | **1.98e-5 p.u.** (p99.9 1.60e-5, median 1.67e-6) |
| max \|V[42] − V[41]\| | 6.99e-6 p.u. |
| events settling > 1e-5 | 2,236 (5.4 %) |
| events settling > 1e-4 | **3 (0.007 %)** |

**The original 1.44e-5 figure came from a 400-event sample and missed the tail.** Three events —
`55_BC_1p0_0007` (7.09e-3), `7_CA_10p0_0002` (6.46e-3), `51_ABC_1p0_0000` (6.37e-3) — settle
~300× slower than the bulk. All three are **ungrounded** faults (BC, CA, ABC), and in each the
voltage *recovers* between steps 40 and 41 (1.0137 → 1.0208 p.u. at node 16 of `55_BC_1p0_0007`)
rather than settling downward. That is a **control device — regulator tap or cap bank — operating
in response to the fault**, not a numerical artefact, and it is almost certainly the same
mechanism as the 3.3 % non-monotone sag cells recorded under "Smaller findings" below. One
control-state decision at S1.3 covers both.

This was found by the S0.3 assertion the author required, on its first run. The sampled statistic
was not wrong; it was sampled.

Consequences:
- The last-timestep observation of `01_SPEC.md` §3 is exact for windows w ≥ 23 and carries a
  ≤ 2e-5 p.u. error for w = 21 and w = 22 — still orders below any usable σ_ε — **except on those
  three events, where it is up to 7e-3 p.u. off**, which is not comfortably below σ_ε. That is
  6 windows out of 1.65 M: too rare to affect training or any reported metric, and now enumerated
  in `SETTLING_OUTLIERS` so a fourth fails the assertion instead of being averaged in.
- **The table must not be built from a single fault solve.** It would sit on the unsettled level
  while almost every observation sits on the settled one: a small, systematic, location-dependent
  bias that S1.4 would misread as operator error. `physics/signature_solver.py` now takes
  `n_fault_steps` (default **3**, converged to 2.8e-8) and `n_prefault_steps` (default 2).
- `audit_window_census.py` asserts all of this (`--n-quasi-static 0` scans the full corpus in
  15 s; the default 100-event sample will not see the outliers).

**Unlabelable-window ceiling (new).** The w = 20 windows carry a fault label with zero
post-inception samples, so they are physically identical to a healthy window and their content
depends only on L, which is drawn independently of the fault location.

- They are **6,175 windows = 2.50 %** of the test split.
- **Macro-F1 ceiling = 0.9744** for a model that is perfect everywhere else and predicts no-fault
  on them (0.9546 if it guesses uniformly instead). Every absolute macro-F1 in **both** papers
  sits under this.
- **No leakage anywhere.** Across all 33 runs (5 standard × 3 seeds, 5 dropout-aware × 3 seeds,
  SDR-STGNN × 3), accuracy on that subset ranges 0.0003–**0.0418** against a chance rate of
  **0.0385**. Nothing is above chance. This is a clean confirmation, not a near miss.
- Excluding them raises RGSAGE-Max seed 42 from 0.8526 to 0.8862, i.e. the unlabelable windows
  cost the headline model ≈ **3.4 macro-F1 points**.

## S0.4 — Snapshot/Daily equivalence — the decisive one

Five events spanning 0.1/1/10 Ω and five fault types, each re-solved in three modes at four
settling depths, against the corpus's own recorded post-fault level.

| mode | 1 step | 2 steps | 3 steps | 5 steps |
|---|---|---|---|---|
| `daily` (recompile each solve) | 5.71e-06 | 9.55e-08 | 1.46e-09 | **1.02e-12** |
| `reuse` (keep the circuit) | 5.71e-06 | 9.55e-08 | 1.46e-09 | **1.02e-12** |
| `snapshot` (Mode 0) | 5.71e-06 | 9.55e-08 | 1.46e-09 | **1.02e-12** |

Worst |V_solve − V_corpus| over instrumented channels, p.u. Tolerance was 1e-3.

- **Freeze assertion passes empirically.** At `L × 0.99953` the error is 1.02e-12 p.u.; at
  `L × 1.0` it is **4.80e-05 p.u.** — seven orders of magnitude worse. The frozen pre-fault
  modulation is confirmed by measurement, not by reading the code.
- **`reuse` mode is identical to `daily` on these five events — and that conclusion does NOT
  generalise.** `[Falsified 2026-09-23; see "The build path" below.]` Measured against 1,000
  events instead of 5, the standing-circuit path disagrees with the corpus by > 1e-3 p.u. on
  **43.3 %** of them. The table must be built with a recompile per solve.
- **Snapshot mode is also identical**, so the `01_SPEC.md` §2 wording ("built in OpenDSS snapshot
  mode") and the plan's Daily-path decision turn out to be equivalent here. No branch needed.

## S0.5 — Timing

| quantity | measured |
|---|---|
| signature solve (`reuse`, 3 fault steps) | **2.49 ms** → **402/s** single-core |
| signature solve (`daily`, recompile each time) | 7.3 ms → 137/s |
| full 60-step Daily-mode event (the OOD unit) | **17 ms** |

The plan assumed ~45 solves/s; the measured rate is **9x** that, because the estimate was costing
the *compile*, which `reuse` mode pays once.

| item | planned | measured (serial) |
|---|---|---|
| signature table, L step 0.05 (70,142 solves) | 25–30 min | **2.9 min** |
| signature table, L step 0.025 (136,158 solves) | "< 1 h" | **5.6 min** |
| OOD pilot (1,100 events) | ≈ 25 min | **≈ 20 s** |
| OOD sealed set (11,000 events) | **4–5 h** | **≈ 3 min** |

**Consequence for S1.2:** both candidate L steps cost minutes, so the grid step must be chosen
purely on interpolation error — there is no budget argument for the coarser grid. The GPU budget
is unchanged (not re-measured; `results/runtime/runtime_training.csv` stands), so the reduced-seed
rule still applies.

## The build path — found while investigating the non-monotone cells `[Added 2026-09-23]`

The author asked for the S1.3 non-monotone cells to be characterised before the full table build.
That investigation turned up something larger, and it is the most consequential finding of the day.

### `reuse` mode silently corrupts the operator

`physics/table.build_table` defaulted to `mode="reuse", reuse_limit=500` — a standing circuit
reused across 500 solves. OpenDSS carries **regulator tap positions** across solves on a standing
circuit, so each hypothesis inherits the tap state left by whichever hypothesis the worker solved
before it. Measured against 1,000 corpus events, each solved at its own stored (f, τ, R, L) and
compared with the corpus's own settled post-fault level:

| build path | median error | p95 | events > 1e-3 p.u. | ms/solve |
|---|---|---|---|---|
| `reuse`, never recompile | 4.9e-09 | 1.3e-02 | **43.3 %** | 4.4 |
| recompile per solve (`daily`/`snapshot`/`reuse_limit=1`) | **1.0e-10** | 6.2e-09 | **1.8 %** | 8.1 |

The error distribution is **bimodal** — either exact to 1e-10, or ≈ 6.5e-3 p.u., which is one
regulator tap step. That is a discrete state difference, not numerical noise. The corpus
recompiles per event (`simulation/runner.py`), so recompiling is what *matches the generator*.

**S0.4 missed this because it sampled 5 events**, all of which happened to agree. The fast path it
validated was the wrong one. `build_table` now defaults to `mode="daily", reuse_limit=1`; the
table costs ~9 minutes serial instead of ~3, which is not a trade worth having an opinion about.

### The residual 1.8 % is real, physical, and concentrated where it hurts

With the corrected path, 18 of 1,000 events still differ by 6.4–7.3e-3 p.u., and **16 of the 18
sit at R = 10 Ω** — the hard case the whole paper turns on. This is the corpus's own tap
hysteresis: its tap position at inception depends on the 40-step Daily history that a fresh
compile does not reproduce. Two consequences:

- It is **the one genuinely physical contribution to σ_ε** that this corpus offers, and it is
  ~6.5e-3 p.u. — far above the discretisation floor and of the same order as the 10 Ω sag itself.
  S1.6 must calibrate with it in view, and S1.4 must report it by R.
- It **slightly softens the degeneracy ruling, in the one place that matters**. The operator is
  the exact generator for 98.2 % of events, so S2.0ᵃ stands unchanged as the project's framing —
  but at 10 Ω there is a genuine, measurable model error, which is where the physics/learning
  contrast is most interesting. Do not oversell this: it is 1.8 % of events, and it is not a
  rescue. It is a footnote to S1.4 and a candidate line in the limitations section.

### The non-monotone cells: real, but NOT control-driven

On the real 15-point R grid (12,378 solves, 0 non-converged), **9.4 % of (f, τ, L) cells** have a
sag depth that does not decrease monotonically with R — not the 3.3 % the smoke grid suggested —
with violations up to **4.1e-2 p.u.** They cluster hard by fault type: **904 of 1,084 violations
are ungrounded** (AB 264, BC 277, CA 304, ABC 59) against 15–47 for each grounded type, although
ungrounded types are only 4 of 11. Every bus is affected; violations concentrate at both ends of
the R grid.

**The corpus shows the same behaviour.** Comparing corpus median sag against table sag at the
corpus's own R ∈ {0.1, 1, 10}: the two agree on monotonicity for **273 of 275 (bus, type) cells**,
and **all 17 corpus non-monotone cells are ungrounded**. Per the author's rule — if the corpus
shows it, fidelity to the corpus wins — **the controls stay exactly as they are** and the fraction
is reported in T3.

**But the documented cause is wrong.** `03_STAGE_B_operator.md` S1.3 says a non-monotonic run
"almost always means a regulator or cap-bank control interacted with the fault". Re-solving nine
non-monotone cells with `ControlMode=OFF`: **4 of 9 remain non-monotone** (buses 80 and 89, types
AB and CA). So for this feeder the effect is largely **intrinsic to ungrounded faults**, not a
control artefact. Disabling controls does move the sags a lot (up to 0.168 p.u.), which is a
second reason to leave them alone.

**Correction to an earlier claim in this note.** The S0.3 write-up above suggested the three
settling outliers and the non-monotone cells were "almost certainly the same mechanism". Tested:
they are not. `55_BC_1p0_0007`'s slow settling **is** control-driven (7.09e-3 p.u. with controls
STATIC, 1.4e-5 with them off), but the outlier cells are monotone in R, and most non-monotone
cells survive controls being disabled. Two distinct phenomena that share only "ungrounded fault".

### What this changes for S1.7

At cells where sag is non-monotone in R, **argmin_R is ambiguous by construction**. S1.7's
"argmin_R within one grid step" check must report those cells separately rather than averaging
them in, or it will read an identifiability property as a grid-density failure.

## Decisions taken — author ruling, 2026-09-23

All three open items are closed. A fourth consequence, larger than any of them, was found in the
same pass and is recorded first.

### 0. The operator is the generator — this is the ruling that matters

S0.4's 1.02e-12 p.u. is not a calibration inconvenience. It makes **every clean, in-distribution
comparison in the project degenerate**: S1.4's bias check returns ~0 by construction, PSM at
k = 0 is an oracle rather than a baseline, the OOD-resistance contrast is degenerate for PSM too
(the OOD sets come from the same generator), and — worst — the fusion cannot learn anything on
clean data, because ρ_true = 0 makes `softmax(−ρ)` one-hot, β learns to trust physics completely
and PC-STGNN becomes PSM with extra steps. No choice of σ_ε fixes that; compressing the scale
shrinks true and wrong residuals together.

What survives is what the paper is now about: **model mismatch, measurement noise, silent
corruption, and identifiability** — in each of these the operator is not the generator of what
the model sees. Three changes follow, all applied to the spec today:

1. **Noise augmentation is the default training condition** for PC-STGNN; noise-free is the
   ablation (`06_STAGE_E_pcstgnn.md` S4.3ᵇ). This forces a new paired reference — **R0ⁿ**, the
   dropout-aware recipe plus noise augmentation and no physics — because comparing a
   noise-augmented R5 against the existing noise-free R0 would confound the two. +42 GPU-h.
2. **The nominal table is an explicitly labelled oracle column**: headline tables lead with the
   realistic operating point (noise + perturbed table) and carry the exact-operator column beside
   it, labelled "oracle upper bound: perfect model knowledge, perfect measurements".
3. **Identifiability is promoted to a contribution** — with an exact operator, D(f₁,f₂;A)
   measures the pure information content of 25 sparse magnitude-only µPMUs with zero model error
   mixed in. It is the one result that survives every Gate-2 branch.

Full ruling: `04_STAGE_C_gonogo.md` S2.0ᵃ.

### 1. S0.1 — resolved by re-evaluation, not by tolerance

Accepting a ~1e-3 drift would import an error bar into every paired comparison Paper 2 makes.
Instead **all 33 Paper-1 arms are re-evaluated fresh inside Paper 2's pipeline** and those
numbers — not the stored CSVs — are the reference for every paired comparison, which is then
internally consistent by construction. Paper 1's published values are cited separately with a
one-line footnote on mask-realisation tolerance; no published number moves. `batch_size` is now
recorded in every curve CSV (`run_dropout_study.py`), so this is never rediscovered.

### 2. σ_ε — source changed, and "finer is better" rejected

`01_SPEC.md` §4 now sets σ_ε from the **grid-discretisation error measured in S1.2**, not from
S0.4. Two amendments came with the ruling:

- **On the nominal table σ_ε is a numerical quantity, not a physical one**, and the paper must not
  imply otherwise. It carries a **stated physical floor** (µPMU accuracy class, IEEE C37.118 1 %
  TVE) so the residual scale still means something outside the oracle setting.
- **The L grid is chosen for usable ρ dynamic range, not maximum fidelity.** A finer grid drives
  ρ_true → 0, widens the top-2 margin and makes the physics prior more one-hot — *more*
  degeneracy, not less. S1.2 selects on the ρ-margin distribution; the earlier "take 0.025 unless
  0.05 passes cleanly" is withdrawn (`03_STAGE_B_operator.md` S1.2).

### 3. S0.3 — restated, asserted, and the restatement then failed on first run

Restated as "constant to < 1e-6 p.u. from step 42 onward; the first two post-inception steps carry
**solver settling**" — and called settling, never a "transient", since this project does not claim
physics-informed temporal dynamics.

Per the ruling it became an **assertion in `audit_window_census.py`, not prose**. It fired
immediately: the bound derived from 400 events (1.4e-5) does not hold on the full corpus. The
re-measurement over all 41,250 events, the three control-driven outliers it exposed, and their
link to the S1.3 non-monotone sag cells are written up in S0.3 above. The sampled statistic was
not wrong; it was sampled — which is the argument for executable checks in one line.

## Smaller findings, recorded so they are not rediscovered

- **`R_GRID` excludes the corpus's own {0.1, 1, 10} Ω.** `np.geomspace(0.05, 30, 15)` gives
  0.0790, 0.1247, …, 1.2247, …, 7.6173, 12.0294 — the training resistances are strictly between
  grid points. This follows `01_SPEC.md` §2 literally and it is the right choice (a grid built
  around the training values would make S1.7's "where does argmin_R land" test vacuous), but it
  should be a conscious choice, not a surprise at S1.7.
- **Monotonic-sag check is not clean.** On a coarse 3×3 smoke grid, 3.3 % of (f, τ, L) cells have
  sag depth that does not decrease monotonically with R. S1.3 must look at this on the real grid;
  `physics/table.check_monotonic_sag` reports it. The usual cause is a regulator or cap-bank
  control interacting with the fault.
- **Two latent repo bugs fixed on the way** (both would have hit Stage A/B scripts):
  - `simulation/feeder.compile_feeder` left the *process* working directory inside
    `fault_localization/data/ieee123/`, because OpenDSS's `Compile` chdir's there and never
    returns. Any script that compiled the feeder and then wrote to a relative path would have
    written into the DSS directory. It also made `tests/test_pignn.py` fail whenever
    `tests/test_feeder.py` ran first — the full suite was **3 failed / 210 passed** before the
    fix and is **229 passed** after. The Redirects inside the master file all execute during the
    call, so restoring the caller's directory cannot affect the circuit.
  - `simulation.simulate_fault_event` hard-coded `deterministic_seed(...)` into its metadata,
    which **raises** for any resistance outside {0.1, 1, 10} — i.e. for every OOD event Stage C
    needs. It now takes an optional `seed`; the default path is unchanged and the five
    bit-identical regenerations were re-verified after the change.

## What exists now

- `fault_localization/physics/` — `grids`, `signature_solver`, `table`, `observation`, `residual`,
  `identifiability`, `cache`.
- `fault_localization/dropout/silent.py` — the four silent-failure injectors.
- `fault_localization/models/pc_stgnn.py` + `models/common.pc_loss` — PC-STGNN and its objective.
- `config/constants.deterministic_seed_ood` — disjoint OOD seeds.
- Tests: `test_physics_residual` (13), `test_signature_solver` (9), `test_silent` (31),
  `test_ood_seed` (8), `test_pc_stgnn` (16). Full suite **229 passed**.

No GPU training has been launched.
