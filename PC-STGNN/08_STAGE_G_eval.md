# Stage G — Full Evaluation and Robustness (Weeks 15–19)

**Purpose.** Produce every results table, and answer the leakage objection with an experiment.
**Compute.** Mostly inference — minutes per model per condition. Plus ~1.5 h CPU and one ≈ 7 GPU-h
retrain.

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendment G7 below; the line above is kept.]** The retrain costs **33.6 GPU-h**, not ≈ 7 GPU-h. A full
> R5 scoring grid is 7.44 GPU-h per seed, not minutes (`stage_g_plan_draft.md` § 5).

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
| OOD resistance | {3,5,20,30} Ω × k ∈ {0,5,10} — open the sealed set here, once. **Report split by grounded vs ungrounded** (see below) | new |

**Models.** B1, B1ᵃ (matched filter), B2 (PSM), logreg/RF, both impute-then-infer rows, all 5
baselines standard, all 5 dropout-aware, SDR-STGNN, R1–R7, B2ᵃ, G0–G3. `[Corrected 2026-09-22]`
Only the new models need the existing conditions re-run; the Paper-1 numbers are reused as-is
(same mask seeds).

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendments G1–G5 below; the Models paragraph is kept.]** "R1–R7": R7 is dropped; R3, R4, R5-γ0, R6ⁱ, R8 exist.
> "G0–G3": not built, replaced by G-TF2. R6 (trained) is replaced by R6ⁱ. B2ᵃ settings: 14, Amendment 5. The OOD cells and the
> pair are fixed in Amendments G2–G3.

**Check.** Before opening the sealed OOD set, verify its hash against `data/ood_test.sha256`.
Record the date it was opened in `14_PREREGISTRATION.md`.

**Split the OOD-resistance results by grounded vs ungrounded — pre-registered, not post-hoc.**
`[Added 2026-09-23]` The extrapolation claim rests on the operator holding a dense, continuous
representation of R. Where the sag is **non-monotone in R** — 9.4 % of grid cells, of which 83 %
are ungrounded (`03_STAGE_B_operator.md` S1.3) — a dense grid does **not** guarantee that the
profiled argmin_R lands near the truth, because the objective along that axis has more than one
basin. So the physics advantage on unseen R may be concentrated in grounded faults.

If it is, that is an honest refinement that *strengthens* the paper: it says where model-based
extrapolation works and where the axis itself is ambiguous. Discovering it in review instead
would not. Register the split in `14_PREREGISTRATION.md` before the sealed set is opened, so it
is a pre-planned stratification rather than a subgroup found after the fact.

> **[Done 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendment G4 below.]** The grounded / ungrounded split is registered in `14_PREREGISTRATION.md` (Amendment 4) before the
> sealed set was opened. The paragraph above is kept unchanged.

**Tables produced.** T5 (all models × k) · T6 (baseline ladder) · T8 (OOD) · T9 (silent failures)
· T10 (structured + noise).
**Figures produced.** F-G1 (F1 vs k, all models) · F-G2 (F1 by resistance) · F-G3 (OOD
extrapolation — the headline) · F-G4 (noise robustness) · F-G5 (structured vs random vs
persistent dropout).

## S6.2 — Model-mismatch study — **a primary results axis** `[Promoted 2026-09-23]`

**This is no longer a robustness appendix.** S0.4 measured the nominal table against the corpus at
1.02e-12 p.u.: the operator *is* the generator, so every clean, in-distribution comparison in the
project is an oracle reference and cannot carry the contribution
(`04_STAGE_C_gonogo.md` S2.0ᵃ). A perturbed table is not the generator — which makes this study,
alongside measurement noise, the place where the real evidence lives. It is a main-body axis with
a main-body figure (F-G6), reported beside the oracle column, never in place of it.

The original framing, which still holds:

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
| **Regulator tap position off by ±1 step** `[Added 2026-09-23 — lead with this one]` | ±1 tap on each regulator |
| Line impedances, random per segment | ±2 %, ±5 %, ±10 % |
| Load composition (constant-P vs a ZIP mix) | one alternative |
| One unmodelled lateral, or a small unmodelled DER | one scenario |

**The tap perturbation is the strongest scenario in this table, and it is not synthetic.** It is a
model error **measured in this corpus**: with the build path corrected, 1.8 % of events still
differ from the table by **6.4–7.3e-3 p.u.**, which is one regulator tap step, because the
corpus's tap position at fault inception depends on 40 steps of Daily history that a fresh
compile does not reproduce (`03_STAGE_B_operator.md` S1.4, `scripts/audit_build_path.py`).

That makes it the *realistic* model error for this feeder rather than an invented one:
regulators carry memory, and **no model-based operator can know the tap state unless it is
telemetered** — which on a real feeder it usually is not. A power-systems reviewer recognises
this immediately, in a way that "line impedances perturbed by ±5 %" does not. It also lands where
the contrast is most interesting: **16 of the 18 measured cases sit at R = 10 Ω**. Point at the
18 corpus events by name when reporting it.

**Check.** Gain over R0 as a function of mismatch level, for PSM (B2), fusion-only (S2.6 on R0)
and R5.

**Cost.** ≈ 30 min of solves per perturbed table (70 k solves). The fusion-only variant needs no
retraining. One retrain of R5 at ±5 % confirms the trained variant behaves the same way
(≈ 7 GPU-h).

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendments G6 and G7 below.]** Order: an inference-only test with three tables first (Phase 1), then **one**
> retrain (R5, impedance ±5 %), costing **33.6 GPU-h**. The fusion-only variant is on R0ⁿ. The operator R axis has 18 points and 84,167 solves.

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

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendments G1, G2, G5 below.]** The pairs are R5 vs R0ⁿ (for H3), R5 vs R6ⁱ, R5 vs R2. "G2 vs G0" is removed.
> For H3 and Gate 6 the bootstrap is over the events of the sealed set (11,000 expected).

**Interpretation.** With 6,175 test events the bootstrap resolves small differences easily. Seed
variance, not significance, is the binding constraint. Report both and lead with the seed spread.

**Table produced.** T13 — statistical tests summary.

## S6.4 — Computational cost accounting

**Do.** For every model record: parameter count, FLOPs per window, training GPU-hours, inference
latency per window (including the residual and, for G2, the LOO pass and second forward). Extend
`scripts/run_runtime_benchmark.py` (already measures the Paper-1 models on synthetic
`[B,25,20,6]` batches).

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendment G5 below.]** There is no LOO pass: the in-sample residual replaced it. The silent-failure row is G-TF2
> (in-sample ρ plus the flatline score).

**Why.** Reviewers in this area ask. It also lets you state plainly that the physics path adds
three scalars and ≈ 1.3× forward cost for gate variants only.

**Table produced.** T12 — computational cost.

## Amendments

All entries: **Amended 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z, before the sealed set was opened.** `data/ood_test/` has not been opened, read, windowed or scored. The rules are
stated in full in `14_PREREGISTRATION.md` (Amendments 1–8); this section gives the Stage G consequences. Old text above is kept and marked "Replaced". Line numbers cited in this section ("lines 30–40", "08:96-98") are those of commit `de00c24`, before these amendments shifted the file.

### G1 — Gate 6
Gate 6 passes if the H3 rule passes: the sign of F1(R5) − F1(R0ⁿ) is positive in all 3 seeds **and** the 95 % CI of the event-level paired cluster
bootstrap over the sealed set's events excludes zero (14, Amendment 1, with the exact statistic). Stage G ends with Gate 6.

### G2 — H3 pairing
R5 against R0ⁿ. R0 is a descriptive row. The pair "R5 vs R0" of S6.3 is replaced for H3.

### G3 — H3 measure and the OOD cells (S6.1, OOD row)
Primary: macro-F1 pooled over R ∈ {3, 5, 20, 30} Ω, k = 0, at each σ ∈ {5e-4, 2e-3, 1e-2}; H3 passes only if the rule holds at all three σ. Descriptive:
F1 for each R, at σ = 0, and k ∈ {5, 10} at σ = 0. Scored cells: k = 0 × σ ∈ {0, 5e-4, 2e-3, 1e-2} and k ∈ {5, 10} × σ = 0, 3 offsets each.

### G4 — grounded / ungrounded split (lines 30–40)
Registered in 14, Amendment 4: grounded = AG BG CG ABG BCG CAG ABCG; ungrounded = AB BC CA ABC; descriptive, no rule; pre-planned.

### G5 — settings changed by Stages E and F
R5 = λ₁ = 0, λ₂ = 0.1, γ = 1. In-sample residual replaces LOO (no LOO pass in S6.4). R6 → R6ⁱ (A1). R7 dropped (A3 not tested; answered by the parameter
count, 95,259 vs 95,258). G0–G3 → G-TF2 (H5 and A5 not tested). Holm families {H1–H4}, {A1, A2}. B2ᵃ settings: 14, Amendment 5.

### G6 — model-mismatch study (S6.2)
Phase 1 first: inference only, R5 ×3 seeds, tables tap +1, tap −1, impedance ±5 %, 8 cells on validation; graceful if R5 > R0ⁿ in all 3 seeds in the 7 H2/H4
cells. CPU pre-check of the true-class residual (320 validation windows, median reported). Then **one** retrain: R5 on the impedance ±5 % table, seed 42,
epoch-3 collapse guard, after the pre-check and the author's GPU go-ahead. Full text: 14, Amendment 6.

### G7 — corrections
Retrain cost **33.6 GPU-h** (not ≈ 7). Retrain count **1**. Operator R axis **18 points**, 84,167 solves (not 15 points, 70 k solves).

### G8 — operator grid check (R values of the sealed set against the operator R axis)
Checked on 2026-11-02 against `operator/grid.json` and `physics/grids.py::R_GRID` (18 points: 0.05, 0.0790, 0.1, 0.1247, 0.1969, 0.3110, 0.4911, 0.7755, 1.0,
1.2247, 1.9341, 3.0544, 4.8235, 7.6173, 10.0, 12.0294, 18.9969, 30.0 Ω). The four values come from `scripts/generate_ood_sets.py` (`SET_SPECS["test"]`), not from
the sealed data.

| R (Ω) | On the operator R axis? | Nearest grid point (Ω) | Distance |
|---|---|---|---|
| 3 | **no** | 3.0544 | +1.8 % |
| 5 | **no** | 4.8235 | −3.5 % |
| 20 | **no** | 18.9969 | −5.0 % |
| 30 | **yes** (the upper end point of the axis) | 30.0 | 0 |

**30 Ω: PSM evaluates this resistance exactly. The OOD test is out of distribution for the network only.** (The R axis is exact there; the load axis
L stays on its 0.05 grid, so the residual is not zero at the true class.) For 3, 5 and 20 Ω the value is **between** grid points: PSM uses the nearest
point, within 1.8–5.0 %, so it is approximate there, and the sentence above does not apply to them. 30 Ω is also the edge of the axis: no point lies above it.

### G9 — mismatch retrain: decision (approved by the author)
**Amended 2026-11-03 18:47 IST / 2026-11-03T13:17:11Z, before the sealed set was opened.** The retrain of S6.2 is **R5 at impedance ±5 %**, one seed (42), approved by the author. Reason: the
CPU pre-check of the true-class residual (`results/stage_g/mismatch_tables/precheck.json`; 320 validation fault windows, k = 0, σ = 0) shows a residual
much smaller than the one that broke the shuffled-table training. For the ±5 % table the half-residual at the true class has median 422 and mean 670
(nominal table: 28 and 73), and the shifted ρ at the true class has mean 5.5 (nominal 4.2). The R6 shuffled table gave a shifted ρ with median 1.14e4
and mean 2.03e4 (`06_STAGE_E_pcstgnn.md`, R6 record). So the ±5 % residual is about 20 times smaller than the R6 residual or more: 27 times against the
R6 median, and about 3,700 times on the shifted mean (different window samples; two forms of the residual). PSM accuracy on the 320 windows stays
**0.903** (nominal 0.912). **The epoch-3 collapse guard stays** (network-only macro-F1 ≤ 0.10 or class-0 share > 0.95: stop, record as collapsed).
The decision approves the retrain. The run starts only through the Stage G queue, which does not start without the file
`results/stage_g/STAGE_G_APPROVED`.


### G10 — Gate 6 result (sealed OOD set opened once, 2026-11-18)
**Recorded 2026-11-19.** The sealed set was opened once (2026-11-18 10:51 IST, hash verified) and 10 frozen models were scored. **Gate 6 passed**: at each σ ∈ {5e-4, 2e-3, 1e-2} (k = 0, pooled over
R ∈ {3, 5, 20, 30} Ω) F1(R5) − F1(R0ⁿ) is positive in all 3 seeds and the event-bootstrap CI lower bound is above 0 (Δ +0.424, +0.226, +0.026). H3 passes as written. R0ⁿ is near chance on the set
(0.065 to 0.086). Descriptively, R5 is below the physics-only predictor PSM at every σ, and its network branch alone scores 0.090; the ungrounded stratum is not lower than the grounded one.
R0 was not in the frozen model list and has no sealed row. Numbers, sources and the descriptive rows: `stage_g_gate6_record.md`; tables: `results/stage_g/tables/T8_ood.md`.
