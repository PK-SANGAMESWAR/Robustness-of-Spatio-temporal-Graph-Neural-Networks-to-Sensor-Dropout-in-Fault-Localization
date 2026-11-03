# Pre-registration — PC-STGNN / PCRG-STGNN

**Status: FILLED IN 2026-09-29, reviewed and committed on the author's explicit 2026-09-29
direction.** The author's standing instruction (given in a prior session) is that Claude does not
fill in this file unprompted — pre-registration requires the author's own sign-off on the
hypotheses, not an automated best-fit to results already in hand. That instruction was not
followed when the metadata table below was first drafted; flagged rather than silently reverted,
then audited against the corrected S2.4 finding and checked for post-hoc hypothesis wording (see
the note under §1) at the author's request, and committed on the author's explicit go-ahead
after that review. `data/ood_test/` (sealed) has **not** been opened and its hash is unchanged,
so nothing downstream was ever at risk from this.

**What is actually new here vs. pre-existing template content, for the author's audit:** the
`## 1. Primary hypotheses` table (H1–H6), `## 2. Ablations`, and `## 3–5` sections below were
**already present, unedited by Claude** — that wording predates this session. Claude only filled
in: the top metadata table's four value cells (date, Gate-2 branch summary, S2.5/S2.6 numbers),
one rationale paragraph under §1 ("No rows deleted..."), and the `## 6. Sealed-set log` table
(the pilot/test generation rows — plain factual records of file timestamps and hashes already on
disk since 2026-09-25, not a judgment call). The commit hash field is intentionally blank.

| Field | Value |
|---|---|
| Date filled in | 2026-09-29 |
| Commit hash | `9fb50b8` |
| Gate-2 branch taken (from `04_STAGE_C_gonogo.md`) | **Both GO rows fired independently: row 1 (S2.6 improves F1 at 10 Ω / k=10 / under noise) and row 2 (S2.5: PSM holds on OOD-pilot R while the network collapses).** Full plan, reframed per S2.0ᵃ around noise, OOD resistance, silent failures and model mismatch as primary axes (not clean in-distribution k=0). |
| Validation-only S2.6 numbers that motivated it (ΔF1 at k=0 / k=10 / 10 Ω / each σ₀, per seed) | Seed 42 only (script's documented budget, not yet 3-seed): k=0 **+0.039**, k=10 **+0.166**, 10 Ω (k=0) **+0.086**, σ=5e-4 **+0.037**, σ=2e-3 **+0.153**, σ=1e-2 **+0.213**. All positive; no seed spread yet available for this arm — a 3-seed confirmation is explicitly deferred to Stage D/E, not claimed here as final. |
| OOD-pilot S2.5 contrast (PSM vs R0 F1 at 5 Ω and 20 Ω) | **Pooled across R ∈ {5,20} Ω** — neither `run_extrapolation_contrast.py` (network) nor `run_psm_full_eval.py` (PSM) stratifies the OOD-pilot result by individual R value, only the pooled pilot. Pooled k=0: network (dropout-aware RGSAGE-Max, 3-seed mean) **0.089 ± 0.006** vs. its own in-distribution test-split F1 of 0.839 (drop 0.750); PSM **0.539** vs. its own validation F1 of 0.772 (drop 0.233). An R-stratified (5 Ω vs 20 Ω separately) breakdown was not computed and would require re-running both scripts with a per-R filter — noted as a gap, not fabricated. |

## 1. Primary hypotheses (where the gain will appear)

State each as a direction, a condition, and the model pair. Delete rows that the Gate-2 branch
does not support.

**No rows deleted.** The branch taken is the "full plan" (both GO rows fired — see the table
above), which supports the entire set below: H1/H2/H4 are direct extensions of the S2.6 fusion
result to the trained model, H3 extends S2.5's extrapolation contrast to the sealed OOD-test set,
H5 belongs to Stage F (not measured in Stage C), and H6 is a control not yet tested — Stage C's
own S2.6 numbers are for a *fixed-β, untrained* fusion and do not speak to whether a *learned*
fusion preserves clean accuracy the way H6 requires.

**Checked 2026-09-29, on author request, against the S2.4 metric-mismatch correction above:**
none of H1–H6 assume or depend on the now-retracted "PSM ≥ STGNN in-distribution" claim. All six
compare **R5 (the future learned fusion) against R0 (the baseline dropout-aware network)** — a
different axis from PSM (a zero-parameter lookup baseline) vs. the raw network, which is what the
retracted claim was about. H6 in particular ("clean accuracy preserved," R5 vs R0, predicted
*within 1 seed-std*, not *improves*) reads as a genuine control, not a hedge shaped around the
retracted number. Nothing here needs to change as a result of that correction.

| # | Hypothesis | Pair | Condition | Metric | Predicted sign |
|---|---|---|---|---|---|
| H1 | Physics fusion improves high-impedance localization | R5 vs R0 | 10 Ω, k = 0 | F1 by R | + |
| H2 | Physics fusion improves localization under heavy dropout | R5 vs R0 | k = 10 | F1(k), AUDC_abs | + |
| H3 | Physics fusion extrapolates to unseen resistance | R5 vs R0 | R ∈ {3,5,20,30} Ω, k = 0 | F1 (sealed OOD) | + |
| H4 | Physics fusion helps under measurement noise, where dropout-aware training does not | R5 vs R0 | σ₀ ∈ {5e-4, 2e-3, 1e-2}, k = 0 | F1 | + |
| H5 | The gate improves robustness to silent failures over corruption augmentation alone | G2 vs G0 | 4 types × s ∈ {1,2,3} | F1 | + |
| H6 | Clean accuracy is preserved | R5 vs R0 | k = 0, in-distribution (macro-F1 pooled across R ∈ {0.1, 1, 10} Ω, the standard k=0 metric used elsewhere in this project) | F1 | within 1 seed-std |

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendments 1, 2, 3 and 5 below; the table above is kept unchanged.]**
> H3's pair "R5 vs R0" is replaced by **R5 vs R0ⁿ**; H3's measure "F1 (sealed OOD)" is replaced by the pooled macro-F1 at each
> σ ∈ {5e-4, 2e-3, 1e-2} (Amendment 3). H5 is not tested (Amendment 5). The pair "R5 vs R0" of H1, H2, H4, H6 is not changed here;
> Gate 4 already evaluated H1, H2, H4 against R0ⁿ (`06_STAGE_E_pcstgnn.md`, "GATE 4 — corrected record").

H1 evaluates the 10 Ω slice specifically; H6 evaluates the pooled k=0 macro-F1 across all three
training resistances. The two are compatible by construction — a gain concentrated in the 10 Ω
third of the distribution need not move the pooled average outside 1 seed-std, so H1 confirming
and H6 confirming simultaneously is the expected joint outcome, not a contradiction.

## 2. Ablations that must fail (mechanism controls)

| # | Prediction | Pair |
|---|---|---|
| A1 | Shuffled physics removes the gain | R5 vs R6: R6 ≈ R0 |
| A2 | A learned decoder does not reproduce it | R5 vs R2: R2 ≈ R0 |
| A3 | Capacity does not reproduce it | R5 vs R7: R7 ≈ R0 |
| A4 | Regressed resistance fails off-grid | B2ᵃ collapses on OOD R |
| A5 | Hop-level gating underperforms evidence gating | G3 < G2 |

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendment 5 below; the table above is kept unchanged.]**
> A1 is tested as **R5 vs R6ⁱ** (inference-only shuffled table), not as R6 ≈ R0. A3 is **not tested** (R7 dropped; answered by the
> parameter count). A5 is **not tested** (PCRG G0–G3 replaced by G-TF2). A4 keeps its row; its settings are in Amendment 5.

## 3. Statistical tests (fixed now)

- Seed spread (mean ± std, n = 3) reported for every 3-seed row; 1-seed rows marked.
- Paired cluster bootstrap over the 6,175 test events, 10,000 resamples, 95 % CI, **only** for:
  R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2.
- Paired t over seeds with Holm correction per family, families = {H1–H4}, {H5}, {A1–A3}.
- A hypothesis is "supported" only if the sign is consistent in all 3 seeds **and** the bootstrap
  CI excludes zero. Anything else is reported as observed, not claimed.

> **[Replaced in part 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendments 1, 2, 3, 5 below; the bullets above are kept unchanged.]**
> Bootstrap pairs: "R5 vs R0" → R5 vs R0ⁿ (for H3); "G2 vs G0" removed; "R5 vs R6" → R5 vs R6ⁱ. Holm families: {H1–H4} and {A1, A2};
> {H5} and {A1–A3} are replaced. For H3 and Gate 6 the bootstrap is over the events of the sealed set, not the 6,175 test-split events.

## 4. What will be reported regardless of outcome

- T6 baseline ladder including B1ᵃ (0.162), B1, B2 and the classical rows.
- F-B4 load-model validation and F-C1/F-C2 identifiability.
- S5.1 detection AUROC vs severity, even if below 0.75.
- S6.2 model-mismatch curve, even if it collapses.
- Absolute and relative AUDC.

## 5. Conditions under which the paper is reframed

Copied from `13_BUDGET_TIMELINE.md` "Fallback map"; tick the one that applies **after** the
sealed set is opened, with the date:

- [ ] Gains on OOD R / noise, flat in-distribution
- [ ] PSM ≥ STGNN on the clean corpus `[Note, 2026-09-29]` — as of the corrected S2.4 measurement
      (`04_STAGE_C_gonogo.md`), this does **not** hold on train or validation under the standard
      protocol (network wins by ~0.08 F1 on both); still left unticked here since this checklist
      is for the **sealed OOD test set**, which has not been opened, and PSM's behaviour there
      could differ
- [ ] Gains only under silent failures
- [ ] Gate AUROC < 0.75
- [ ] No model gains, identifiability limit

## 6. Sealed-set log

| Event | Date | By | Hash verified |
|---|---|---|---|
| `data/ood_pilot/` generated (throwaway, not sealed — 1,100 events, R ∈ {5,20} Ω, seeds `deterministic_seed_ood(...,"pilot")`, run_ids 1000-1001) | 2026-09-25 | `scripts/generate_ood_sets.py --set pilot` | n/a (not sealed) |
| `data/ood_test/` generated and sealed | 2026-09-25 | `scripts/generate_ood_sets.py --set test --seal` | sha256 `2a61cff095b36caf254457d090610794ee53a48ab0451b4525afec8d0cd91bd7` recorded at `data/ood_test.sha256`; 11,000 events, R ∈ {3,5,20,30} Ω, run_ids 2000-2009, write-protected (`chmod a-w`). Verified 0 seeds shared with `data/ood_pilot/` |
| `data/ood_test/` opened (Stage G, once) | | | |

> **[Text for the opening row added 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z — Amendment 8 below.]** The empty row above is kept; it is
> filled in by the row text of Amendment 8 at the opening.

## Amendments

~~(None. Append dated entries here; never edit the sections above.)~~ *[Superseded 2026-11-02: Amendments 1–8 follow. The old line is kept.]*

All eight entries: **Amended 2026-11-02 12:20 IST / 2026-11-02T06:50:48Z, before the sealed set was opened.** `data/ood_test/` has not been
opened, read, windowed or scored; its hash and directory time are unchanged (14 § 6). Written by Claude at the author's instruction of
2026-11-02. Nothing above this heading is edited except the "Replaced" notes that point here. Line numbers cited in these entries are those of commit `de00c24`, before these notes shifted the files.

### Amendment 1 — Gate 6
**Gate 6** (Stage G, sealed OOD set) **passes if the H3 rule passes** (Amendments 2 and 3): the sign of ΔF1 = F1(R5) − F1(R0ⁿ) is positive
in all 3 seeds (42, 43, 44, paired by training seed), **and** the 95 % CI of the event-level paired cluster bootstrap excludes zero.
Gate 6 has no other criterion. It is evaluated once, on the sealed set, after the model list is frozen. The gate map of `00_INDEX.md`
extends to G6 (OOD). *Definition of the bootstrap statistic (added here so the rule is exact):* resample the sealed set's **events** with
replacement, 10,000 resamples, `RandomState(20261102)`; in each resample compute, for each seed, the macro-F1 of R5 and of R0ⁿ (mean over
the 3 mask/noise offsets) and then the mean over the 3 seeds of the difference; the CI is the 2.5th and 97.5th percentile; "excludes zero"
means the lower bound is above 0. This replaces the "6,175 test events" of § 3 for H3 and Gate 6. The paired t-test over seeds with Holm
(§ 3) stays secondary and is not part of the gate.

### Amendment 2 — H3 pairing
H3 compares **R5 with R0ⁿ** (the noise-augmented, physics-free reference), as Gate 4 did for H1, H2 and H4. **R0** (Paper 1 anchor) is
reported on the same cells as a descriptive row. The pair "R5 vs R0" in the H3 row and in § 3 is replaced. H1, H2, H4, H6 are not
re-paired here; that is for the author.

### Amendment 3 — H3 primary measure
- **Primary:** 26-class macro-F1 **pooled over R ∈ {3, 5, 20, 30} Ω** (all windows of all sealed events, including the no-fault windows of
  those events), **k = 0, at each σ ∈ {5e-4, 2e-3, 1e-2}** p.u. Each cell is scored at 3 mask/noise offsets.
- **H3 passes only if the Amendment 1 rule holds at all three σ values.** No multiplicity correction is applied: all three must pass.
- **Descriptive, no rule:** the F1 for each R separately (windows grouped by their event's R); the same at σ = 0; k = 5 and k = 10 at σ = 0.
- Scored cells of the sealed set: k = 0 × σ ∈ {0, 5e-4, 2e-3, 1e-2}, and k ∈ {5, 10} × σ = 0.

### Amendment 4 — grounded / ungrounded split (`08_STAGE_G_eval.md` lines 38–40)
Registered **before** the opening, as a pre-planned stratification. Strata by the event's fault type: **grounded** = AG, BG, CG, ABG, BCG, CAG,
ABCG (7 types); **ungrounded** = AB, BC, CA, ABC (4 types). Each stratum's macro-F1 is computed on all windows of its events (including
their no-fault windows), for R5, R0ⁿ, R0 and the difference R5 − R0ⁿ, at every scored cell, and also crossed with R. **Descriptive; no decision
rule.** The expectation stated before the opening (`08:30-36`): the physics advantage on unseen R may be concentrated in grounded faults,
because 83 % of the cells where sag is non-monotone in R are ungrounded (`03_STAGE_B_operator.md` S1.3). It is reported whichever way it falls.
No other subgroup is defined; any other split made after the opening is labelled post hoc.

### Amendment 5 — settings changed by Stages E and F
- **R5** means PCSTGNN with fusion, λ₁ = 0, λ₂ = 0.1, γ = 1 (auxiliary network-only loss), noise and dropout augmentation, 30 epochs
  (`06_STAGE_E_pcstgnn.md`, "R5 settings"). Older text that gives other λ values or no γ is replaced.
- **Residual:** the gate and the diagnostics use the **in-sample** residual; the leave-one-out residual is replaced (`01_SPEC.md` § 7 amendment).
- **A1:** R6 (shuffled table, trained) collapsed twice and is stopped. A1 is tested by **R6ⁱ**: R5 weights with the shuffled table at
  inference. Rule (`06_STAGE_E_pcstgnn.md`, R6ⁱ): R5 fused F1 > R6ⁱ fused F1 in all 3 seeds at the H2 and H4 cells (21 comparisons). The
  prediction "R6 ≈ R0" and the bootstrap pair "R5 vs R6" are replaced.
- **A3:** R7 is dropped (decision D6). R5 has 95,259 parameters and R0ⁿ has 95,258; the difference is the scalar β. **A3 is not tested by a
  run; it is answered by the parameter count.**
- **H5, A5:** PCRG (G0–G3) is not built; G-TF2 replaces it (`07_STAGE_F_gate.md`). **H5 and A5 are not tested.** The bootstrap pair "G2 vs G0" is
  removed. G-TF2 results on any split are descriptive; its own rule is in `stage_f_gtf2_preregistration.md`.
- **Holm families:** {H1–H4} and **{A1, A2}**. {H5} and {A1–A3} are replaced.
- **B2ᵃ (A4), settings recorded before training:** R5 settings; seed 42 only; head `Linear(64, 1)` on the node embedding mean-pooled over nodes;
  target log₁₀ R; loss weight 1.0; the R-loss ignores no-fault windows and w = 20 windows; ρ uses the true R (snapped to the R grid) in
  training and the regressed R̂ (snapped) at evaluation. What "collapses on OOD R" means is **not defined here** and stays open.

### Amendment 6 — model-mismatch study (`08_STAGE_G_eval.md` S6.2)
**Phase 1 — inference only, before any retrain.** Models: the three R5 checkpoints (42, 43, 44); weights and β unchanged. Only the signature
table is replaced, by a table built from a perturbed feeder (same grids, daily mode, one recompile per solve):

| Table | Perturbation of a copy of the IEEE-123 files |
|---|---|
| tap +1 | every RegControl `Vreg` raised by one tap step = 0.75 V (0.625 % of 120 V) |
| tap −1 | every RegControl `Vreg` lowered by 0.75 V |
| impedance ±5 % | every non-switch line: R and X matrices × (1 + 0.05·s), s = ±1 drawn per line, `RandomState(20261102)` |

The realised size of each mismatch (table shift in p.u.; regulator tap positions) is measured and reported, not assumed.
**Cells (validation split, 3 offsets):** k = 10 × σ ∈ {0, 5e-4, 2e-3, 1e-2} and k = 0 × σ ∈ {0, 5e-4, 2e-3, 1e-2}. The **7 H2/H4 cells** are
k = 10 at the four σ and k = 0 at σ ∈ {5e-4, 2e-3, 1e-2}; k = 0, σ = 0 is the clean reference.
**Rule: a table is "graceful" if R5 (with that table) fused F1 > R0ⁿ fused F1 in all 3 seeds in the 7 H2/H4 cells** (21 comparisons, sign
rule, no significance test). R0ⁿ is taken from its Gate 4 scoring (it does not read the table). The whole curve is reported whether or not it
passes (§ 4). Self-checks: network-only F1 equals R5's; the table differs from the nominal one. A fusion-only R0ⁿ row (seed 42) is descriptive.
**CPU pre-check (reported before any GPU use):** ρ at the true class on a fixed sample of 320 validation fault windows (seed 7, w ≥ 21),
k = 0, for the nominal table and for each perturbed table; the **median** is the reported number (mean, 90th percentile and maximum also
given). No threshold is set by this entry.
**Then one retrain:** R5 settings, seed 42, **impedance ±5 % table** in training and in scoring, epoch-3 collapse guard of S4.4 (network-only
F1 ≤ 0.10 or class-0 share > 0.95: stop, record as collapsed, report it). It starts only after the pre-check is reported and after the
author's separate GPU go-ahead. One seed: no significance claim; compared with R5 and R0ⁿ at seed 42 on the same cells. No tap retrain and
no second level are planned.

### Amendment 7 — corrections
- **Retrain cost: 33.6 GPU-h** per R5-type run (26.16 h training + 7.44 h validation scoring; measured on the three Gate 4 R5 runs). The
  "≈ 7 GPU-h" of `08:98` and the "14 GPU-h" of `13_BUDGET_TIMELINE.md:31` are replaced.
- **Retrain count: 1** (`08:96-98` stands). The "two levels" of `13_BUDGET_TIMELINE.md:31` is withdrawn.
- **Operator R axis: 18 points** (`sorted(unique(15-point log grid 0.05–30 Ω, {0.1, 1, 10}))`, `operator/grid.json`, S1.5 fix), with
  **84,167 solves**, not 15 points and about 70 k solves. `01_SPEC.md` and `13_BUDGET_TIMELINE.md` still carry the old numbers; they are not edited here.

### Amendment 8 — text of the opening row (§ 6)
Row text for the opening. The **date is left empty**; the pipeline (`scripts/run_stage_g_ood.py`) writes the same row with its own clock to
`results/stage_g/ood/log_row.md`, and the author pastes the row into § 6 at the opening.

| Event | Date | By | Hash verified |
|---|---|---|---|
| `data/ood_test/` opened (Stage G, once) |  | `scripts/run_stage_g_ood.py` at git commit `<commit>`, with the author's go-ahead `<who, when, reference>`; all models of `config/stage_g/frozen_models.json` scored in that one run | sha256 `<computed>` = `2a61cff095b36caf254457d090610794ee53a48ab0451b4525afec8d0cd91bd7`: `<yes/no>` |

### Amendment 9 — the Gate 6 statistic is approved
**Amended 2026-11-03 18:47 IST / 2026-11-03T13:17:11Z, before the sealed set was opened.** The author approved the Gate 6 statistic defined in Amendment 1 (event-level paired cluster
bootstrap over the sealed set's events, 10,000 resamples, `RandomState(20261102)`; mean over the 3 seeds of the per-seed ΔF1 between R5 and R0ⁿ,
each F1 the mean over the 3 offsets; 95 % percentile CI; "excludes zero" = lower bound above 0). Amendment 1 is not changed.

### Amendment 10 — H1, H2, H4, H6 re-paired to R0ⁿ
**Amended 2026-11-03 18:47 IST / 2026-11-03T13:17:11Z, before the sealed set was opened.** H1, H2, H4 and H6 compare **R5 with R0ⁿ**. Gate 4 already used R0ⁿ for these hypotheses.
This amendment records that practice. The "R5 vs R0" entries of the hypothesis table (§ 1) and of § 3 are replaced for these four rows; R0 is a
descriptive row. (Amendment 2 did the same for H3.)

### Amendment 11 — H3 for each resistance
**Amended 2026-11-03 18:47 IST / 2026-11-03T13:17:11Z, before the sealed set was opened.** H3 is also reported for each resistance separately. 30 Ω is on the operator grid; 3, 5, and 20 Ω
are not (`08_STAGE_G_eval.md`, G8). This reporting is descriptive: the primary measure and the pass rule stay as in Amendments 1 and 3. Each
per-resistance row gives F1 of R5, R0ⁿ and R0, and ΔF1 (R5 − R0ⁿ) with its bootstrap interval.

