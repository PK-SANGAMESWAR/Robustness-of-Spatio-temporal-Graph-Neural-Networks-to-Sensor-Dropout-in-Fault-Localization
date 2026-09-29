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

## 3. Statistical tests (fixed now)

- Seed spread (mean ± std, n = 3) reported for every 3-seed row; 1-seed rows marked.
- Paired cluster bootstrap over the 6,175 test events, 10,000 resamples, 95 % CI, **only** for:
  R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2.
- Paired t over seeds with Holm correction per family, families = {H1–H4}, {H5}, {A1–A3}.
- A hypothesis is "supported" only if the sign is consistent in all 3 seeds **and** the bootstrap
  CI excludes zero. Anything else is reported as observed, not claimed.

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

## Amendments

(None. Append dated entries here; never edit the sections above.)
