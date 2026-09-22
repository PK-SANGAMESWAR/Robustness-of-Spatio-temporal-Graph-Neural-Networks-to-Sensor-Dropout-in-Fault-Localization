# Pre-registration — PC-STGNN / PCRG-STGNN

**Status: TEMPLATE. Not yet filled in.** Fill in at the end of Stage C (S2.7), using validation
split and OOD-pilot results only, before any test-split or sealed-OOD number is viewed. Commit
it, record the hash here and in the paper, and do not edit it afterwards — append an "Amendments"
section instead.

| Field | Value |
|---|---|
| Date filled in | |
| Commit hash | |
| Gate-2 branch taken (from `04_STAGE_C_gonogo.md`) | |
| Validation-only S2.6 numbers that motivated it (ΔF1 at k=0 / k=10 / 10 Ω / each σ₀, per seed) | |
| OOD-pilot S2.5 contrast (PSM vs R0 F1 at 5 Ω and 20 Ω) | |

## 1. Primary hypotheses (where the gain will appear)

State each as a direction, a condition, and the model pair. Delete rows that the Gate-2 branch
does not support.

| # | Hypothesis | Pair | Condition | Metric | Predicted sign |
|---|---|---|---|---|---|
| H1 | Physics fusion improves high-impedance localization | R5 vs R0 | 10 Ω, k = 0 | F1 by R | + |
| H2 | Physics fusion improves localization under heavy dropout | R5 vs R0 | k = 10 | F1(k), AUDC_abs | + |
| H3 | Physics fusion extrapolates to unseen resistance | R5 vs R0 | R ∈ {3,5,20,30} Ω, k = 0 | F1 (sealed OOD) | + |
| H4 | Physics fusion helps under measurement noise, where dropout-aware training does not | R5 vs R0 | σ₀ ∈ {5e-4, 2e-3, 1e-2}, k = 0 | F1 | + |
| H5 | The gate improves robustness to silent failures over corruption augmentation alone | G2 vs G0 | 4 types × s ∈ {1,2,3} | F1 | + |
| H6 | Clean accuracy is preserved | R5 vs R0 | k = 0, in-distribution | F1 | within 1 seed-std |

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
- [ ] PSM ≥ STGNN on the clean corpus
- [ ] Gains only under silent failures
- [ ] Gate AUROC < 0.75
- [ ] No model gains, identifiability limit

## 6. Sealed-set log

| Event | Date | By | Hash verified |
|---|---|---|---|
| `data/ood_test/` generated and sealed | | | |
| `data/ood_test/` opened (Stage G, once) | | | |

## Amendments

(None. Append dated entries here; never edit the sections above.)
