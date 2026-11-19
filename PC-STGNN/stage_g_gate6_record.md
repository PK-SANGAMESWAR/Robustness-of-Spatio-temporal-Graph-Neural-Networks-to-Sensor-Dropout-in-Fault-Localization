# Gate 6 and H3: sealed OOD result `[Recorded 2026-11-19, from files written 2026-11-18/19]`

Sources: `results/stage_g/ood/gate6_analysis.json` (written by `scripts/run_gate6_analysis.py`), `results/stage_g/tables/T8_ood.{md,csv}`,
`results/stage_g/ood/{r5,r0n,r2,b2a}_seed*_scoring.json`, `results/stage_g/ood/COMPLETED.json`, `results/stage_g/ood/log_row.md`.
The sealed set was opened once (2026-11-18 10:51 IST, hash verified) by `scripts/run_stage_g_ood.py`; 10 models of `config/stage_g/frozen_models.json` were scored
(R0ⁿ ×3, R5 ×3, R2 ×3, B2ᵃ ×1). 11,000 events, 440,000 windows, R ∈ {3, 5, 20, 30} Ω. R0 is not in the frozen list, so there is no R0 row.
Statistic (14, Amendments 1, 3, 9): macro-F1 pooled over all events, k = 0; mean of 3 mask/noise offsets; event-level paired cluster bootstrap (10,000 resamples,
`RandomState(20261102)`); Δ = seed mean of F1(R5) − F1(R0ⁿ).

## Result: Gate 6 passed; H3 passed as written
`gate6_analysis.json`: `evaluable: true`, `passed: true`. All three σ pass: the sign is positive in 3/3 seeds and the CI lower bound is above 0.

| σ (p.u.) | R5 | R0ⁿ | Δ per seed (42 / 43 / 44) | Δ (seed mean) | 95 % CI | pass |
|---|---|---|---|---|---|---|
| 5e-4 | 0.508 | 0.084 | +0.416 / +0.433 / +0.422 | +0.424 | [+0.416, +0.431] | yes |
| 2e-3 | 0.307 | 0.081 | +0.209 / +0.246 / +0.222 | +0.226 | [+0.220, +0.231] | yes |
| 1e-2 | 0.090 | 0.065 | +0.026 / +0.022 / +0.028 | +0.026 | [+0.024, +0.027] | yes |

## What the pass does and does not mean (descriptive; no decision rule)
- **R0ⁿ is near chance on this set** (0.065 to 0.086; the always-"no fault" classifier scores 0.026), so a pass over R0ⁿ is a low bar. The size of Δ falls from +0.424 to +0.026 as σ rises.
- **R5's learned branch collapses off-distribution.** Network-only F1 of the R5 runs at σ = 0 is 0.090 (`f1_network_only_mean`).
- **R5 is below its own physics-only predictor (PSM) at every noise level.** PSM F1 (`f1_psm_mean`, mean of the 3 R5 files) against fused F1: σ=0 0.571 vs 0.534; 5e-4 0.567 vs 0.508;
  2e-3 0.520 vs 0.307; 1e-2 0.242 vs 0.090. PSM was not a registered comparison for H3, so this is a descriptive finding. It says that the OOD gain over R0ⁿ comes from the
  operator, and that the fusion does not keep the operator's full OOD score once the network is confidently wrong.
- R2 (learned decoder) is at 0.096 at σ = 0, close to R0ⁿ (0.086). B2ᵃ (1 seed, epoch-1 checkpoint of an unstable run) is at 0.231.

## Descriptive rows (Amendments 3, 4, 11), σ = 0, k = 0 (T8 has every cell)
| Subset | n events | R5 | R0ⁿ | Δ | CI |
|---|---|---|---|---|---|
| pooled | 11,000 | 0.534 | 0.086 | +0.448 | [+0.440, +0.456] |
| 3 Ω (off the operator grid) | 2,750 | 0.872 | 0.068 | +0.804 | [+0.795, +0.814] |
| 5 Ω (off the grid) | 2,750 | 0.720 | 0.116 | +0.604 | [+0.590, +0.617] |
| 20 Ω (off the grid) | 2,750 | 0.256 | 0.082 | +0.174 | [+0.165, +0.183] |
| 30 Ω (on the grid, axis edge) | 2,750 | 0.159 | 0.046 | +0.113 | [+0.107, +0.119] |
| grounded (7 types) | 7,000 | 0.525 | 0.084 | +0.442 | [+0.432, +0.452] |
| ungrounded (4 types) | 4,000 | 0.550 | 0.091 | +0.460 | [+0.446, +0.472] |

- The gain falls as R rises: 3 Ω is the best, 30 Ω the worst. 30 Ω is the only resistance on the operator's R axis, so the weak 30 Ω score is not caused by a table gap on R.
  (A weaker voltage sag at high resistance is a possible reason; it was not tested here.)
- Grounded / ungrounded (registered before the opening, with the expectation that grounded faults would gain more): the ungrounded stratum is slightly higher for R5
  (0.550 vs 0.525) and its Δ is slightly larger (+0.460 vs +0.442). The expectation is not supported. No decision rule.
- k = 5 and k = 10 at σ = 0: pooled R5 0.518 and 0.491 against R0ⁿ 0.076 and 0.066 (Δ +0.442 and +0.426, 3/3 seeds, CIs above 0).

## Fallback map (`14_PREREGISTRATION.md` § 5): no row ticked
"Gains on OOD R / noise, flat in-distribution" does not match clearly: the in-distribution gain is not flat (test split, R5 − R0ⁿ = +0.061 at k = 0, Holm-adjusted trend/significant
cells in T13). No other row applies. The box stays unticked; the author decides how to frame the paper.

## Claims for the papers (from this record)
- Say: R5 passes Gate 6, with a large gain at 3 and 5 Ω at low noise and a small gain at σ = 10⁻², all over a near-chance R0ⁿ.
- Say: R5 is below PSM alone at every noise level on this set (descriptive), and its network branch alone is at 0.090.
- Do not say: "the fusion extrapolates to unseen resistance" without the PSM sentence. Do not say: "grounded faults benefit more".
