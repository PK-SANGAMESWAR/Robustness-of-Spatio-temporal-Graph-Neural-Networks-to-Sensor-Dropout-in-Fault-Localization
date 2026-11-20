# G-TF2 on the test split: result `[Recorded 2026-11-20, from files written 2026-11-20]`

Registration: `stage_f_gtf2_test_preregistration.md` (commit adca737; wording fix 4621518). Scoring: `scripts/run_gtf2_test.py score` (started 2026-11-19 12:05 IST,
39 of 39 shards, 1,104.5 min), then `aggregate` and `summarize`. Sources: `results/stage_f/gtf2_test/summary.json`, `sigma_0.01/seed_{00042,00043,00044}.json`,
`results/stage_g/tables/T9c_gate_test.{md,csv}`. Test split, 247,000 windows, k = 0, σ₀ = 1e-2, R5 seeds 42/43/44, thresholds frozen from validation half A.
Nothing in the detector, thresholds or rules was changed after the first test number was seen.

## Decision rules (copied, not changed)
| Type | Role | Rule: gate above no gate in 3/3 seeds at s = 1, 2, 3 | Result |
|---|---|---|---|
| stuck | primary | 9 of 9 pairs | **met** (9/9) |
| bias ±2 % | secondary | 9 of 9 | met |
| bias ±5 % | secondary | 9 of 9 | met |
| swap | secondary | 9 of 9 | met |
| Gaussian 0.02 | reported only | not a rule | sign positive in 2/3 seeds at each s; not useful |

A pass is a sign result at n = 3, not a significance claim. Per-seed sign is positive in every pair for stuck, bias and swap.

## What the numbers say (descriptive)
- **Stuck (primary): the gain is real in sign but small.** Gate minus no gate: +0.0047 / +0.0091 / +0.0097 at s = 1 / 2 / 3 (no-gate F1 0.671 / 0.583 / 0.502). The recovered share of the
  gap to the oracle gate is 0.066 / 0.060 / 0.043 (all denominators ≥ 0.01). Raw paired-t p = 0.062 / 0.003 / 0.099; Holm over s: 0.123 / 0.009 / 0.123. Only s = 2 is below 0.05 after Holm.
  The oracle gate gains 0.07 to 0.23, so the gate recovers a small part of what is available. Detector TPR is 1.0 because the simulated stuck sensor is exactly flat (AUROC 1.000 by construction;
  see the registration § 5). Why the recovered share is small was not tested here.
- **Bias:** ±5 % recovers 0.905 / 0.912 / 0.914 of the gap (TPR about 0.96). ±2 % recovers 0.391 / 0.392 at s = 2, 3 (TPR about 0.34); at s = 1 the denominator is 0.013 for seed 42 only and
  below 0.01 for seeds 43 and 44, so `rec` is not read (—).
- **Swap:** +0.084 / +0.124 / +0.132; recovered 0.485 / 0.400 / 0.327. The false-flag rate on clean sensors of corrupted windows grows with s (0.108 / 0.216 / 0.316).
- **Gaussian 0.02 p.u. (reported only):** deltas −0.0002 to +0.0008, p ≥ 0.54. The gate does nothing here; the oracle gap is below 0.005, so there is nothing to recover.
- **Cost on clean windows:** −0.0007 mean (gate below no gate in 3/3 seeds; per seed −0.0008, −0.0009, −0.0006). Realized clean-window FPR 0.010.
- **False flags on stuck and swap windows are high:** realized FPR on clean sensors of corrupted windows is 0.053 / 0.098 / 0.136 for stuck s = 1 / 2 / 3 against the 0.005 per-sensor target on clean windows.

## Against validation half B (T9b; the splits are not pooled)
The same sign in all rules. Stuck gains are a little smaller on test (+0.0024 to +0.0158 per seed, against +0.0062 to +0.0204 on validation), and the recovered share is lower (0.066 / 0.060 / 0.043 against
0.102 / 0.075 / 0.061). Bias and swap results are close. The stuck-sensor gain therefore holds on data that took no part in the detector choice or the thresholds, with a smaller effect.

## Claims for the papers
- Say: a training-free gate (in-sample PSM residual OR temporal flatline) beats no gate in all 9 stuck pairs, and in all bias and swap pairs, on the test split; stuck gains are small (about 0.005 to 0.010 F1).
- Say: the stuck gate recovers about 4 to 7 % of the oracle gap, and the Gaussian case is not helped.
- Do not say: "significant" for stuck (3 seeds; Holm-adjusted p is 0.12 at s = 1 and s = 3). Do not drop the clean-window cost (−0.0007) or the swap false-flag rate.
- Limit L4 of `stage_g_known_limitations_DRAFT.md` is now resolved for G-TF2. The residual-only arm (3b) remains validation-only (dropped on test by the registration).
