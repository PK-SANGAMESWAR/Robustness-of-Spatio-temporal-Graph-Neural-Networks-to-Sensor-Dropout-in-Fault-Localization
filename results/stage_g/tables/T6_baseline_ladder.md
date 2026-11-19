# T6 — baseline ladder (two blocks: validation and test; numbers are not compared across blocks)

| block | rung | model | n_seeds | learned parameters | k=0 | k=10 | AUDC abs | protocol |
|---|---|---|---|---|---|---|---|---|
| validation | B1ᵃ | Rank-1 sag-template matched filter | 1 | 0 | 0.162 | 0.165 | — | 26-class window macro-F1, validation split, zero_messages, 1 mask seed for B1/B1ᵃ/PSM (the audit scripts) |
| validation | B1 | ΔV nearest signature (no profiling) | 1 | 0 | 0.649 | 0.601 | 0.626 | 26-class window macro-F1, validation split, zero_messages, 1 mask seed for B1/B1ᵃ/PSM (the audit scripts) |
| validation | B2 | PSM (profiled τ, R, L; argmin ρ) | 1 | 0 | 0.772 | 0.749 | — | 26-class window macro-F1, validation split, zero_messages, 1 mask seed for B1/B1ᵃ/PSM (the audit scripts) |
| validation | B3′ | R0ⁿ (noise-aug., no physics) | 3 | 95,258 | 0.839 ± 0.008 | 0.655 ± 0.012 | — | 26-class window macro-F1, validation, Gate-4 scorer, 3 offsets, σ=0 |
| validation | B5 | R5 (PC-STGNN) | 3 | 95,259 | 0.898 ± 0.003 | 0.857 ± 0.001 | — | 26-class window macro-F1, validation, Gate-4 scorer, 3 offsets, σ=0 |
| test | B2ᵇ | Logistic regression (summary features) | 1 | n/a (classical) | 0.713 | 0.039 | 0.106 | 26-class window macro-F1, test split, zero_messages, 1 fit, 3 mask seeds |
| test | B2ᵇ | Random forest (summary features) | 1 | n/a (classical) | 0.633 | 0.227 | 0.470 | 26-class window macro-F1, test split, zero_messages, 1 fit, 3 mask seeds |
| test | B2 | PSM (profiled τ, R, L; argmin ρ) | 1 | 0 | 0.767 | 0.746 | 0.757 | 26-class window macro-F1, test, Stage G scorer, σ=0 (read from the R5 scorer files; identical in all 3) |
| test | B3 | RGSAGE-Max, standard (Paper 1) | 3 | 95,258 | 0.846 ± 0.007 | 0.048 ± 0.015 | 0.340 | 26-class window macro-F1, test split, zero_messages, Paper-1 evaluator |
| test | B4 = R0 | RGSAGE-Max, dropout-aware (Paper 1) | 3 | 95,258 | 0.839 ± 0.011 | 0.674 ± 0.017 | 0.766 | 26-class window macro-F1, test split, zero_messages, Paper-1 evaluator |
| test | B3′ | R0ⁿ (noise-aug., no physics) | 3 | 95,258 | 0.833 ± 0.010 | 0.647 ± 0.011 | 0.749 | 26-class window macro-F1, test split, zero_messages, Stage G scorer, 3 offsets, σ=0 |
| test | B5 | R5 (PC-STGNN) | 3 | 95,259 | 0.894 ± 0.002 | 0.851 ± 0.001 | 0.871 | 26-class window macro-F1, test split, zero_messages, Stage G scorer, 3 offsets, σ=0 |

- Direction-only physics → table physics → profiled physics → non-graph learning → learned representation → both.
- The validation block holds the physics baselines of the audit scripts (B1ᵃ, B1, PSM) next to R0ⁿ and R5 on the same split. The test block holds the classical and Paper-1 rows next to R0ⁿ, R5 and PSM on the test split. B1ᵃ and B1 were not scored on the test split.
- PSM's clean k = 0 row is an operator-fidelity check (table and corpus come from the same simulator), not physics beating the network (09_STAGE_H_writing, framing 1).
- AUDC abs: raw F1 trapezoid over k ∈ {0, 1, 2, 5, 10}, divided by 10 (Paper 1 Eq. 8 form without normalisation); B1ᵃ and PSM(val) have no 5-level curve in the audit files.
- n_seeds = 1 rows have no seed std; classical rows are one fit.
