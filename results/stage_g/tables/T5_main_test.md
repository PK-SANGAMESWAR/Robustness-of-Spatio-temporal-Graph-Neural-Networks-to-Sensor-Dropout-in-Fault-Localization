# T5 — main results, test split (247,000 windows), random dropout, σ = 0

| model | n_seeds | k=0 | k=1 | k=2 | k=5 | k=10 | Ret@10 | AUDC rel | AUDC abs | protocol |
|---|---|---|---|---|---|---|---|---|---|---|
| R0ⁿ (noise-aug., no physics) | 3 | 0.833 ± 0.010 | 0.819 ± 0.010 | 0.805 ± 0.010 | 0.757 ± 0.011 | 0.647 ± 0.011 | 77.7 % | 0.899 ± 0.005 | 0.749 ± 0.010 | Stage G scorer, test, random dropout, sigma=0, 3 offsets |
| R5 (PC-STGNN) | 3 | 0.894 ± 0.002 | 0.889 ± 0.002 | 0.884 ± 0.002 | 0.871 ± 0.001 | 0.851 ± 0.001 | 95.2 % | 0.975 ± 0.001 | 0.871 ± 0.001 | Stage G scorer, test, random dropout, sigma=0, 3 offsets |
| R2 (learned decoder) | 3 | 0.851 ± 0.006 | 0.836 ± 0.006 | 0.821 ± 0.006 | 0.766 ± 0.008 | 0.640 ± 0.010 | 75.2 % | 0.890 ± 0.003 | 0.757 ± 0.008 | Stage G scorer, test, random dropout, sigma=0, 3 offsets |
| R6ⁱ (shuffled table, inference) | 3 | 0.079 ± 0.001 | — | — | — | 0.063 ± 0.003 | — | — | — | Stage G scorer, test, random dropout, sigma=0, 3 offsets |
| B2ᵃ (regressed R̂; epoch-1 checkpoint) | 1 | 0.706 | 0.701 | 0.692 | 0.641 | 0.502 | 71.1 % | 0.885 | 0.625 | Stage G scorer, test, random dropout, sigma=0, 3 offsets |
| PSM (B2, zero parameters) | 1 | 0.767 | 0.766 | 0.764 | 0.758 | 0.746 | 97.2 % | 0.987 | 0.757 | Stage G scorer, test, sigma=0; identical in the 3 R5 files |
| GRU (standard) | 3 | 0.286 ± 0.009 | 0.273 ± 0.010 | 0.252 ± 0.007 | 0.111 ± 0.046 | 0.051 ± 0.012 | 17.9 % | 0.522 ± 0.081 | 0.149 ± 0.022 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGCN (standard) | 3 | 0.712 ± 0.017 | 0.664 ± 0.013 | 0.610 ± 0.005 | 0.418 ± 0.037 | 0.164 ± 0.046 | 23.2 % | 0.608 ± 0.047 | 0.432 ± 0.024 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGSAGE-Mean (standard) | 3 | 0.786 ± 0.023 | 0.720 ± 0.004 | 0.647 ± 0.014 | 0.393 ± 0.033 | 0.108 ± 0.051 | 13.8 % | 0.541 ± 0.041 | 0.425 ± 0.027 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGSAGE-Max (standard) | 3 | 0.846 ± 0.007 | 0.736 ± 0.013 | 0.590 ± 0.024 | 0.236 ± 0.023 | 0.048 ± 0.015 | 5.7 % | 0.402 ± 0.020 | 0.340 ± 0.015 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGATv2 (standard) | 3 | 0.522 ± 0.087 | 0.490 ± 0.063 | 0.444 ± 0.059 | 0.302 ± 0.079 | 0.157 ± 0.081 | 31.0 % | 0.626 ± 0.112 | 0.324 ± 0.061 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| GRU (dropout-aware) | 3 | 0.300 ± 0.037 | 0.300 ± 0.037 | 0.299 ± 0.038 | 0.278 ± 0.054 | 0.078 ± 0.028 | 25.4 % | 0.781 ± 0.038 | 0.236 ± 0.042 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGCN (dropout-aware) | 3 | 0.613 ± 0.029 | 0.607 ± 0.027 | 0.600 ± 0.026 | 0.579 ± 0.023 | 0.533 ± 0.020 | 87.0 % | 0.940 ± 0.011 | 0.576 ± 0.023 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGSAGE-Mean (dropout-aware) | 3 | 0.673 ± 0.031 | 0.666 ± 0.029 | 0.657 ± 0.027 | 0.628 ± 0.020 | 0.550 ± 0.041 | 81.9 % | 0.922 ± 0.034 | 0.620 ± 0.019 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGSAGE-Max (dropout-aware) = R0 | 3 | 0.839 ± 0.011 | 0.827 ± 0.011 | 0.815 ± 0.011 | 0.774 ± 0.013 | 0.674 ± 0.017 | 80.3 % | 0.913 ± 0.006 | 0.766 ± 0.013 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |
| RGATv2 (dropout-aware) | 3 | 0.553 ± 0.043 | 0.550 ± 0.042 | 0.546 ± 0.040 | 0.533 ± 0.034 | 0.505 ± 0.026 | 91.4 % | 0.961 ± 0.014 | 0.531 ± 0.034 | Paper-1 evaluator, test, zero_messages; other mask stream for k > 0 |

- Soft-voted 26-class macro-F1, mean ± sample std over seeds 42/43/44; 1-seed rows have no std.
- AUDC: Paper 1 Eq. 8, per training seed, then averaged; relative = on F1(k)/F1(0); absolute = on raw F1(k).
- Stage G rows: same mask/noise seeds for every model (3 offsets). Paper-1 rows: Paper-1 evaluator, same split and metric; at k = 0 the two evaluators agree (R0ⁿ check: |Δ| < 1e-9 on validation); for k > 0 the mask streams differ (same distribution).
- PSM is read from the R5 scorer (physics-only prediction). R6ⁱ was scored at k ∈ {0, 10} only.
- B2ᵃ: the best-validation checkpoint is epoch 1 of 30; validation F1 then varies between 0.46 and 0.67 (metrics_by_epoch.csv) with a train loss near 100 against 2.6 for R5. It is the epoch-1 checkpoint of an unstable run, so it does not show how well a regressed R̂ can work.
- Ceiling: 0.9744 (unlabelable w = 20 windows, 11_METRICS § 1). Trivial always-no-fault: 0.0256.
