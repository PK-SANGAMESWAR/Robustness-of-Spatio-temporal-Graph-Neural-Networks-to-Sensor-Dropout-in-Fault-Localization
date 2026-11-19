# T13 — statistical tests, test split (6,175 events, 247,000 windows)

| pair | hypothesis | cell | R5 | reference | ΔF1 | seeds > 0 | p (t, raw) | p Holm (cells chosen post hoc) | 95 % CI (event bootstrap) | supported (3/3 and CI excl. 0) |
|---|---|---|---|---|---|---|---|---|---|---|
| R5 vs R0ⁿ | H1 | k=0, σ=0, R=10 | 0.739 | 0.639 | +0.099 | 3/3 | 0.0138 | 0.0546 | [+0.084, +0.114] | yes |
| R5 vs R0ⁿ | H2 | k=10, σ=0, pooled | 0.851 | 0.647 | +0.203 | 3/3 | 0.0011 | 0.0089 | [+0.198, +0.209] | yes |
| R5 vs R0ⁿ | H2 | k=10, σ=0.0005, pooled | 0.848 | 0.648 | +0.200 | 3/3 | 0.0012 | 0.0089 | [+0.196, +0.206] | yes |
| R5 vs R0ⁿ | H2 | k=10, σ=0.002, pooled | 0.806 | 0.637 | +0.169 | 3/3 | 0.0027 | 0.0162 | [+0.165, +0.173] | yes |
| R5 vs R0ⁿ | H2 | k=10, σ=0.01, pooled | 0.677 | 0.574 | +0.103 | 3/3 | 0.0109 | 0.0546 | [+0.100, +0.107] | yes |
| R5 vs R0ⁿ | H4 | k=0, σ=0.0005, pooled | 0.893 | 0.832 | +0.061 | 3/3 | 0.0109 | 0.0546 | [+0.056, +0.067] | yes |
| R5 vs R0ⁿ | H4 | k=0, σ=0.002, pooled | 0.871 | 0.814 | +0.057 | 3/3 | 0.0155 | 0.0546 | [+0.053, +0.062] | yes |
| R5 vs R0ⁿ | H4 | k=0, σ=0.01, pooled | 0.747 | 0.720 | +0.028 | 3/3 | 0.1208 | 0.1208 | [+0.024, +0.031] | yes |
| R5 vs R0ⁿ | H6 | k=0, σ=0, pooled | 0.894 | 0.833 | +0.061 | 3/3 | 0.0129 |  | [+0.055, +0.067] | rule is |Δ| ≤ 1 seed-std (0.010): no (R5 higher) |
| R5 vs R6ⁱ | A1 | k=0, σ=0, R=10 | 0.739 | 0.141 | +0.598 | 3/3 | 0.0001 | 0.0007 | [+0.582, +0.612] | yes |
| R5 vs R6ⁱ | A1 | k=10, σ=0, pooled | 0.851 | 0.063 | +0.788 | 3/3 | 0.0000 | 0.0001 | [+0.780, +0.794] | yes |
| R5 vs R6ⁱ | A1 | k=10, σ=0.0005, pooled | 0.848 | 0.069 | +0.779 | 3/3 | 0.0000 | 0.0001 | [+0.772, +0.786] | yes |
| R5 vs R6ⁱ | A1 | k=10, σ=0.002, pooled | 0.806 | 0.094 | +0.711 | 3/3 | 0.0001 | 0.0007 | [+0.703, +0.719] | yes |
| R5 vs R6ⁱ | A1 | k=10, σ=0.01, pooled | 0.677 | 0.159 | +0.518 | 3/3 | 0.0001 | 0.0007 | [+0.509, +0.527] | yes |
| R5 vs R6ⁱ | A1 | k=0, σ=0.0005, pooled | 0.893 | 0.087 | +0.807 | 3/3 | 0.0000 | 0.0000 | [+0.799, +0.814] | yes |
| R5 vs R6ⁱ | A1 | k=0, σ=0.002, pooled | 0.871 | 0.131 | +0.740 | 3/3 | 0.0001 | 0.0007 | [+0.732, +0.748] | yes |
| R5 vs R6ⁱ | A1 | k=0, σ=0.01, pooled | 0.747 | 0.219 | +0.529 | 3/3 | 0.0001 | 0.0008 | [+0.518, +0.539] | yes |
| R5 vs R2 | A2 | k=0, σ=0, R=10 | 0.739 | 0.667 | +0.072 | 3/3 | 0.0029 | 0.0115 | [+0.057, +0.086] | yes |
| R5 vs R2 | A2 | k=10, σ=0, pooled | 0.851 | 0.640 | +0.211 | 3/3 | 0.0009 | 0.0065 | [+0.206, +0.216] | yes |
| R5 vs R2 | A2 | k=10, σ=0.0005, pooled | 0.848 | 0.639 | +0.209 | 3/3 | 0.0008 | 0.0065 | [+0.204, +0.214] | yes |
| R5 vs R2 | A2 | k=10, σ=0.002, pooled | 0.806 | 0.630 | +0.175 | 3/3 | 0.0011 | 0.0065 | [+0.171, +0.179] | yes |
| R5 vs R2 | A2 | k=10, σ=0.01, pooled | 0.677 | 0.569 | +0.109 | 3/3 | 0.0008 | 0.0065 | [+0.105, +0.112] | yes |
| R5 vs R2 | A2 | k=0, σ=0.0005, pooled | 0.893 | 0.849 | +0.044 | 3/3 | 0.0061 | 0.0177 | [+0.039, +0.049] | yes |
| R5 vs R2 | A2 | k=0, σ=0.002, pooled | 0.871 | 0.832 | +0.039 | 3/3 | 0.0059 | 0.0177 | [+0.035, +0.043] | yes |
| R5 vs R2 | A2 | k=0, σ=0.01, pooled | 0.747 | 0.739 | +0.008 | 3/3 | 0.0813 | 0.0813 | [+0.005, +0.012] | yes |

- Event-level paired cluster bootstrap: 10000 resamples of the test events, RandomState(20261102), statistic = mean over the 3 seeds of ΔF1 (each F1 the mean over 3 offsets); 95 % percentile CI (fault_localization/stage_g/gate6.py::analyze_cell).
- Pairs pre-registered (14 § 3, amended): R5 vs R0ⁿ, R5 vs R6ⁱ, R5 vs R2. 'Supported' = sign 3/3 and CI excludes zero (14 § 3).
- Holm families {H1–H4} and {A1, A2} are pre-registered; the cells inside each family were chosen after the Gate-4 results, so the Holm column is post hoc. H3 is on the sealed set (T8), not here.
- The bootstrap CI covers sampling of test events only; the 3 trained models are fixed, so it does not include training-seed variance and is narrower than the seed spread. With n = 3 the t-test over seeds is the seed-level evidence: a cell with 3/3 signs and a CI above zero but raw p > 0.05 (H4 and A2 at σ = 1e-2) is a trend, not a result.
- H6 is a control: rule |ΔF1| ≤ 1 seed-std of R0ⁿ; it has no Holm family. The rule was written to catch a loss of clean accuracy; R5 is higher, which also lies outside the band, so the control is not met as written (R5 does not lose clean accuracy).
- R=10 rows use only the test events at R = 10 Ω (n_events in the CSV).
