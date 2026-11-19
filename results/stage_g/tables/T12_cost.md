# T12 — computational cost

| model | n_seeds | params | MFLOPs / window (backbone + residual) | training GPU-h | best-val epoch | scoring ms / window-forward | latency b=1 / b=32 (ms) |
|---|---|---|---|---|---|---|---|
| R0ⁿ | 3 | 95,258 | 53.6 (53.6 + 0.0) | 7.17 ± 0.01 | 30/29/27 | 0.325 ± 0.001 | 0.64 / 5.55 (cuda) |
| R5 (PC-STGNN) | 3 | 95,259 | 79.8 (53.6 + 26.3) | 21.51 ± 0.02 | 19/29/17 | 1.799 ± 0.002 | 2.59 / 50.21 (cuda) |
| R2 (learned decoder) | 3 | 99,613 | 53.6 (53.6 + 0.0) | 7.28 ± 0.03 | 29/28/22 | 0.320 ± 0.001 | 0.63 / 5.32 (cuda) |
| R8 (no noise aug.) | 1 | 95,259 | 79.8 (53.6 + 26.3) | 21.30 | 24 | 1.814 | 2.59 / 50.21 (cuda) |
| B2ᵃ (regressed R̂) | 1 | 95,324 | 79.8 (53.6 + 26.3) | 8.95 | 1 | — | 2.38 / 14.93 (cuda) |
| R0 (Paper 1) | 3 | 95,258 | 53.6 (53.6 + 0.0) | 6.87 ± 0.01 | 28/27/29 | — | 0.63 / 5.35 (cuda) |

- Parameters: model rebuilt from its config.yaml. R5 − R0ⁿ = 1 parameter (β).
- FLOPs: backbone by torch FlopCounterMode (matrix products, multiply-add = 2); residual = 4 × table entries per window (difference, square, weight, sum; physics/residual.py 'direct'). Residual counted only for models that use it at inference (fusion on).
- Training GPU-h: sum of train_seconds in metrics_by_epoch.csv (T1000 8 GB), mean ± std over seeds.
- Scoring ms / window-forward: validation full grid (60 forwards × 248,000 windows), wall clock including data loading and noise/mask generation; a throughput figure, not a latency.
- Latency: synthetic batch [B, 25, 20, 6], measured only when no other job runs (12_CHECKLISTS § 0).
