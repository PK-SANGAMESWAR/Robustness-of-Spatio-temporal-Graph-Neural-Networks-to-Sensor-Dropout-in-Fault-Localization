**standard**, seed 42 (`results/fullscale_final/rgsage_max/seed_00042/checkpoint_best.pt`, sha256 `a7f9218e…`), 247,000 test windows, CPU, 2 threads, 9.6 min.

| k | paper seed 42 | CPU, paper rank (control) | CPU, corrected rank | control − paper | corrected − paper | corrected − control (95 % CI, test runs) | windows with a changed mask | accuracy on those: control → corrected |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8526 | 0.8526 | (same) | -0.0000 | – | – | – | – |
| 1 | 0.4804 | 0.4803 | 0.4821 | -0.0001 | +0.0017 | +0.0018 (+0.0016 to +0.0020) | 4,940 (4.0% of fault windows) | 0.923 → 0.992 |
| 2 | 0.3893 | 0.3894 | 0.3849 | +0.0001 | -0.0043 | -0.0045 (-0.0057 to -0.0032) | 9,880 (8.0% of fault windows) | 0.755 → 0.656 |
| 5 | 0.2346 | 0.2346 | 0.2135 | +0.0000 | -0.0211 | -0.0211 (-0.0243 to -0.0178) | 34,580 (28.0% of fault windows) | 0.292 → 0.139 |
| 10 | 0.0881 | 0.0881 | 0.0657 | +0.0000 | -0.0224 | -0.0224 (-0.0254 to -0.0192) | 54,340 (44.0% of fault windows) | 0.129 → 0.050 |

**dropout_aware**, seed 42 (`results/dropout_aware/rgsage_max/seed_00042/checkpoint_best.pt`, sha256 `0172a60a…`), 247,000 test windows, CPU, 2 threads, 9.8 min.

| k | paper seed 42 | CPU, paper rank (control) | CPU, corrected rank | control − paper | corrected − paper | corrected − control (95 % CI, test runs) | windows with a changed mask | accuracy on those: control → corrected |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8515 | 0.8515 | (same) | -0.0000 | – | – | – | – |
| 1 | 0.7659 | 0.7659 | 0.7576 | -0.0000 | -0.0083 | -0.0083 (-0.0103 to -0.0066) | 4,940 (4.0% of fault windows) | 0.803 → 0.595 |
| 2 | 0.7090 | 0.7090 | 0.7073 | -0.0000 | -0.0017 | -0.0017 (-0.0032 to -0.0003) | 9,880 (8.0% of fault windows) | 0.537 → 0.511 |
| 5 | 0.5891 | 0.5891 | 0.6048 | +0.0000 | +0.0157 | +0.0157 (+0.0125 to +0.0188) | 34,580 (28.0% of fault windows) | 0.456 → 0.492 |
| 10 | 0.5579 | 0.5579 | 0.5324 | +0.0000 | -0.0256 | -0.0256 (-0.0297 to -0.0213) | 54,340 (44.0% of fault windows) | 0.484 → 0.441 |

**Dropout-aware minus standard under the same targeting** (macro-F1; one training seed; no confidence interval):

| k | paper, seed 42 | CPU, paper rank | CPU, corrected rank | paper, mean of 3 seeds (context only) |
|---|---|---|---|---|
| 1 | +0.2855 | +0.2856 | +0.2755 | +0.3067 |
| 2 | +0.3198 | +0.3196 | +0.3224 | +0.3336 |
| 5 | +0.3546 | +0.3546 | +0.3913 | +0.3942 |
| 10 | +0.4698 | +0.4698 | +0.4666 | +0.4730 |
