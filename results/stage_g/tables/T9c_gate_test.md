# T9c — G-TF2 training-free gate, TEST split (247,000 windows), σ0 = 1e-2, k = 0 (3 seeds)

| arm | condition | role | no gate | gate | oracle gate | gate − no gate | seeds > 0 | p (raw t) | recovered of gap to oracle | FPR clean sensors |
|---|---|---|---|---|---|---|---|---|---|---|
| gtf2 | stuck_s1 | primary | 0.671 ± 0.013 | 0.676 ± 0.011 | 0.742 ± 0.008 | +0.0047 | 3/3 | 0.0617 | 0.066 | 0.053 |
| gtf2 | stuck_s2 | primary | 0.583 ± 0.016 | 0.592 ± 0.015 | 0.735 ± 0.008 | +0.0091 | 3/3 | 0.0031 | 0.060 | 0.098 |
| gtf2 | stuck_s3 | primary | 0.502 ± 0.011 | 0.512 ± 0.017 | 0.729 ± 0.009 | +0.0097 | 3/3 | 0.0988 | 0.043 | 0.136 |
| gtf2 | bias_2pct_s1 | secondary | 0.733 ± 0.009 | 0.736 ± 0.009 | 0.742 ± 0.008 | +0.0030 | 3/3 | 0.0397 | — | 0.010 |
| gtf2 | bias_2pct_s2 | secondary | 0.716 ± 0.012 | 0.724 ± 0.010 | 0.735 ± 0.008 | +0.0073 | 3/3 | 0.0275 | 0.391 | 0.011 |
| gtf2 | bias_2pct_s3 | secondary | 0.702 ± 0.014 | 0.713 ± 0.012 | 0.729 ± 0.009 | +0.0106 | 3/3 | 0.0169 | 0.392 | 0.011 |
| gtf2 | bias_5pct_s1 | secondary | 0.709 ± 0.008 | 0.739 ± 0.007 | 0.742 ± 0.008 | +0.0300 | 3/3 | 0.0447 | 0.905 | 0.014 |
| gtf2 | bias_5pct_s2 | secondary | 0.670 ± 0.011 | 0.729 ± 0.006 | 0.735 ± 0.008 | +0.0592 | 3/3 | 0.0213 | 0.912 | 0.018 |
| gtf2 | bias_5pct_s3 | secondary | 0.637 ± 0.012 | 0.721 ± 0.006 | 0.729 ± 0.009 | +0.0841 | 3/3 | 0.0131 | 0.914 | 0.022 |
| gtf2 | gauss_0.02_s1 | reported only | 0.741 ± 0.008 | 0.741 ± 0.008 | 0.742 ± 0.008 | -0.0002 | 2/3 | 0.6827 | — | 0.010 |
| gtf2 | gauss_0.02_s2 | reported only | 0.734 ± 0.008 | 0.734 ± 0.008 | 0.735 ± 0.008 | +0.0004 | 2/3 | 0.5387 | — | 0.011 |
| gtf2 | gauss_0.02_s3 | reported only | 0.727 ± 0.008 | 0.728 ± 0.008 | 0.729 ± 0.009 | +0.0008 | 2/3 | 0.5845 | — | 0.011 |
| gtf2 | swap_s1 | secondary | 0.564 ± 0.012 | 0.649 ± 0.008 | 0.738 ± 0.008 | +0.0844 | 3/3 | 0.0020 | 0.485 | 0.108 |
| gtf2 | swap_s2 | secondary | 0.420 ± 0.005 | 0.544 ± 0.008 | 0.730 ± 0.008 | +0.1238 | 3/3 | 0.0011 | 0.400 | 0.216 |
| gtf2 | swap_s3 | secondary | 0.318 ± 0.006 | 0.450 ± 0.009 | 0.721 ± 0.008 | +0.1320 | 3/3 | 0.0043 | 0.327 | 0.316 |
| gtf2 | clean windows (no corruption) | cost check | 0.747 ± 0.008 | 0.746 ± 0.008 | — | -0.0007 | 0/3 | — | — | 0.010 |

- Same design as T9b with the split changed; thresholds frozen from validation half A (sha256 in stage_f_gtf2_test_preregistration.md). Registered before any test number existed.
- Primary rule (stuck, 9/9 pairs) and the secondary rules are copied from the validation registration. A pass is a sign result, not a significance claim (n = 3).
- 'recovered' is read only where oracle − no gate ≥ 0.01 (— otherwise). Gaussian 0.02 is reported only.
- Compare with T9b (validation half B); the two tables use different splits and are not pooled.
- Source: results/stage_f/gtf2_test/summary.json.
