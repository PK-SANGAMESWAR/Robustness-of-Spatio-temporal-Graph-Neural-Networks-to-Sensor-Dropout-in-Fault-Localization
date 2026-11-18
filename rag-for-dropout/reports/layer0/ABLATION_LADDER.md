# Ablation ladder A0 to A4 (Layer 0 data, assembled by `gridsense.eval.ablation_ladder`)

Written for the author. Seed 42, post-fault windows, held-out store, accuracy target 95 %. Every number is copied or derived from `reports/layer0/layer0_val_seed42.json` and `test_run_seed42.json`. No model run; the test split was not used again. Definitions: coverage = share of windows accepted; selective risk = 1 − accuracy of the accepted windows; false acceptance = accepted and wrong, as a share of all windows; error recall = share of wrong predictions that were NOT accepted.

## A. `val` (cal-B): the rungs

Thresholds are chosen on cal-B and measured on cal-B: coverage is an in-sample figure. Compare rungs with the accuracy column. No confidence intervals are computed for these rows. A1 overshoots the 95 % target (accuracy about 0.96): the isotonic probability is a step function with ties, so its thresholds are coarse. Its lower coverage and higher error recall are therefore NOT a like-for-like comparison with A2. A2, probability + margin and the three-feature composition differ by about 0.01 coverage or less; no interval is available, so no ranking among them is claimed.

| k | rung | coverage | accuracy of accepted | selective risk | false acceptance | error recall |
|---|---|---|---|---|---|---|
| 0 | A0 | 1.000 | 0.881 | 0.119 | 0.119 | 0.000 |
| 0 | A1 | 0.800 | 0.962 | 0.038 | 0.031 | 0.744 |
| 0 | A2 | 0.839 | 0.950 | 0.050 | 0.042 | 0.650 |
| 0 | p+margin | 0.854 | 0.950 | 0.050 | 0.043 | 0.644 |
| 0 | gate3 (p+margin+agreement) | 0.853 | 0.950 | 0.050 | 0.042 | 0.644 |
| 1 | A0 | 1.000 | 0.868 | 0.132 | 0.132 | 0.000 |
| 1 | A1 | 0.778 | 0.962 | 0.038 | 0.029 | 0.777 |
| 1 | A2 | 0.825 | 0.950 | 0.050 | 0.041 | 0.688 |
| 1 | p+margin | 0.832 | 0.950 | 0.050 | 0.041 | 0.685 |
| 1 | gate3 (p+margin+agreement) | 0.832 | 0.950 | 0.050 | 0.041 | 0.685 |
| 2 | A0 | 1.000 | 0.854 | 0.146 | 0.146 | 0.000 |
| 2 | A1 | 0.759 | 0.963 | 0.037 | 0.028 | 0.806 |
| 2 | A2 | 0.803 | 0.950 | 0.050 | 0.040 | 0.726 |
| 2 | p+margin | 0.810 | 0.950 | 0.050 | 0.040 | 0.723 |
| 2 | gate3 (p+margin+agreement) | 0.812 | 0.950 | 0.050 | 0.040 | 0.725 |
| 5 | A0 | 1.000 | 0.812 | 0.188 | 0.188 | 0.000 |
| 5 | A1 | 0.683 | 0.960 | 0.040 | 0.027 | 0.856 |
| 5 | A2 | 0.724 | 0.950 | 0.050 | 0.036 | 0.808 |
| 5 | p+margin | 0.740 | 0.950 | 0.050 | 0.037 | 0.805 |
| 5 | gate3 (p+margin+agreement) | 0.740 | 0.950 | 0.050 | 0.037 | 0.805 |
| 10 | A0 | 1.000 | 0.701 | 0.299 | 0.299 | 0.000 |
| 10 | A1 | 0.483 | 0.952 | 0.048 | 0.023 | 0.922 |
| 10 | A2 | 0.501 | 0.951 | 0.049 | 0.025 | 0.917 |
| 10 | p+margin | 0.520 | 0.950 | 0.050 | 0.026 | 0.914 |
| 10 | gate3 (p+margin+agreement) | 0.520 | 0.950 | 0.050 | 0.026 | 0.914 |

A3 (+ silent-failure check): **NOT RUN** (the Stage F check is not recorded as usable; `PC-STGNN/07_*`).
A4 (+ retrieval and LLM report): the report does not change who is accepted, so A4 has the selection columns of the gate. Report quality is measured apart (S4, S6).

## B. AURC on `val` (cal-B, post-fault; lower is better)

| k | probability (isotonic) | raw vote share | margin |
|---|---|---|---|
| 0 | 0.022 | 0.021 | 0.020 |
| 1 | 0.024 | 0.023 | 0.023 |
| 2 | 0.027 | 0.026 | 0.025 |
| 5 | 0.041 | 0.039 | 0.038 |
| 10 | 0.094 | 0.089 | 0.087 |

## C. `test`: the frozen gate, one run (the only rung that was run on test)

| k | A0 accuracy | gate_v1 coverage | selective risk | false acceptance | error recall | AURC (isotonic) |
|---|---|---|---|---|---|---|
| 0 | 0.876 | 0.845 | 0.051 | 0.043 | 0.654 | 0.022 |
| 1 | 0.863 | 0.825 | 0.051 | 0.042 | 0.691 | 0.025 |
| 2 | 0.850 | 0.801 | 0.050 | 0.040 | 0.734 | 0.029 |
| 5 | 0.802 | 0.731 | 0.051 | 0.037 | 0.812 | 0.042 |
| 10 | 0.694 | 0.513 | 0.051 | 0.026 | 0.914 | 0.094 |

A1 and A2 were never run on `test` (the test split is used once per frozen configuration). A second test use needs the author's approval.
