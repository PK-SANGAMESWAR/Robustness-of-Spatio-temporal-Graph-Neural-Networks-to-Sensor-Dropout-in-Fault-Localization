# T9a — silent-sensor detection (PSM-refit residual), validation split, k = 0

| condition | AUROC LOO σ0=0 | AUROC in-sample σ0=0 | TPR@FPR0.05 LOO σ0=0 | AUROC LOO σ0=0.01 | AUROC in-sample σ0=0.01 | TPR@FPR0.05 LOO σ0=0.01 |
|---|---|---|---|---|---|---|
| stuck_s1 | 0.6440 | 0.6452 | 0.293 | 0.6235 | 0.6227 | 0.289 |
| stuck_s2 | 0.6103 | 0.6112 | 0.253 | 0.6002 | 0.5998 | 0.253 |
| stuck_s3 | 0.5900 | 0.5906 | 0.234 | 0.5823 | 0.5820 | 0.232 |
| bias_2pct_s1 | 0.9991 | 0.9993 | 0.996 | 0.8537 | 0.8491 | 0.578 |
| bias_2pct_s2 | 0.9981 | 0.9985 | 0.992 | 0.8504 | 0.8464 | 0.569 |
| bias_2pct_s3 | 0.9971 | 0.9977 | 0.988 | 0.8482 | 0.8446 | 0.564 |
| bias_5pct_s1 | 0.9996 | 0.9997 | 0.998 | 0.9908 | 0.9911 | 0.973 |
| bias_5pct_s2 | 0.9990 | 0.9993 | 0.995 | 0.9892 | 0.9896 | 0.968 |
| bias_5pct_s3 | 0.9984 | 0.9988 | 0.992 | 0.9878 | 0.9883 | 0.963 |
| gauss_0.02_s1 | 0.9753 | 0.9761 | 0.922 | 0.7833 | 0.7829 | 0.498 |
| gauss_0.02_s2 | 0.9667 | 0.9677 | 0.901 | 0.7812 | 0.7809 | 0.496 |
| gauss_0.02_s3 | 0.9601 | 0.9612 | 0.883 | 0.7802 | 0.7801 | 0.493 |
| swap_s1 | 0.9969 | 0.9973 | 0.996 | 0.9959 | 0.9963 | 0.981 |
| swap_s2 | 0.9908 | 0.9913 | 0.945 | 0.9894 | 0.9900 | 0.947 |
| swap_s3 | 0.9788 | 0.9793 | 0.937 | 0.9771 | 0.9777 | 0.936 |

- AUROC per sensor: positives = corrupted sensors, negatives = clean sensors of the corrupted windows; all 248,000 validation windows.
- S5.1 hard gate (bias ±2 %, LOO, threshold 0.75) passed at every σ0. Stuck sensors are the weak case (AUROC about 0.6 at σ0 = 0).
- S5.2: in-sample AUROC ≥ LOO at σ0 ≤ 2e-3 (the expected 'LOO clearly better' did not hold); the in-sample residual is used.
- Source: results/pcstgnn/audit/s5_1_silent_detection/silent_detection.json and results/stage_f/silent_detection_noise/sigma_0.01/.
