# Quarantined test-split outputs (S4.3 and S4.3ᶜ)

Quarantined 2026-10-05 08:37 IST. Nothing here was deleted; paths below are relative to the repo root.

**Why these files were moved.** `fault_localization/experiments/runner.py` always ran a test-split
evaluation at the end of training. So the six noise-free S4.3 tuning runs and the two S4.3ᶜ
noise-diagnostic runs wrote test-split outputs, although both steps are pre-registered as
validation-only ("Never test", `PC-STGNN/06_STAGE_E_pcstgnn.md`). The files are moved here so that
no later step can read a test score by accident.

Each run contributed three files:
- `predictions_test.csv`: per-window test predictions.
- `confusion_matrix.csv`: the test-split graph confusion matrix (the runner builds it from the
  test evaluation).
- `metrics.json`: holds `test_graph` and `test_node_diagnostic`, mixed with the validation fields
  `best_validation_graph_f1` and `beta_final`. The whole file was moved. The validation values it
  holds are also in each run's `metrics_by_epoch.csv` and in `results/pcstgnn_tuning/lambda_selection.json`.

**Side effect.** `scripts/run_pcstgnn_tuning.py` and `scripts/run_pcstgnn_noise_diagnostic.py` skip
an arm only if its `metrics.json` exists. If either script is run again, it would now retrain these arms.

**Use.** No result, selection or verdict used these files (see the S4.3ᶜ quarantine entry in
`06_STAGE_E_pcstgnn.md`). Do not open them, and do not report any number from them.

From now on, every tuning and diagnostic run sets `training.skip_test_eval: true`.

| # | Run | File | Original path | File modification time (IST) |
|---|---|---|---|---|
| 1 | pcstgnn_tuning_lam1_0p0333_lam2_0p0 | predictions_test.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-03 03:32:00 |
| 2 | pcstgnn_tuning_lam1_0p0333_lam2_0p1 | predictions_test.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p1/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-03 15:52:41 |
| 3 | pcstgnn_tuning_lam1_0p0333_lam2_1p0 | predictions_test.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_1p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-04 04:11:41 |
| 4 | pcstgnn_tuning_lam1_0p1_lam2_0p0 | predictions_test.csv | `results/pcstgnn_tuning_lam1_0p1_lam2_0p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-01 06:00:03 |
| 5 | pcstgnn_tuning_lam1_0p3_lam2_0p0 | predictions_test.csv | `results/pcstgnn_tuning_lam1_0p3_lam2_0p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-02 02:54:30 |
| 6 | pcstgnn_tuning_lam1_1p0_lam2_0p0 | predictions_test.csv | `results/pcstgnn_tuning_lam1_1p0_lam2_0p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-02 15:10:02 |
| 7 | pcstgnn_tuning_lam1_0p0333_lam2_0p0 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-03 03:32:00 |
| 8 | pcstgnn_tuning_lam1_0p0333_lam2_0p1 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p1/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-03 15:52:41 |
| 9 | pcstgnn_tuning_lam1_0p0333_lam2_1p0 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_0p0333_lam2_1p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-04 04:11:41 |
| 10 | pcstgnn_tuning_lam1_0p1_lam2_0p0 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_0p1_lam2_0p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-01 06:00:03 |
| 11 | pcstgnn_tuning_lam1_0p3_lam2_0p0 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_0p3_lam2_0p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-02 02:54:30 |
| 12 | pcstgnn_tuning_lam1_1p0_lam2_0p0 | confusion_matrix.csv | `results/pcstgnn_tuning_lam1_1p0_lam2_0p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-02 15:10:02 |
| 13 | pcstgnn_tuning_lam1_0p0333_lam2_0p0 | metrics.json | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p0/pcstgnn/seed_00042/metrics.json` | 2026-10-03 03:31:59 |
| 14 | pcstgnn_tuning_lam1_0p0333_lam2_0p1 | metrics.json | `results/pcstgnn_tuning_lam1_0p0333_lam2_0p1/pcstgnn/seed_00042/metrics.json` | 2026-10-03 15:52:40 |
| 15 | pcstgnn_tuning_lam1_0p0333_lam2_1p0 | metrics.json | `results/pcstgnn_tuning_lam1_0p0333_lam2_1p0/pcstgnn/seed_00042/metrics.json` | 2026-10-04 04:11:40 |
| 16 | pcstgnn_tuning_lam1_0p1_lam2_0p0 | metrics.json | `results/pcstgnn_tuning_lam1_0p1_lam2_0p0/pcstgnn/seed_00042/metrics.json` | 2026-10-01 06:00:02 |
| 17 | pcstgnn_tuning_lam1_0p3_lam2_0p0 | metrics.json | `results/pcstgnn_tuning_lam1_0p3_lam2_0p0/pcstgnn/seed_00042/metrics.json` | 2026-10-02 02:54:29 |
| 18 | pcstgnn_tuning_lam1_1p0_lam2_0p0 | metrics.json | `results/pcstgnn_tuning_lam1_1p0_lam2_0p0/pcstgnn/seed_00042/metrics.json` | 2026-10-02 15:10:02 |
| 19 | pcstgnn_noisediag_lam1_0p0333 | predictions_test.csv | `results/pcstgnn_noisediag_lam1_0p0333/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-05 06:12:57 |
| 20 | pcstgnn_noisediag_lam1_0p0 | predictions_test.csv | `results/pcstgnn_noisediag_lam1_0p0/pcstgnn/seed_00042/predictions_test.csv` | 2026-10-05 01:46:00 |
| 21 | pcstgnn_noisediag_lam1_0p0333 | confusion_matrix.csv | `results/pcstgnn_noisediag_lam1_0p0333/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-05 06:12:57 |
| 22 | pcstgnn_noisediag_lam1_0p0 | confusion_matrix.csv | `results/pcstgnn_noisediag_lam1_0p0/pcstgnn/seed_00042/confusion_matrix.csv` | 2026-10-05 01:46:00 |
| 23 | pcstgnn_noisediag_lam1_0p0333 | metrics.json | `results/pcstgnn_noisediag_lam1_0p0333/pcstgnn/seed_00042/metrics.json` | 2026-10-05 06:12:56 |
| 24 | pcstgnn_noisediag_lam1_0p0 | metrics.json | `results/pcstgnn_noisediag_lam1_0p0/pcstgnn/seed_00042/metrics.json` | 2026-10-05 01:45:59 |
