# S2 report — seed 42, CPU, 2026-10-28

Written for: the author (decisions at the end). Every number is copied from `reports/s2/*.json` (written by
`src/gridsense/eval/`). Data: `val` and `test` of `dataset_fullscale_50`; checkpoint `seed_00042` (dropout-aware RGSAGE-Max).
All metrics drop w0020 (`unlabelable_window_mask`). Bootstrap CIs resample whole runs (2,000 resamples (P)).

## 1. Gates
| Gate | Rule (author, 2026-10-28) | Result |
|---|---|---|
| k = 0 predictions | differ from the paper only where vote gap < 2e-3 | **pass**: 6 of 247,000 test windows differ, max gap 1.835e-03, labels equal |
| k = 0 macro-F1 | — | test 0.8515212 vs 0.8515282 (diff -7.04e-06); val 0.85549072 vs 0.85549097 (diff -2.6e-07) |
| Determinism | re-run is byte-identical | **pass**: votes, latents, labels, window IDs identical (test, k = 0) |
| Mask generator | k > 0 masks = paper's seeded generator | **pass**: 480 of 480 `apply_sensor_dropout_batch` calls identical (k, seed, dropped nodes); F1 identical in 12 of 12 cells on 1,280 windows (`mask_check_seed42.json`) |
| k > 0 macro-F1 | within 1e-3 of `degradation_curves.csv` | **not met at k = 10** (see § 2) |

The 6 k = 0 window IDs (CPU class / paper class / gap): `21_ABG_0p1_0042_w0020` 19/5 1.2e-5; `21_BCG_0p1_0030_w0020` 5/0 2.2e-4;
`51_AG_0p1_0030_w0020` 5/0 1.2e-3; `62_BG_10p0_0034_w0031` 14/15 1.9e-6; `65_CA_10p0_0031_w0029` 14/15 8.3e-5;
`93_BCG_0p1_0013_w0020` 5/0 1.835e-3. The rule was set after the data was seen (limitation 7 in `dropout-rag-intro.md`).

## 2. k > 0 macro-F1 on `test` (mean of 3 mask offsets, batch 32)
| k | CPU (this run) | stored paper curve | abs diff | within 1e-3 |
|---|---|---|---|---|
| 1 | 0.839662 | 0.839839 | 1.77e-04 | yes |
| 2 | 0.828166 | 0.827873 | 2.93e-04 | yes |
| 5 | 0.786952 | 0.787359 | 4.07e-04 | yes |
| 10 | 0.690345 | 0.688959 | 1.39e-03 | **no** |

Why k = 10 may miss: the mask draw depends on the batch size, and nothing on disk records the batch size of the stored
curve (`PC-STGNN/02_STAGE_A_audit.md` S0.1). Diagnostic (same code, k = 10, only the batch size changes):
| batch size | 3-offset mean F1 | abs diff to stored 0.688959 |
|---|---|---|
| batch 32 (protocol) | 0.690345 | 1.39e-03 |
| batch 64 | 0.689869 | 9.10e-04 |
| batch 16 | 0.689545 | 5.86e-04 |
The three means spread by 8.0e-4. One mask offset has a standard deviation of 1.31e-3 (9 values, ddof = 1), so the difference of two
independent 3-offset means has a standard deviation of 1.07e-3: the 1e-3 tolerance is about 0.9 standard deviations of pure
mask-draw noise at k = 10. This is evidence, not proof: the stored batch size is unknown.

## 3. Report window (O12), `val`, k = 0
Rule: earliest w ≥ 21 within 1 pp of the max over w ≥ 21. Max 0.9097 at w = 32; threshold 0.8997.
**Chosen index: w = 25** (accuracy 0.9061, 95 % CI 0.8990 to 0.9135; 5 post-fault samples).
| w | 21 | 22 | 23 | 24 | **25** | 28 | 32 | 36 | 39 |
|---|---|---|---|---|---|---|---|---|---|
| acc | 0.5718 | 0.8106 | 0.8832 | 0.8952 | 0.9061 | 0.9076 | 0.9097 | 0.9040 | 0.8879 |
The same rule gives w = 25 at k = 1, 2, 5 and 10 (`s2_window_k_k0_val.json`). Pre-fault windows (w < 20) are 1.0000.

## 4. Latent method (O4, O15)
Store: 10 % train-run subsample (115,520 windows, 2,888 whole runs, stratified by scenario). Queries: 241,800 val windows.
Scaler and PCA fitted on the subsample only (256 dims explain 0.9985 of the standardized variance).
| method | dim | k_nn = 5 | k_nn = 10 (95 % CI) | k_nn = 20 | post-fault (w ≥ 21), k_nn = 10 |
|---|---|---|---|---|---|
| a_concat | 1600 | 0.9652 | 0.9627 (0.9602–0.9650) | 0.9610 | 0.9235 |
| b_mean_max | 128 | 0.9660 | 0.9635 (0.9611–0.9657) | 0.9616 | 0.9252 |
| c_concat_pca256 | 256 | 0.9657 | 0.9629 (0.9604–0.9651) | 0.9613 | 0.9238 |
Chosen by the pre-set rule: **b_mean_max** (mean + max, 128-d). Paired lead over c_concat_pca256: 0.066 pp, CI -0.002 to 0.132 pp,
so the three methods are a statistical tie; (b) is also the cheapest (41 s vs 230 s for (a)). Post-fault accuracy agrees.
Context, not a result: the STGNN's own top-1 on the same windows is 0.9419 (all) and 0.8807 (post-fault). The kNN vote on a
train subsample is higher, but train and val share all 825 scenarios (P5), so neighbours can be near-duplicates.
Do not read this as the kNN beating the STGNN before the scenario-held-out control has run.

## 5. Calibration (O2), `val` split by run: cal-A fit, cal-B compare
cal-A 3,100 runs / 120,900 windows; cal-B 3,100 runs / 120,900 windows (stratified by scenario). Argmax never changes.
| on cal-B | ECE all | Brier all | ECE post-fault | Brier post-fault |
|---|---|---|---|---|
| raw_top1_share | 0.0585 | 0.0450 | 0.1200 | 0.0924 |
| platt | 0.0145 | 0.0356 | 0.0271 | 0.0730 |
| isotonic | 0.0022 | 0.0343 | 0.0071 | 0.0703 |
Lower ECE: **isotonic**. Top-1 accuracy on cal-B 0.9419 (post-fault 0.8807).
Margin (top-1 − top-2 vote share) on cal-B: mean 0.8216, median 1.0000; saved with the calibrated
probabilities in `eval/calibration_k0_val.npz` (Git-ignored). ECE uses 15 equal-mass bins (P). Pooled ECE is flattered by the
trivial pre-fault windows; the post-fault columns are the ones to use for the gate.

## 6. Compute and the S4.4 queue
| | S4.4 train time per epoch | queue CPU |
|---|---|---|
| Before any S2 job (epochs 2–6) | 2,586.4 s (2,574–2,591) | 144.1 % |
| Epoch 7 (determinism re-run, comparisons) | 2,609.5 s (+0.90 %) | — |
| Epoch 8 (almost all S2 jobs) | 2,658.4 s (+2.79 %) | mean 150 %, min 144 % since 16:00Z |
Threshold was +10 %: not reached, 4 threads kept. Extraction times: k = 0 test 98 s, val 98 s, train subsample 45 s; k > 0 test 889 s,
val 898 s. Peak RAM 2.6 GB. CPU only, `nice -n 19`. The monitor shows short dips (to 109 %) when I ran `cmp` and pandas jobs (disk/RAM), not during the extractions.

## 7. Not done in S2 (and why)
- Layer 0 metrics (gate, coverage, selective risk, AURC, error-detection AUROC/AUPRC, candidate sets): not in your S2 order; O6 (gate target) is open.
- Case store index (FAISS), leakage assertion test, leakage sanity run, scenario-held-out control: not built.
- Seeds 43 and 44: extraction alone is about 34 min each (k = 0 ≈ 4 min + k > 0 ≈ 30 min), above your 30-min rule.

## 8. Decisions for the author
1. k = 10 macro-F1 is 1.39e-3 from the stored curve (tolerance 1e-3). Accept as mask-draw noise (recommended; diagnostic above), set 2e-3 at k = 10, or re-evaluate at another batch size.
2. Seeds 43/44: run (about 34 min each) or skip.
3. Next step: Layer 0 metrics and the case store, including the scenario-held-out control (O6 needed for the gate target).
