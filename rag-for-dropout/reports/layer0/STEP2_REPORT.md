# Layer 0 step 2 report — seed 42, `val`, CPU, 2026-10-30

Written for: the author (decisions at the end). Numbers are copied from `reports/layer0/layer0_val_seed42.json` and `leakage_checks_seed42.json`.
Method fixed before the run: `DECISION_RULE.md` (commit `aa7454b`). Primary windows: post-fault (w ≥ 21); secondary: all windows. Held-out store = primary (O20).
`test` was not touched. Candidate sets (conformal) were not computed: the nominal coverage is an open O6 value.

## 0. The decision
**kNN agreement is KEPT in the gate** (pre-registered rule). D = mean over k of the within-k AUROC difference (probability + agreement) − (isotonic probability) = **+0.0112**, 95 % CI **[+0.0086, +0.0138]** (1000 run-bootstrap samples on cal-B; 58,900 windows per k). The CI lower end is above 0.
Every reading agrees (none would change the decision): each k alone, pooled over k, all windows, mask offsets 1 and 2, and the standard store (+0.040, inflated by near-duplicates). Nothing was changed after the first smoke run showed a 2-k preview (20 resamples, not a result).
Read it with three cautions:
1. The effect is small: AUROC 0.9074 → 0.9174 at k = 0 (+1.0 point), +0.9 to +1.3 points at other k. The rule tests the CI, not the size.
2. **The margin alone beats probability and agreement** (AUROC 0.9137 vs 0.9074 vs 0.8170 at k = 0), and at the same 95 % target the p + margin gate has *more* coverage than p + agreement (mean over k 0.7512 vs 0.7386).
   Post hoc (not pre-registered): agreement still adds beyond probability + margin (AUROC +0.0056, CI [+0.0037, +0.0074]); the three-feature gate (p, margin, agreement) has the same coverage at 95 % (0.7514) and +3.6 points at 98 % (0.6579 vs 0.6223).
3. One global threshold pair does **not** keep 95 % accuracy at k = 10 (93.3 % on cal-B, 93.2 % on cal-A). Per-k thresholds do (§ 4).
The O6 stop rule (coverage at 95 % below 50 %) is **not triggered**, but only just: the lowest value is 0.5015 (k = 10, per-k thresholds, p + agreement). The p-only gate is below it at k = 10 (0.483).
**Seeds 43 and 44: your rule says to ask before running them. Asking now (§ 8).**

## 1. Error-detection AUROC on cal-B, post-fault, held-out store (95 % CI, run bootstrap)
| signal | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |
|---|---|---|---|---|---|
| raw vote share | 0.9066 (0.897–0.916) | 0.9081 (0.899–0.917) | 0.9120 (0.904–0.920) | 0.9075 (0.901–0.914) | 0.8862 (0.881–0.891) |
| isotonic probability (a) | 0.9074 (0.897–0.917) | 0.9088 (0.900–0.917) | 0.9125 (0.905–0.920) | 0.9073 (0.901–0.913) | 0.8853 (0.880–0.890) |
| margin (top-1 − top-2) | 0.9137 (0.904–0.923) | 0.9141 (0.905–0.922) | 0.9177 (0.910–0.925) | 0.9121 (0.906–0.918) | 0.8889 (0.884–0.894) |
| kNN agreement | 0.8170 (0.801–0.832) | 0.8387 (0.827–0.850) | 0.8527 (0.842–0.862) | 0.8572 (0.850–0.864) | 0.8164 (0.810–0.823) |
| neighbour error rate | 0.8125 (0.796–0.827) | 0.8211 (0.809–0.832) | 0.8255 (0.815–0.836) | 0.8151 (0.807–0.823) | 0.7624 (0.755–0.769) |
| nearest-neighbour similarity | 0.4157 (0.392–0.438) | 0.5124 (0.492–0.532) | 0.5768 (0.558–0.596) | 0.6713 (0.658–0.684) | 0.7218 (0.714–0.729) |
| probability + agreement (b) | 0.9174 (0.908–0.926) | 0.9205 (0.912–0.928) | 0.9248 (0.918–0.931) | 0.9203 (0.914–0.926) | 0.8943 (0.889–0.899) |
| (standard store) kNN agreement | 0.9450 | 0.9354 | 0.9274 | 0.8955 | 0.8310 |
| (standard store) probability + agreement | 0.9690 | 0.9608 | 0.9556 | 0.9364 | 0.8998 |
Error rate (positive class) of the post-fault windows: k = 0: 0.119, k = 1: 0.132, k = 2: 0.146, k = 5: 0.188, k = 10: 0.299. AUPRC and all-window AUROC are in the JSON.
The raw share and the isotonic probability rank almost identically, so the isotonic plateaus (ties) are not what the combination gains from.
Nearest-neighbour similarity is below 0.5 at k = 0 (0.416): at clean input a *more* similar neighbour goes with *more* errors, so it is not a usable OOD signal there.

## 2. Decision statistic by k (AUROC (b) − (a), 95 % CI)
| k | (a) isotonic | (b) p + agreement | difference |
|---|---|---|---|
| 0 | 0.9074 | 0.9174 | +0.0100 [+0.0063, +0.0138] |
| 1 | 0.9088 | 0.9205 | +0.0116 [+0.0083, +0.0151] |
| 2 | 0.9125 | 0.9248 | +0.0123 [+0.0093, +0.0153] |
| 5 | 0.9073 | 0.9203 | +0.0130 [+0.0106, +0.0154] |
| 10 | 0.8853 | 0.8943 | +0.0091 [+0.0071, +0.0110] |
| pooled | 0.9083 | 0.9191 | +0.0108 [+0.0083, +0.0132] |
Secondary mean over k: standard store +0.0401 [+0.0353, +0.0451]; all windows heldout +0.0051 [+0.0039, +0.0063]; offset1 heldout +0.0112 [+0.0086, +0.0136]; offset2 heldout +0.0113 [+0.0087, +0.0139].

## 3. Layer 0 per k (`val`, w0020 excluded, offset 0; held-out store primary)
| | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |
|---|---|---|---|---|---|
| STGNN top-1, post-fault | 0.8807 | 0.8678 | 0.8538 | 0.8106 | 0.6999 |
| STGNN top-1, all windows | 0.9419 | 0.9356 | 0.9288 | 0.9077 | 0.8538 |
| macro-F1, all windows (w0020 excluded) | 0.8855 | 0.8735 | 0.8604 | 0.8200 | 0.7170 |
| ECE isotonic, cal-B post-fault | 0.0071 | 0.0052 | 0.0128 | 0.0235 | 0.0479 |
| ECE raw vote share, cal-B post-fault | 0.1200 | 0.1243 | 0.1270 | 0.1475 | 0.1764 |
| AURC isotonic, cal-B post-fault | 0.0216 | 0.0244 | 0.0274 | 0.0410 | 0.0938 |
| risk at 80 % coverage, cal-B post-fault | 0.0381 | 0.0619 | 0.0633 | 0.0884 | 0.1855 |
| kNN vote (k_nn = 10), held-out, post-fault | 0.8551 | 0.8438 | 0.8312 | 0.7820 | 0.6567 |
| kNN vote (k_nn = 10), standard, post-fault | 0.9252 | 0.9039 | 0.8830 | 0.8152 | 0.6742 |
| kNN vote (k_nn = 10), held-out, all | 0.9294 | 0.9239 | 0.9178 | 0.8938 | 0.8326 |
Calibration does **not** hold as k grows: the k = 0 isotonic map has ECE 0.007 at k = 0 and 0.048 at k = 10 (post-fault). The raw share is far worse (0.12–0.18). Per-k thresholds absorb this.
The held-out kNN vote is below the STGNN's own accuracy at every k (post-fault 0.855 vs 0.881 at k = 0; 0.657 vs 0.700 at k = 10).

## 4. O6 gate thresholds (selected on cal-B post-fault; checked on cal-A)
Gate as the rule defines it: CONFIDENT if isotonic p ≥ τ_p **and** kNN agreement (held-out, 10 neighbours) ≥ τ_k.
Target 95 % accuracy on CONFIDENT windows — per-k thresholds:
| k | τ_p | τ_k | coverage cal-B | accuracy cal-B | coverage cal-A | accuracy cal-A | error recall cal-B |
|---|---|---|---|---|---|---|---|
| 0 | 0.6223 | 0.4 | 0.8392 | 0.9503 | 0.8385 | 0.9501 | 0.650 |
| 1 | 0.5567 | 0.6 | 0.8245 | 0.9502 | 0.8258 | 0.9479 | 0.688 |
| 2 | 0.5454 | 0.6 | 0.8033 | 0.9501 | 0.8050 | 0.9472 | 0.726 |
| 5 | 0.5648 | 0.7 | 0.7244 | 0.9501 | 0.7230 | 0.9488 | 0.808 |
| 10 | 0.6260 | 0.9 | 0.5015 | 0.9506 | 0.5013 | 0.9490 | 0.917 |
One global pair (τ_p = 0.6223, τ_k = 0.5): pooled coverage 0.7381, pooled accuracy 0.9503; accuracy by k (cal-B): 0: 0.953, 1: 0.954, 2: 0.954, 5: 0.952, 10: 0.933.
Target 98 % accuracy on CONFIDENT windows — per-k thresholds:
| k | τ_p | τ_k | coverage cal-B | accuracy cal-B | coverage cal-A | accuracy cal-A | error recall cal-B |
|---|---|---|---|---|---|---|---|
| 0 | 0.7213 | 0.3 | 0.7589 | 0.9800 | 0.7585 | 0.9780 | 0.873 |
| 1 | 0.7212 | 0.3 | 0.7352 | 0.9801 | 0.7370 | 0.9783 | 0.889 |
| 2 | 0.7216 | 0.3 | 0.7130 | 0.9800 | 0.7104 | 0.9787 | 0.903 |
| 5 | 0.6794 | 0.9 | 0.6211 | 0.9807 | 0.6193 | 0.9820 | 0.936 |
| 10 | 0.8368 | 0.9 | 0.3678 | 0.9801 | 0.3691 | 0.9797 | 0.976 |
One global pair (τ_p = 0.6794, τ_k = 0.9): pooled coverage 0.6450, pooled accuracy 0.9803; accuracy by k (cal-B): 0: 0.982, 1: 0.983, 2: 0.983, 5: 0.981, 10: 0.969.
Mean coverage over k, per-k thresholds, cal-B (95 % / 98 % target):
| gate | 95 % | 98 % |
|---|---|---|
| p only | 0.7006 | 0.6120 |
| p + margin | 0.7512 | 0.6223 |
| **p + agreement, held-out (the rule's gate)** | 0.7386 | 0.6392 |
| p + agreement, standard store (secondary) | 0.7838 | 0.7007 |
| p + margin + agreement (post hoc) | 0.7514 | 0.6579 |
Optimism: thresholds fitted on cal-B sit at the boundary; on cal-A the accuracy of CONFIDENT windows is up to 0.3 points lower at the 95 % target (0.947 at k = 2; 0.948 at k = 1) and within 0.2 points at 98 %.

## 5. Leakage tests and the FAISS case store
- Runs: train 28,875, val 6,200, test 6,175; overlaps train∩val 0, train∩test 0, val∩test 0. Store: 2,888 runs, all in train; in val or test: 0.
- Simulation seeds: 41,250 distinct over 41,250 runs; store seeds also in val/test: 0. Building the store raises `LeakageError` otherwise.
- Case store `eval/case_store/seed42_sub10/` (FAISS exact inner product, 115,520 entries, fingerprint `c95e8700c51a62c3`); saved store equals the rebuilt one: True.
- FAISS vs the exact torch search of Layer 0 (3,000 val queries): votes identical 100.0 % (standard) and 100.0 % (held-out); max agreement difference 2.4e-08. Own-scenario neighbours in the top 20: 19,757 (standard), 0 (held-out).
- Sanity run (§ 13.9; 4,836 val windows of 124 runs put into the store; val, not test): k_nn = 1 accuracy 0.9739 (clean) → **1.0000** (queries in store) → 0.9737 (leave-one-run-out).
  At k_nn = 10 the vote reaches 0.9857 (0.9707 post-fault), not ~1.0: 9 of the 10 neighbours are other windows, some across the fault-onset label change. Leave-one-run-out gives 0.9661 (clean 0.9663); scenario exclusion 0.8731 post-fault (clean held-out 0.8744).
- Tests: `tests/gridsense/test_leakage.py` (synthetic: forbidden run or seed, exclusion, fingerprint, stale store, wrong checkpoint) and `test_leakage_real.py` (slow). 83 tests pass.

## 6. Compute (S4.4 queue is training R4 seed 44)
Baseline epoch time 2,551.8 s (epochs 3–13). Epoch 14, which contained the whole neighbour search, took 2,596.5 s (+1.75 %; limit +10 %). Queue CPU rate: 145.3 % before my first job; minimum 142.7 % afterwards (the mean is lifted by end-of-epoch validation). 4 threads kept; CPU only; no GPU, no paid call.

## 7. Limits
- Store = 10 % train subsample, k = 0 clean cases only; seed 42; `val` only. The gate is not frozen and was not evaluated on `test`.
- Only the store is held out; the encoder saw every scenario. Held-out numbers are the honest ones for new scenarios, but not a scenario-held-out model.
- The combination and the isotonic map were fitted on cal-A, thresholds on cal-B; both halves were seen by checkpoint selection (small optimistic bias).
- The 98 % target at k = 10 has coverage 0.37–0.38 (per-k). Candidate sets and error recall at the LOW_CONFIDENCE/REJECT split are not done.

## 8. Decisions for the author
1. **Seeds 43 and 44** (about 34 min each, CPU): run them? Your rule requires asking because agreement was kept.
2. **Gate composition (O21).** Keep exactly the rule's gate (p + agreement), or add the margin (p + margin + agreement, post hoc)? I recommend the three-feature gate: same coverage at 95 %, +3.6 points at 98 %, and it does not drop the margin that is the strongest single signal. I did not implement it as the gate.
3. **Thresholds (O22).** Per-k thresholds (the gate knows k from the sensor mask), because one global pair breaks the 95 % target at k = 10. I recommend per-k.
4. **Next:** freeze the gate and evaluate it once on `test`; build the full train store (about 0.6 GB; needs your approval, O15); choose the conformal coverage for candidate sets (proposal 90 %, O23).
