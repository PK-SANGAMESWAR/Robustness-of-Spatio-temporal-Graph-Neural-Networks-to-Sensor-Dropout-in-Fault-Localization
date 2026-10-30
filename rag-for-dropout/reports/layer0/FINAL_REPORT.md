# Layer 0 final report — frozen gate_v1, one run on `test` (seed 42), seeds 43 and 44

Written for: the author (decisions at the end). Numbers are copied from `reports/layer0/*.json` by `src/gridsense/eval/final_report.py`.
Frozen gate: `configs/gate_v1.json`, sha256 `693ff351157dc70ee748657b92c036d1b5e0ab672aeed1fdcab099457ac8b497`; frozen in commit `b7a80b9` before any run on test. Run on test once: 2026-10-30T07:43:41Z.
**The three-feature composition (probability, margin, kNN agreement) was selected on val after the pre-registered test** (`DECISION_RULE.md`, `aa7454b`), not before it.
Windows: post-fault (w ≥ 21), mask offset 0, held-out full train store (O20). 95 % CIs: bootstrap by run, 1000 samples.

## 0. Result
- **Coverage at k = 10 on test: 0.5126** (95 % CI 0.5042–0.5207); not below 50 %.
- CONFIDENT accuracy on test against the 95 % target: k = 0: 0.9493, k = 1: 0.9487, k = 2: 0.9502, k = 5: 0.9491, k = 10: 0.9488.
- At the 95 % point the frozen thresholds switch kNN agreement off (τ_k = 0) at k = 0, 1, 2, 5, 10 and the probability off (τ_p = 0) at k = 0, 1, 5, 10: the gate is effectively **margin-only** there. Agreement is active at the 98 % point.
- The held-out kNN vote (10 neighbours) is below the STGNN's own accuracy at k = 0, 1, 2, 5, 10 (§ 4).
- 90 % candidate sets: coverage over all post-fault windows k = 0: 0.899, k = 1: 0.898, k = 2: 0.898, k = 5: 0.897, k = 10: 0.900; over the LOW_CONFIDENCE windows only k = 0: 0.547, k = 1: 0.575, k = 2: 0.592, k = 5: 0.671, k = 10: 0.807 (§ 3).
- **Seed 43, frozen seed-42 thresholds:** CONFIDENT accuracy at the 95 % point 0.9180–0.9318 (target 0.95), coverage 0.511–0.878; coverage at k = 10: 0.5106: **the frozen thresholds do not transfer to another training seed** (below the target at every k).
- **Seed 44, frozen seed-42 thresholds:** CONFIDENT accuracy at the 95 % point 0.8971–0.9221 (target 0.95), coverage 0.497–0.854; coverage at k = 10: 0.4966 (**below 50 %**): **the frozen thresholds do not transfer to another training seed** (below the target at every k).
  Thresholds on the raw margin belong to the checkpoint they were selected on; a new checkpoint needs its own thresholds (fitted on its own cal split). New thresholds were NOT selected for seeds 43 and 44 (your instruction).

## 1. The frozen gate (per k; CONFIDENT iff p ≥ τ_p and margin ≥ τ_m and agreement ≥ τ_k)
| k | operating point | τ_p | τ_m | τ_k | conformal q | isotonic knots |
|---|---|---|---|---|---|---|
| 0 | acc95 | 0.0000 | 0.1459 | 0.0 | 0.7003 | 126 |
| 0 | acc98 | 0.6144 | 0.2142 | 0.4 | 0.7003 | 126 |
| 1 | acc95 | 0.0000 | 0.1459 | 0.0 | 0.7261 | 140 |
| 1 | acc98 | 0.5585 | 0.1925 | 0.8 | 0.7261 | 140 |
| 2 | acc95 | 0.5424 | 0.1368 | 0.0 | 0.7518 | 120 |
| 2 | acc98 | 0.0000 | 0.2191 | 0.7 | 0.7518 | 120 |
| 5 | acc95 | 0.0000 | 0.1633 | 0.0 | 0.8201 | 176 |
| 5 | acc98 | 0.6323 | 0.2109 | 1.0 | 0.8201 | 176 |
| 10 | acc95 | 0.0000 | 0.2518 | 0.0 | 0.9054 | 180 |
| 10 | acc98 | 0.0000 | 0.3939 | 0.7 | 0.9054 | 180 |
A k or dropout pattern that is not fitted (k ∉ {0, 1, 2, 5, 10}, or a pattern other than random) returns REJECT with `UNSUPPORTED_DROPOUT_PATTERN`.

## 2. The single run on `test` vs the selection set (cal-B), 95 % point
| k | STGNN accuracy (test) | coverage cal-B | coverage test (CI) | CONFIDENT accuracy cal-B | CONFIDENT accuracy test (CI) | error recall test |
|---|---|---|---|---|---|---|
| 0 | 0.8760 | 0.8500 | 0.8449 (0.838–0.852) | 0.9524 | 0.9493 (0.945–0.954) | 0.654 |
| 1 | 0.8632 | 0.8300 | 0.8247 (0.818–0.832) | 0.9513 | 0.9487 (0.944–0.953) | 0.691 |
| 2 | 0.8497 | 0.8107 | 0.8008 (0.793–0.808) | 0.9500 | 0.9502 (0.946–0.954) | 0.734 |
| 5 | 0.8023 | 0.7400 | 0.7310 (0.723–0.739) | 0.9504 | 0.9491 (0.945–0.953) | 0.812 |
| 10 | 0.6943 | 0.5200 | 0.5126 (0.504–0.521) | 0.9503 | 0.9488 (0.946–0.952) | 0.914 |
98 % point on test (thresholds from cal-B): k = 0: coverage 0.772, accuracy 0.9782; k = 1: coverage 0.745, accuracy 0.9786; k = 2: coverage 0.719, accuracy 0.9799; k = 5: coverage 0.628, accuracy 0.9799; k = 10: coverage 0.376, accuracy 0.9804.
Outcomes on test (95 % point): k = 0: {'CONFIDENT': 99132, 'LOW_CONFIDENCE': 18193, 'REJECT': 0}; k = 10: {'CONFIDENT': 60146, 'LOW_CONFIDENCE': 57179, 'REJECT': 0}. No window was rejected: every test window has a fitted k and the random pattern.

## 3. Conformal 90 % candidate sets (per k; fitted on cal-A)
| k | cal-B coverage | cal-B mean size | test coverage (CI) | test mean size | test LOW_CONFIDENCE coverage | LOW_CONFIDENCE mean size | n LOW_CONFIDENCE |
|---|---|---|---|---|---|---|---|
| 0 | 0.9048 | 1.06 | 0.8986 (0.893–0.905) | 1.05 | 0.547 | 1.09 | 18,193 |
| 1 | 0.9019 | 1.09 | 0.8980 (0.892–0.904) | 1.09 | 0.575 | 1.19 | 20,564 |
| 2 | 0.9017 | 1.14 | 0.8979 (0.893–0.903) | 1.14 | 0.592 | 1.28 | 23,368 |
| 5 | 0.9029 | 1.40 | 0.8971 (0.893–0.902) | 1.41 | 0.671 | 1.81 | 31,558 |
| 10 | 0.9010 | 2.35 | 0.9003 (0.897–0.904) | 2.37 | 0.807 | 2.99 | 57,179 |
The 90 % guarantee is marginal (over all post-fault windows). Among the LOW_CONFIDENCE windows, which the sets are shown for, coverage is much lower: a candidate set on a LOW_CONFIDENCE report is not a 90 % set.

## 4. Layer 0 on test (held-out store primary), per k
| | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |
|---|---|---|---|---|---|
| ECE isotonic | 0.0084 | 0.0056 | 0.0064 | 0.0086 | 0.0057 |
| ECE raw vote share | 0.1211 | 0.1247 | 0.1292 | 0.1443 | 0.1726 |
| AURC isotonic | 0.0223 | 0.0252 | 0.0286 | 0.0424 | 0.0944 |
| risk at 80 % coverage | 0.0624 | 0.0618 | 0.0672 | 0.0890 | 0.1948 |
| kNN vote (10), held-out | 0.8311 | 0.8231 | 0.8126 | 0.7695 | 0.6458 |
| kNN vote (10), standard (secondary) | 0.9582 | 0.9274 | 0.8973 | 0.8166 | 0.6669 |
Error-detection AUROC (95 % CI):
| signal | k = 0 | k = 1 | k = 2 | k = 5 | k = 10 |
|---|---|---|---|---|---|
| raw_vote_share | 0.9100 (0.904–0.917) | 0.9117 (0.906–0.918) | 0.9123 (0.907–0.918) | 0.9110 (0.907–0.915) | 0.8850 (0.881–0.889) |
| isotonic_prob | 0.9109 (0.905–0.917) | 0.9125 (0.907–0.918) | 0.9130 (0.908–0.918) | 0.9111 (0.907–0.915) | 0.8850 (0.881–0.889) |
| margin | 0.9152 (0.909–0.922) | 0.9160 (0.910–0.922) | 0.9166 (0.912–0.922) | 0.9150 (0.911–0.919) | 0.8868 (0.883–0.891) |
| knn_agreement_heldout | 0.7478 (0.735–0.761) | 0.7806 (0.771–0.791) | 0.7999 (0.791–0.808) | 0.8255 (0.820–0.831) | 0.7959 (0.791–0.801) |
| neighbour_error_rate_heldout | 0.7511 (0.739–0.764) | 0.7672 (0.758–0.777) | 0.7783 (0.771–0.786) | 0.7783 (0.773–0.784) | 0.7315 (0.727–0.736) |
| knn_agreement_standard (secondary) | 0.9825 (0.979–0.986) | 0.9558 (0.952–0.959) | 0.9333 (0.930–0.937) | 0.8829 (0.879–0.887) | 0.8137 (0.809–0.818) |
| logistic_combination_p+margin+agreement | 0.9163 (0.910–0.923) | 0.9179 (0.913–0.924) | 0.9197 (0.915–0.925) | 0.9227 (0.919–0.927) | 0.8977 (0.894–0.901) |

## 5. Seeds 43 and 44 (frozen thresholds; isotonic map and conformal quantile refitted per seed on cal-A; test windows)
| seed | k | STGNN accuracy | coverage (95 % point) | CONFIDENT accuracy | ECE isotonic | conformal coverage | AUROC margin | AUROC isotonic | AUROC agreement |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0 | 0.8760 | 0.8449 | 0.9493 | 0.0084 | 0.8986 | 0.9152 | 0.9109 | 0.7478 |
| 42 | 1 | 0.8632 | 0.8247 | 0.9487 | 0.0056 | 0.8980 | 0.9160 | 0.9125 | 0.7806 |
| 42 | 2 | 0.8497 | 0.8008 | 0.9502 | 0.0064 | 0.8979 | 0.9166 | 0.9130 | 0.7999 |
| 42 | 5 | 0.8023 | 0.7310 | 0.9491 | 0.0086 | 0.8971 | 0.9150 | 0.9111 | 0.8255 |
| 42 | 10 | 0.6943 | 0.5126 | 0.9488 | 0.0057 | 0.9003 | 0.8868 | 0.8850 | 0.7959 |
| 43 | 0 | 0.8631 | 0.8781 | 0.9234 | 0.0109 | 0.8957 | 0.9057 | 0.9028 | 0.7379 |
| 43 | 1 | 0.8518 | 0.8594 | 0.9210 | 0.0075 | 0.8973 | 0.9022 | 0.8994 | 0.7605 |
| 43 | 2 | 0.8387 | 0.8395 | 0.9192 | 0.0104 | 0.8985 | 0.9007 | 0.8972 | 0.7782 |
| 43 | 5 | 0.7966 | 0.7587 | 0.9180 | 0.0080 | 0.8991 | 0.8877 | 0.8845 | 0.7932 |
| 43 | 10 | 0.6904 | 0.5106 | 0.9318 | 0.0098 | 0.8978 | 0.8608 | 0.8627 | 0.7665 |
| 44 | 0 | 0.8558 | 0.8541 | 0.9221 | 0.0116 | 0.8961 | 0.8787 | 0.8739 | 0.7475 |
| 44 | 1 | 0.8428 | 0.8371 | 0.9176 | 0.0086 | 0.8971 | 0.8783 | 0.8734 | 0.7614 |
| 44 | 2 | 0.8272 | 0.8177 | 0.9121 | 0.0103 | 0.8980 | 0.8765 | 0.8715 | 0.7736 |
| 44 | 5 | 0.7773 | 0.7393 | 0.9022 | 0.0084 | 0.9020 | 0.8687 | 0.8652 | 0.7859 |
| 44 | 10 | 0.6620 | 0.4966 | 0.8971 | 0.0062 | 0.8994 | 0.8386 | 0.8429 | 0.7703 |

## 6. Leakage checks on the full store
Train / val / test runs disjoint (overlaps 0, 0, 0); store = 28,875 train runs, 0 in val or test; seeds shared with val/test: 0; fingerprint matches: True.
FAISS vs the exact torch search (1,000 val queries): votes identical 100.0 % / 100.0 %; own-scenario neighbours in the held-out top 20: 0; in the standard top 20: 16,665 of 20,000.

## 7. Compute (S4.4 queue, R4 seed 44)
baseline 2551.8 s (epochs 3–13), limit 2807 s; epochs 14 onward: 14: 2597 s (+1.8 %); 15: 2567 s (+0.6 %); 16: 2547 s (-0.2 %); 17: 2761 s (+8.2 %); 18: 2791 s (+9.4 %); 19: 2727 s (+6.9 %); 20: 2625 s (+2.9 %); 21: 2615 s (+2.5 %); 22: 2652 s (+3.9 %). Threads: 4, then 2 from epoch 19 (precaution at +9.4 %). CPU only; no GPU, no paid call.

## 8. Limits
- Mask offset 0 only; one realisation per window and k. Post-fault windows only for the gate and for every neighbour metric.
- Thresholds fitted on cal-B sit at the selection boundary; the test run is the unbiased check. Only the store is held out; the encoder saw every scenario.
- `test` was used once for the frozen gate (seed 42). Seeds 43 and 44 are reported test numbers with frozen thresholds, not new selections.

## 9. For the author
1. **The gate is effectively margin-only at its 95 % point.** Agreement and (at four k) the probability add nothing there; they matter at the 98 % point. gate_v1 stays as frozen and tested. A simpler gate_v2 would be a new version, and `test` has now been used, so a v2 must not be judged on the same test numbers.
2. **Thresholds belong to a checkpoint.** On seeds 43 and 44 the frozen thresholds miss the 95 % target by 2–5 points (§ 5). Any other checkpoint needs its own cal-A / cal-B fit and its own frozen run. Which checkpoint is the system: seed 42?
3. **Candidate sets on LOW_CONFIDENCE reports are not 90 % sets** (§ 3: 55–81 % at seed 42). Options for the report text: state the measured conditional coverage; fit the conformal quantile on LOW_CONFIDENCE windows only (new fit and check); or use a fixed top-k set.
4. **Next stage: Layer 1 (document ingestion and retrieval).** Ready: `corpus/v1` (the revised paper stays a local copy, SHA-256 in `MANIFEST.json`), the span-based golden schema, the sweep skills. Open before it: O7 (extra corpus sources), O1 (OpenAI models and prices, needed only for the paid steps), O13 (human labels), and the golden retrieval drafts for your review.
