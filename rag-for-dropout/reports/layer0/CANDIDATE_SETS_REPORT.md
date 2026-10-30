# Candidate sets for LOW_CONFIDENCE windows — frozen run on test (seed 42)

Written for: the author. Numbers are copied from `reports/layer0/candidate_sets_test_run_seed42.json` and `configs/candidate_sets_v1.json`.
Component `candidate_sets_v1`, sha256 `686e7d0edd70f5b8a59b8f33b2dad43e1acb395345ee808abdbe187299454980`, frozen in commit `9afd240` before its run. `gate_v1` is unchanged (sha256 `693ff351157d…`).
**This was the second use of the test split, for the candidate-set component only** (`TEST_USE.md`).

## Method (O26)
The quantile is fitted per k on the cal-A windows that `gate_v1` labels LOW_CONFIDENCE (95 % point), post-fault, mask offset 0. Nonconformity = 1 − vote share of the true class; the set holds the classes with vote share ≥ 1 − q.
The 90 % target therefore holds for the LOW_CONFIDENCE windows, which is where a set is shown. CONFIDENT windows keep a single location; REJECT windows have no set.

## Result (LOW_CONFIDENCE windows only)
| k | windows (share of post-fault) | cal-B coverage | cal-B mean size | test coverage (95 % CI) | test mean size (CI) | median size | old all-window sets: coverage / size |
|---|---|---|---|---|---|---|---|
| 0 | 18,193 (0.155) | 0.9048 | 3.08 | 0.9058 (0.901–0.910) | 3.13 (3.06–3.20) | 3 | 0.547 / 1.09 |
| 1 | 20,564 (0.175) | 0.9057 | 3.20 | 0.9073 (0.903–0.911) | 3.24 (3.18–3.30) | 3 | 0.575 / 1.19 |
| 2 | 23,368 (0.199) | 0.9066 | 3.45 | 0.9101 (0.906–0.914) | 3.49 (3.43–3.54) | 4 | 0.592 / 1.28 |
| 5 | 31,558 (0.269) | 0.9053 | 3.70 | 0.9034 (0.899–0.907) | 3.72 (3.67–3.76) | 4 | 0.671 / 1.81 |
| 10 | 57,179 (0.487) | 0.9011 | 4.27 | 0.8999 (0.896–0.904) | 4.27 (4.24–4.30) | 4 | 0.807 / 2.99 |

Covering 90 % of the LOW_CONFIDENCE windows takes about 3 candidates at k = 0 and about 4 at k = 10 (median 3–4). The earlier all-window sets were smaller (1.1–3.0) but covered only 55–81 % of these windows.
At k = 10 about half of the post-fault windows are LOW_CONFIDENCE (0.487), so the sets matter most at high dropout.

## Statement on the gate (O27)
`gate_v1` is kept; there is no gate_v2. **At the 95 % target the selected thresholds use only the margin** (τ_k = 0 at every k; τ_p = 0 at k = 0, 1, 5, 10; τ_p = 0.5424 at k = 2). **kNN agreement has an effect only at the 98 % target.**

## Limits
Seed 42 only (deployment rule: each new checkpoint needs its own fit and frozen run). Mask offset 0. Held-out neighbours from the full train store decide which windows are LOW_CONFIDENCE.
