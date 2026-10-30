# TEST_USE.md — every use of the `test` split in the GridSense track

Append only. The test split is for reported numbers of frozen components. A use that follows a choice made on val is listed with its frozen config hash.

| # | Date (UTC) | Component | Config sha256 | What | Result file |
|---|---|---|---|---|---|
| 1 | 2026-10-30 07:43 | `gate_v1` (seed 42) | `693ff351157d…` | One run of the frozen gate; the gate is not changed afterwards | `reports/layer0/test_run_seed42.json` |
| 1b | 2026-10-30 | `gate_v1` thresholds on seeds 43, 44 | `693ff351157d…` | Layer 0 reported numbers with frozen thresholds (isotonic and conformal refitted per seed on val); no selection | `reports/layer0/seed43_layer0_test.json`, `seed44_layer0_test.json` |
| 2 | 2026-10-30 19:18 (= 2026-10-31 00:48 IST) | `candidate_sets_v1` (seed 42) | `686e7d0edd70…` | LOW_CONFIDENCE candidate sets only (O26); second use of test; `gate_v1` unchanged | `reports/layer0/candidate_sets_test_run_seed42.json` |
