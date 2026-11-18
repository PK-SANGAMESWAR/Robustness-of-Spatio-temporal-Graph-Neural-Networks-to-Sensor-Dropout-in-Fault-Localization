# Comparison: FAIL

Baseline: `20261111T091213Z_s4_main_live.json`
Candidate: `20261118T054551Z_s4_main_live_rescored.json`

- BASELINE NOT HEALTHY: layer2.field_consistency = 0.8 misses the floor 1.0
- gate not measured in either snapshot: layer5.injection_pass
- gate not measured in either snapshot: layer5.no_control_command
- gate not measured in either snapshot: layer5.no_aggregate_metric_as_confidence
- gate not measured in either snapshot: layer0.no_leakage_assertion
- gate not measured in either snapshot: layer0.selective_risk_confident
- gate not measured in either snapshot: layer0.error_recall

| metric | role | baseline | candidate | delta | result | rule |
|---|---|---|---|---|---|---|
| layer2.answer_relevancy | guardrail | 0.9948 | 0.9948 | 0 | REVIEW | tolerance is provisional |
| layer2.citation_validity | gate | 1 | 1 | 0 | ok | met |
| layer2.faithfulness | guardrail | 0.983 | 0.983 | 0 | REVIEW | tolerance is provisional |
| layer2.field_consistency | gate | 0.8 | 1 | 0.2 | BASELINE NOT HEALTHY | baseline misses the floor |
| layer2.schema_validity | gate | 1 | 1 | 0 | ok | met |

A PASS does not make the candidate the new baseline: only the author accepts a baseline.
