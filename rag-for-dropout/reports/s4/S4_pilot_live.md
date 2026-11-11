# S4 pilot (live); events_status: draft

Primary result: per stratum, CONFIDENT-wrong (`high_conf_error`) first. Checks: share of events that pass (1 = all pass). Judge: GEval rubrics on a 1 to 5 scale, DeepEval metrics 0 to 1. n counts the events measured for that metric (a failed metric is left out, never scored 0).

| metric | high_conf_error (n=3) | confident_correct (n=1) | low_confidence (n=1) |
|---|---|---|---|
| check.schema | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.field_consistency | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.neighbour_counts | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.citation_validity | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.no_confirmation | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.no_control_command | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.no_approval_claim | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.no_injection_echo | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| check.evidence_consistency | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| judge.faithfulness | 0.97 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| judge.answer_relevancy | 1.00 (n=3) | 1.00 (n=1) | 1.00 (n=1) |
| judge.contextual_relevancy | 0.97 (n=3) | 0.85 (n=1) | 0.87 (n=1) |
| judge.geval_prediction_vs_confirmed | 4.33 (n=3) | 4.00 (n=1) | 4.00 (n=1) |
| judge.geval_completeness | 4.00 (n=3) | 4.00 (n=1) | 4.00 (n=1) |
| judge.geval_adherence | 4.33 (n=3) | 5.00 (n=1) | 5.00 (n=1) |
| judge.geval_evidence_use | 3.00 (n=3) | 4.00 (n=1) | 2.00 (n=1) |
| judge.geval_no_aggregate_metric_as_confidence | 5.00 (n=3) | 5.00 (n=1) | 5.00 (n=1) |
| evidence.final_insufficient | 0.67 (n=3) | 1.00 (n=1) | 0.00 (n=1) |

## Secondary: weighted figure, over the covered strata
Weights: share of the stratum among the test runs at the same k (`reports/events/stratum_shares.json`), renormalised over the three strata; mean over k. Share of runs in no stratum (CONFIDENT, wrong, p < 0.9) per k: k=0 0.031, k=1 0.028, k=2 0.029, k=5 0.029, k=10 0.016.

| metric (over the covered strata) | value |
|---|---|
