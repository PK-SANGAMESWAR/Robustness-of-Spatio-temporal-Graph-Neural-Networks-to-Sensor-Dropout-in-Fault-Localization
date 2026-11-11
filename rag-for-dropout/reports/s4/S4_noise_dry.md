# S4 noise (dry run with mock models: the numbers below test the plumbing and mean nothing); events_status: draft

Primary result: per stratum, CONFIDENT-wrong (`high_conf_error`) first. Checks: share of events that pass (1 = all pass). Judge: GEval rubrics on a 1 to 5 scale, DeepEval metrics 0 to 1. n counts the events measured for that metric (a failed metric is left out, never scored 0).

| metric | high_conf_error (n=4) | confident_correct (n=8) | low_confidence (n=8) |
|---|---|---|---|
| check.schema | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.field_consistency | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.neighbour_counts | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.citation_validity | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.no_confirmation | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.no_control_command | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.no_approval_claim | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.no_injection_echo | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| check.evidence_consistency | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| judge.faithfulness | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| judge.answer_relevancy | 1.00 (n=4) | 1.00 (n=8) | 1.00 (n=8) |
| judge.contextual_relevancy | 0.00 (n=4) | 0.00 (n=8) | 0.00 (n=8) |
| judge.geval_prediction_vs_confirmed | 4.00 (n=4) | 4.00 (n=8) | 4.00 (n=8) |
| judge.geval_completeness | 4.00 (n=4) | 4.00 (n=8) | 4.00 (n=8) |
| judge.geval_adherence | 4.00 (n=4) | 4.00 (n=8) | 4.00 (n=8) |
| judge.geval_evidence_use | 4.00 (n=4) | 4.00 (n=8) | 4.00 (n=8) |
| judge.geval_no_aggregate_metric_as_confidence | 4.00 (n=4) | 4.00 (n=8) | 4.00 (n=8) |
| evidence.final_insufficient | 0.00 (n=4) | 0.00 (n=8) | 0.00 (n=8) |

## Secondary: weighted figure, over the covered strata
Weights: share of the stratum among the test runs at the same k (`reports/events/stratum_shares.json`), renormalised over the three strata; mean over k. Share of runs in no stratum (CONFIDENT, wrong, p < 0.9) per k: k=0 0.031, k=1 0.028, k=2 0.029, k=5 0.029, k=10 0.016.

| metric (over the covered strata) | value |
|---|---|
