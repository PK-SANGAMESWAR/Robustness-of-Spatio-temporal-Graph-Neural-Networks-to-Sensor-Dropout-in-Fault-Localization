# S4 noise (live, rescored offline with E + A; judge scores are of the raw reports)

Primary result: per stratum, CONFIDENT-wrong (`high_conf_error`) first. Checks: share of events that pass (1 = all pass). Judge: GEval rubrics on a 1 to 5 scale, DeepEval metrics 0 to 1. n counts the events measured for that metric (a failed metric is left out, never scored 0).

| metric | high_conf_error (n=6) | confident_correct (n=12) | low_confidence (n=12) |
|---|---|---|---|
| check.schema | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.field_consistency | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.neighbour_counts | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.citation_validity | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.no_confirmation | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.no_control_command | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.no_approval_claim | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.no_injection_echo | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| check.evidence_consistency | 1.00 (n=6) | 1.00 (n=12) | 1.00 (n=12) |
| judge.faithfulness | 0.97 (n=6) | 0.99 (n=12) | 1.00 (n=12) |
| judge.answer_relevancy | 0.98 (n=6) | 0.97 (n=12) | 1.00 (n=12) |
| judge.contextual_relevancy | 0.99 (n=6) | 0.90 (n=12) | 0.94 (n=12) |
| judge.geval_prediction_vs_confirmed | 4.17 (n=6) | 4.08 (n=12) | 4.00 (n=12) |
| judge.geval_completeness | 4.00 (n=6) | 4.00 (n=12) | 4.00 (n=12) |
| judge.geval_adherence | 4.33 (n=6) | 4.08 (n=12) | 4.00 (n=12) |
| judge.geval_evidence_use | 3.83 (n=6) | 4.17 (n=12) | 3.58 (n=12) |
| judge.geval_no_aggregate_metric_as_confidence | 5.00 (n=6) | 4.67 (n=12) | 4.83 (n=12) |
| evidence.final_insufficient | 0.83 (n=6) | 0.75 (n=12) | 0.25 (n=12) |
| evidence.generator_review_while_insufficient | 0.50 (n=6) | 0.00 (n=12) | 0.08 (n=12) |
| evidence.generator_coverage_incomplete | 0.00 (n=6) | 0.00 (n=12) | 0.00 (n=12) |

## Secondary: weighted figure, over the covered strata
Weights: share of the stratum among the test runs at the same k (`reports/events/stratum_shares.json`), renormalised over the three strata; mean over k. Share of runs in no stratum (CONFIDENT, wrong, p < 0.9) per k: k=0 0.031, k=1 0.028, k=2 0.029, k=5 0.029, k=10 0.016.

| metric (over the covered strata) | value |
|---|---|
