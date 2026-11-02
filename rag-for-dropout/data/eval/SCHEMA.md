# data/eval/SCHEMA.md — fields of the GridSense golden sets

Written 2026-10-28 (S0). Source: `rag-eval.md` §5 (`spec/eval_03.md`), with the author decisions of 2026-10-28.
The `.jsonl` files do not exist yet. One JSON object per line. UTF-8. Each file starts its life at version 0.

## Common rules
1. `status` is `draft` or `reviewed`. An agent writes only `draft`. Only the author sets `reviewed` (`rag-golden-review`).
2. A reported result uses `reviewed` items only. The snapshot records the set version and the item count.
3. A change to a `reviewed` item makes a new set version. Record it in `EXPERIMENT_LOG.md`.
4. A golden context is identified by `document_id` + `section_path` + `span` + `span_hash`, never by a chunk ID.
   `span_hash` = sha256 of the normalized span (lower case, collapsed whitespace, stripped).
5. Event items come only from the `test` split. Their windows are never in the case store (assert in code).
6. Each item has `created` (date), `created_by` (`agent:<model>` or `author`), and `reviewed` (date or null).

## golden_retrieval.jsonl (40–60 items, about 15 % unanswerable)
| Field | Type | Meaning |
|---|---|---|
| `query_id` | str | `R-0001` … |
| `query` | str | Operator-style question |
| `query_type` | str | `paraphrase`, `exact_identifier`, `dropout_observability`, `unanswerable` |
| `golden_contexts` | list | `{document_id, section_path, span, span_hash, grade}`; `grade` 1–3 for nDCG |
| `reference_answer` | str | Short answer from the golden contexts; empty for unanswerable |
| `answerable` | bool | False → the sufficiency check must return "insufficient" |
| `critical_terms` | list[str] | Identifiers that must survive normalization (for example `k = 10`, `RGSAGE-Max`) |
| `split` | str | `dev` (tuning) or `test` (final) |
| `status`, `created`, `created_by`, `reviewed` | | Common rules |

Optional fields of an unanswerable item (added 2026-11-02, near-miss queries): `near_miss_of` (documents that discuss the topic) and `absence_check` (`patterns_not_found_in_the_corpus`, `note`); a test re-runs the patterns.

## golden_events.jsonl (60–100 items)
| Field | Type | Meaning |
|---|---|---|
| `event_id` | str | Window-ID prefix `<bus>_<type>_<R>_<run>`, for example `7_ABCG_10p0_0014` |
| `window_id` | str | Window at the report index of `configs/eval.yaml` `splits.report_window.chosen_index` (O3) |
| `true_label`, `true_bus` | int | Label 1–25 and `FAULT_BUSES[label-1]` |
| `fault_type`, `resistance_ohm`, `load_multiplier` | | From `metadata.csv` |
| `k`, `mask_seed`, `dropped_pmu_nodes` | | Dropout setting of the item (k ∈ {0,1,2,5,10}) |
| `stratum` | str | `confident_correct`, `high_conf_error`, `low_confidence`, `reject` (filled after S2 by the seeded sampler) |
| `golden_passages` | list | As `golden_contexts` above |
| `reference_report_points` | list[str] | Facts that a correct report must state |
| `split` | str | Always `test` |
| `status`, `created`, `created_by`, `reviewed` | | Common rules |

## golden_safety.jsonl (20–30 items)
| Field | Type | Meaning |
|---|---|---|
| `safety_id` | str | `S-0001` … |
| `test_type` | str | `counterfactual_swap`, `evidence_flip`, `empty_evidence`, `prompt_injection`, `control_command_trap`, `approval_string`, `out_of_scope`, `sensitive_info`, `aggregate_metric_as_confidence` |
| `base_event_id` | str or null | Event that the test modifies |
| `perturbation` | object | What changes (new bus, new neighbours, injected chunk text, question text) |
| `expected_behaviour` | str | What a passing report does |
| `check` | str | `deterministic` (with the rule name) or `geval` (with the rubric name) |
| `status`, `created`, `created_by`, `reviewed` | | Common rules |
At least one item has `test_type: aggregate_metric_as_confidence` (author decision 2026-10-28).

## golden_human_labels.jsonl (about 50 items, O13)
| Field | Type | Meaning |
|---|---|---|
| `label_id` | str | `H-0001` … |
| `report_ref` | str | Snapshot + item ID of the labelled report |
| `faithfulness` | int | 0/1 per claim summary, or the rubric scale |
| `rubric_scores` | object | `{rubric_name: score}` |
| `labeller` | str | Always `author` |
| `labelled` | date | Date |
