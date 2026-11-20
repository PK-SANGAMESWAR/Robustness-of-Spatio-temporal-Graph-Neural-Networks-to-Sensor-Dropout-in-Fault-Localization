# GridSense closure: decisions and what is left (updated 2026-11-20)

Written for the author. On 2026-11-20 you said: "you can take decision and start, pls continue until finish ... keep in mind the money; inform before any api calls". Section A lists the decisions I took on that delegation.
Every one is reversible; the way back is in the last column. Section B lists what only you can do. I did not set anything to `reviewed` and did not write any label.

## A. Decisions taken by the agent (2026-11-20)
| # | Decision | What changed | How to reverse |
|---|---|---|---|
| O46 | **Yes**: Python writes `sensor_coverage_summary` from the event | `decision.assemble(..., view)`; raw generator text kept in `report_raw`; new metric `evidence.generator_coverage_incomplete` (raw failures stay visible: 2 of 75 safety reports, 0 of 60 S4 reports). S4 and safety snapshots rescored offline | drop the `view` argument in the 4 call sites |
| O44 | **(a)**: a Python validator blocks a report with a key, address, path or password pattern, also inside an excerpt | `checks.no_sensitive_info`; `no_sensitive_info` is a BLOCKING check in the workflow: such a report goes to manual review | remove it from `BLOCKING` in `agent/workflow.py` |
| O45 | **Later**: keep the simulated out-of-scope type; stated as a limitation | none (validation report § 5) | write a new prompt version with a question field |
| Tol | **Approved**: the five measured tolerances (field consistency, schema, citation 0.0; faithfulness 0.011; answer relevancy 0.024) | `configs/registry.yaml`: `tolerance_source: measured`; all others stay provisional | set them back to `provisional` |
| Flip | **Limitation**, no 3-sample vote | validation report § 5 | n/a |
| Wording | **Yes**: "the gate is the evidence layer; the RAG report is a traceable explanation, written for a person who still decides" | `reports/validation_report.md` § 3 (I did not edit the paper or the thesis: other tracks) | n/a |
| L2 | Layer 2 run with 4 generator candidates and golden context, deterministic floors only; prices read on the OpenAI pricing page; spend caps fixed in code (0.95 USD worst case) | `eval/layer2.py`, `configs/eval.yaml` prices and candidates; snapshot `s5_layer2_live` | delete the 3 price lines |
| S8 | **Done as far as safe** (you said "use langsmith online"): dataset `gridsense-retrieval` (48 reviewed queries, references only) and one dev-split experiment with local evaluators are online; sensitive sources withheld; one mistaken first experiment deleted and rerun. No tracing, no events, no safety items | `reports/s7/LANGSMITH_STATUS.md`, `configs/eval.yaml` `langsmith.*` | set `upload_approved: false`; delete the dataset in the UI |
| GEval | Run the aggregate-trap GEval rubric (9 reports, cap 0.10 USD) | `safety geval` | n/a |
| Pre-review | Agent recommendations next to your review files, in separate columns and files | `reports/events/judgement_points_s4_agent_recommendation.csv` (25 accept, 8 edit), `reports/s7/SAFETY_REVIEW_TABLE.csv` (columns `agent_recommendation`, `agent_note`) | n/a |

## B. Only you can do these
| # | Task | File | Time |
|---|---|---|---|
| 1 | 33 judgement points: accept, edit or reject. My recommendation is in `judgement_points_s4_agent_recommendation.csv`: accept 25, edit 8 (the 8 `neighbour_disagreement` rows in CONFIDENT-wrong events, with the suggested wording) | `reports/events/judgement_points_s4.csv` | 15 min if you follow it |
| 2 | 25 draft safety items: accept, edit or reject (my recommendation: accept all; notes per type) | `reports/s7/SAFETY_REVIEW_TABLE.csv` | 15 min |
| 3 | O13: label 50 reports, blind (an agent labelling them would make the judge validation circular) | `reports/s4/O13_labelling_sheet.csv`, `O13_LABELLING_GUIDE.md` | 1.5 to 2 h |
| 4 | O40 (162 passages) and O39 (10 near-miss queries) | `reports/layer1/labelling_table.csv`, `NEAR_MISS_CHECK.md` | optional |
| 5 | LangSmith: read `reports/s7/LANGSMITH_STATUS.md`; say whether tracing, online evaluators and the reviewed events may go up, and which sources may leave the machine | `configs/eval.yaml` `langsmith.*` | 10 min |

## C. After your answers
Review (1, 2) makes the events and safety items `reviewed`. Labels (3) give the judge kappa via `python -m gridsense.eval.judge_validation`. Then the S9 report becomes final and the track closes.
