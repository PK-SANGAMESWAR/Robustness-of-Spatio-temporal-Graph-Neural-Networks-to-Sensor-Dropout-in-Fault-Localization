# GridSense closure: your decisions in one place (2026-11-20)

Written for the author. Each row has my recommended default. Answer "accept all recommendations" or list the rows you change, and I apply them (code, tests, log, commit). Rows marked [you] cannot be done by an agent.

## A. Decisions I can apply on your answer
| # | Question | Recommended default | Effect |
|---|---|---|---|
| O46 | Python writes `sensor_coverage_summary` from the event (as for the human-review field) | **Yes** | Removes the only real generator failure of the live safety run (dark buses not named at k = 10, 2 of 75). Changes the report schema; S4 and safety numbers are rescored offline |
| O44 | Sensitive strings in verbatim excerpts | **(a)** a Python output validator blocks a report that holds a key, address, path or password pattern | Fail-closed; the live model passed 6 of 6, the mock failed: the check is needed for a poisoned passage |
| O45 | Operator-question field for the out-of-scope test | **Later**: keep the simulated type, state it as a limitation | A new prompt version costs a re-run; not needed for the validation story |
| Tol | Measured tolerances (faithfulness 0.0101, answer relevancy 0.0238, three gate checks 0) | **Approve** and set `tolerance_source: measured` in the registry | `compare` can then return PASS instead of REVIEW |
| Flip | Generator evidence-status flipping (5 of 10 events change between repeats) | **Record as a limitation**; no 3-sample vote | A vote costs 3 times the generator cost and does not fix the cause (the status follows the passages) |
| Wording | Use "the gate is the evidence layer; the RAG report is a traceable explanation" in the paper and thesis | **Yes** | Matches Layer 4 and the safety results |

## B. Your reviews [you]
| # | Task | File | Time |
|---|---|---|---|
| 1 | 33 S4 judgement points: accept, edit or reject (8 `neighbour_disagreement` rows: replace the wording as suggested in the validation report) | `reports/events/judgement_points_s4.csv` | 30 min |
| 2 | 25 draft safety items with the live result next to each | `reports/s7/SAFETY_REVIEW_TABLE.csv` | 20 min |
| 3 | O13: label 50 reports, blind | `reports/s4/O13_labelling_sheet.csv` and `O13_LABELLING_GUIDE.md` | 1.5 to 2 h |
| 4 | O40 (162 passages) and O39 (10 near-miss queries) | `reports/layer1/labelling_table.csv`, `NEAR_MISS_CHECK.md` | 1 to 2 h; optional |

## C. Needs your approval or information
| # | Item | Why |
|---|---|---|
| S8 | LangSmith: first upload of the reviewed golden sets and the traces (O8) | Rule: ask before the first upload. Only reviewed sets go up |
| L2 | Layer 2 generator comparison | Only `gpt-6-luna` has an approved price. Name 1 or 2 more candidate models and prices (about 0.05 USD each for 30 events x 3 repeats) |

## D. What happens after your answers
1. A + B1 + B2: golden events and safety items become `reviewed` (by you); Layer 5 and the S4 tables become reportable.
2. B3: `judge_validation` runs; judge metrics are `usable` or stay `informational` by the pre-registered bar.
3. S8 (if approved) and L2 (if you name models): then the S9 validation report is final.
4. The RAG track then closes: `DONE.md` § 3 and the memory file are updated, and the next part of the project can start.
