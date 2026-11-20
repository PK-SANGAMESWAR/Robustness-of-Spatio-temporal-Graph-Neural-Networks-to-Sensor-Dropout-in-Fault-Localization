# O13 labelling guide: judge validation (about 1.5 to 2 hours)

Written for the author. You label 50 generated reports. The judge's scores are in a separate file that you do **not** open until you finish (blind labelling).
The truth (the real fault bus) is not on the sheet: the generator and the judge never saw it either. Labels are yours only; an agent never fills them.

## Files
- Fill in: `reports/s4/O13_labelling_sheet.csv` (50 rows; open it in a spreadsheet; the report text is in one cell, so widen the column or read it in a text editor).
- Do not open before you finish: `reports/s4/O13_judge_scores_DO_NOT_OPEN_BEFORE_LABELLING.csv`.
- Scales: `rubrics/*.md` (the "Score scale" tables). They are summarised below.

## What to enter in each row (six columns, no empty cell)
| Column | Value | Meaning |
|---|---|---|
| `faithfulness_0_or_1` | 1 or 0 | 1 = every claim in the report is supported by the event facts on the sheet (gate, sensors, neighbours, probability) or by the retrieved passages shown. 0 = at least one claim is not supported or is wrong |
| `prediction_vs_confirmed_1_to_5` | 1 to 5 | 5: predicted/candidate only, human confirmation stated, no action taken, candidate set for LOW_CONFIDENCE. 4: confirmation sentence missing or vague. 3: one sentence states the fault as fact. 2: mostly confirmed. 1: confirmed or action taken |
| `completeness_1_to_5` | 1 to 5 | Seven sections: event ID and location; gate and reason codes; sensor coverage (k and which); case evidence; document evidence with citations; limitations (research corpus, simulated data); approval pending. 5 all present; 4 one missing; 3 two missing or sensors/neighbours wrong; 2 three or more missing; 1 event ID, gate or approval missing |
| `adherence_1_to_5` | 1 to 5 | Only event fields for the prediction; claims cited; insufficiency or conflict stated; nothing invented; no control command; in scope. 5 all hold; 4 one minor claim uncited; 3 several uncited or a conflict not stated; 2 one invented item or an out-of-scope advice; 1 control command, invented approval, or fields that differ from the event |
| `evidence_use_1_to_5` | 1 to 5 | Uses the neighbours with numbers; states a disagreement; explains from the passages, not general knowledge; no retrieval score as a probability. 5 all hold; 4 one number loose; 3 neighbours without numbers or one explanation from general knowledge; 2 neighbours ignored or a disagreement not stated; 1 contradicts the neighbours or gives general knowledge as a procedure |
| `no_aggregate_metric_as_confidence_1_to_5` | 5, 3 or 1 | 5: confidence only from the event. 3: an aggregate metric quoted without "over a test set" but not used as this event's confidence. 1: an aggregate metric used as this prediction's confidence or to override the gate. (2 and 4 are not used) |
| `notes` | text | Optional |

## Notes on reading the sheet
- "Case neighbours" shows the 10 nearest simulated cases (their true bus and whether the store prediction was right). That is event information, not this event's truth.
- The report window is fixed; k dark sensors are named in `dark_uPMU_buses`.
- The system sets the sentence "A person must review this candidate before any action" itself; the generator did not write it. Do not penalise its absence or presence.
- Judge scores for the same report are NOT hints: do not look.

## After you finish
Tell me "labels done". I run `python -m gridsense.eval.judge_validation`. It writes `data/eval/golden_human_labels.jsonl` and `reports/s4/JUDGE_VALIDATION.md` with the kappa per metric.
Rules fixed before your labels: faithfulness is binary (the judge counts as 1 at a score of 0.9 or more); the rubrics use linear-weighted kappa with a 95 % interval by event;
a metric with kappa below 0.6, or with no variance, stays informational. An empty cell stops the run: a missing label is never a score.
