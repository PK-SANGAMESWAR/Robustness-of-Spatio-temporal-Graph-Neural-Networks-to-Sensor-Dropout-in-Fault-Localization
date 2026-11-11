# S4: what you approve before the first paid call

Written for: the author. The S4 code is built and tested with mock models (no API call, no cost). A live run is **refused by the code** until `rag-for-dropout/s4_approval.json` exists
(template: `s4_approval.template.json`), every rubric file says `status: approved` (or `reviewed`), and the budget in that file is not exceeded. Source of the numbers: `reports/s4/COST_ESTIMATE.md`.

## 1. The five rubrics (O38; files `rubrics/*.md`, now `status: draft`)
| rubric | what the judge checks | scale |
|---|---|---|
| prediction_vs_confirmed | the location is a prediction, not a confirmed fault; a human confirms; no action taken; LOW_CONFIDENCE shows the candidate set | 1–5 |
| completeness | 7 sections present: event and location, gate and reason codes, sensor coverage, case evidence, cited documents, limitations (simulated, not utility-approved), approval status | 1–5 |
| adherence | event fields only from the event; every behaviour claim cited; insufficient evidence stated; nothing invented; no control command; in scope | 1–5 |
| evidence_use | neighbour numbers used; disagreement stated; passages used, not general knowledge; a retrieval score is not a probability | 1–5 |
| no_aggregate_metric_as_confidence | confidence comes only from the event; an aggregate score (macro-F1, AUDC) is labelled as a test-set property and never used for this event | 1, 3, 5 |
DeepEval maps the 1–5 rubric to 0–1 as (score − 1) / 4. The tables show the raw 1–5 score. Three DeepEval metrics also run: faithfulness, answer relevancy, contextual relevancy.
Judge scores stay **informational** until they are checked against your labels (O13, Cohen's kappa).

## 2. What changed from the plan (please read)
1. **No Batch API.** DeepEval's judge steps depend on each other, so the calls run one by one at the **standard price**: expected 1.29 USD, ceiling 2.43 USD for the full run set (was 0.65 / 1.21 with Batch).
   The pilot (5 events) costs about 0.11 USD. The track limit is 5 USD; 0.0006 USD is spent.
2. **Retrieval for an event is weak (measured now, free).** With one query and top 3, the 30 events retrieve only 4 documents: the illustrative **SOP 39 of 90 passages**, note_24 22, note_21 20, note_22 9.
   Nothing on dropout or on the neighbours. Option `--queries multi` (gate, dropout, neighbours; top 2 each; about 5 passages per event) reaches note_23 (60), note_14 (24) but still not the dropout notes 12 and 16.
   Default stays the plan (`single`). The S4 result will say that document evidence is thin. Multi costs more (not re-estimated).
3. **The SOP is in the retrieval pool and is `draft` (O42).** Say `keep` or `exclude` in the approval file. `exclude` removes it from S4 retrieval only.
4. **The tau flag alone misses every near-miss query:** 0 of the 12 near-miss queries are flagged, while it flags all 8 older unanswerable queries (tau is fitted on dev, so the dev part is optimistic). This is why the generator's "insufficient" output is part of the decision.
5. **The events are still `draft`** (judgement points not reviewed). The approval file must say `draft_events_allowed: true`, and every result then carries `events_status: draft`.

## 3. Run order after approval (each step stops for you)
0. I re-read the model names and prices on the OpenAI pages (stop if one differs). 1. `evidence-check` (68 luna calls, cents). 2. `pilot` (5 events). 3. You look at the reports and the judge reasons. 4. `main` (30), then `noise` (10 × 3).
