# S4 pilot (live, 5 events, 2026-11-11): how to read it

Written for: the author. Snapshot `eval/snapshots/20261111T065933Z_s4_pilot_live.json`; items `eval/items/` (same stem). **Plumbing check on 5 events: these numbers are not results.** No claim about the system. Events are `draft`; retrieval: frozen v2, one query, top 3, SOP excluded.

| k | stratum | top-1 | evidence decision (by) | faithfulness | ctx. relevancy | pred. vs confirmed | completeness | adherence | evidence use | no aggregate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | high_conf_error | wrong | insufficient (generator) | 1.0 | 1.0 | 5.0 | 4.0 | 3.0 | 4.0 | 5.0 |
| 1 | confident_correct | right | insufficient (generator) | 1.0 | 0.85 | 4.0 | 4.0 | 5.0 | 4.0 | 5.0 |
| 2 | low_confidence | right | sufficient (–) | 1.0 | 0.87 | 4.0 | 4.0 | 5.0 | 2.0 | 5.0 |
| 5 | high_conf_error | wrong | insufficient (generator) | 0.91 | 1.0 | 4.0 | 4.0 | 5.0 | 3.0 | 5.0 |
| 10 | high_conf_error | wrong | sufficient (–) | 1.0 | 0.92 | 4.0 | 4.0 | 5.0 | 2.0 | 5.0 |
GEval columns: raw 1 to 5. DeepEval columns: 0 to 1.

## What happened (facts)
- All 5 reports parsed, 0 failures, all deterministic checks pass. 16 judge calls per report (as estimated). Cost 0.10 USD for the judge rerun; about 0.03 USD per report with generation; ledger total 0.28 USD of 5 USD. Full main run: about 1 USD.
- `gpt-6-luna` **refuses an explicit temperature**; the default runs (so the generator is not deterministic: the same prompt gave "sufficient" once and "insufficient" once). `gpt-6.1-terra` accepts temperature 0. Noise runs measure the generator spread.
- Both models **reason before answering and bill the reasoning tokens as output.** First attempt (900 max tokens) returned empty replies. Settings now: generator `reasoning_effort: low`, judge `none` (the cost estimate assumes none). Both are in `configs/eval.yaml`.
- In the last run the generator said "insufficient" for 3 of 5 events: 2 of the 3 CONFIDENT-wrong events and 1 CONFIDENT-correct event (that one said "sufficient" in the first run). tau flagged none of the 5.

## Two defects found and fixed (both mine, both in the harness)
1. The first pilot scored `adherence` = 1 for all 5 reports: the judge read the system fields `candidate_not_confirmed` and `pending_human_review` as inventions (they were not in its context). Fixed: the judge context names them. Adherence is now 3 to 5. The first pilot snapshot (`...T065548Z`) is kept and is invalid for adherence.
2. My check `evidence_consistency` required empty `document_evidence` when the evidence is insufficient: stricter than your rule ("no procedural statement"). It now checks only `recommended_operator_review` and the "insufficient" statement; citing a passage is flagged as informational (`cites_while_insufficient`).
3. A cache bug (a refused temperature changed the cache key between runs) is fixed; the refusal is remembered.

## Read these first
- Event 1 (CONFIDENT, wrong): all 10 neighbours say bus 89 (the wrong prediction); the report states "10 of 10" truthfully and the generator still says the evidence is insufficient. Check whether the report could mislead a reader.
- Event 3 and 5: `evidence_use` 2. Judge reason: the report lists neighbour buses but does not state the disagreement or the neighbour error rate clearly.
- The judge scores are informational until you label about 50 reports (O13).
