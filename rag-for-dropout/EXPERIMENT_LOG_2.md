# EXPERIMENT_LOG_2.md — entries from E16 (continues EXPERIMENT_LOG.md)

One entry per stage and per run. Append only.

## E16 — 2026-11-02 — O30 review applied: golden set v1, notes, feeder documents accepted
- The author's review was applied. Record: `reports/layer1/O30_REVIEW_RECORD.md`. Scripts: `data/eval/apply_o30_review.py` and `corpus/notes_v1/_review_2026-11-02.py` (both assert the facts before they write).
- Golden retrieval v0 (54 drafts) → **v1: 56 reviewed items**: R-0005 edited (results_md row removed), R-0034 and R-0044 edited (reference answers), R-0055 and R-0056 added, R-0036, R-0047, R-0048 accepted unchanged. Dev 25 answerable + 4 unanswerable; test 23 + 4.
- Notes: 19 reviewed. The 11 notes that the review did not mention stay draft, with the SOP (O34). Found while checking: note_16 had a wrong "indistinguishable" pair, note_20 overstated "the margin is the only threshold" (k = 2 has a probability threshold 0.5424; the margin thresholds are not monotone), note_09 had one unsupported sentence. All corrected. note_07: 25 μPMUs confirmed against the dataset config. 0.846 appears only as the standard model (a test checks every note).
- Feeder documents accepted, not edited (manifest hashes intact). The nearest μPMU to bus 51 is bus 47, 0.229 km, 3 hops; independent check from the raw DSS lines (3 × 0.25 kft = 0.229 km).

## E17 — 2026-11-02 — S3 sweeps (Layer 1, deterministic) and the frozen test runs
- Code: `eval/sweep.py` (axes, selection, frozen runs), `eval/store_bench.py`, `layer1.evaluate()` with reachability (golden spans cut by chunk boundaries), context tokens and memory, a chunk meter, Chroma and Qdrant variants. 184 tests pass.
- Author instructions during the run. (1) "Look for best + cheap, not just cheap": selection rule v1 (the cheapest candidate inside the 95 % interval) was replaced by rule v2 (quality first; equal = within one query on one metric = 0.013; then the cheapest) after the dev results of three axes had been seen. Rule v1 results: `reports/sweeps/rule_v1/`. All axes were rerun. (2) "Use a good vector store, not FAISS (Chroma and others), keep the precision good": FAISS only as the exact reference; end-to-end test with the real hybrid retriever.
- Winners on dev: MiniLM; recursive, 200 tokens, 10 % overlap; hybrid RRF (k 60, 20 candidates); no reranker; top_k 3; Qdrant local. Chroma (default and high-recall HNSW) gives identical top-10 lists and metrics for all 29 dev queries and for all test queries.
- Frozen, one run each on the golden test split (`reports/layer1/GOLDEN_TEST_USE.md`): final: Recall@1 0.74, Recall@5 0.91, MRR@10 0.79, P@3 0.29, P@5 0.21 (23 answerable). final with Chroma: identical. final dense-only: Recall@1 0.48, MRR@10 0.65 (hybrid gain +0.145 MRR@10, 95 % interval +0.045 to +0.255). No-answer (tau 0.490 from dev): 4 of 4 unanswerable flagged, 5 of 23 answerable flagged.
- Low Recall@1: 0 of 31 dev and 0 of 28 test golden contexts are split by the final chunking; the causes were read by hand (sibling note, alternative evidence, table row, vocabulary gap): `reports/sweeps/SUMMARY.md` § 4.
- OpenAI `text-embedding-3-small` as one candidate: 31,287 tokens, 0.000626 USD (estimate 0.000626, cap 0.002); ledger total 0.00063 of 5 USD. The text of all 37 corpus documents, including the local revision manuscript, was sent. On this corpus, with settings tuned for MiniLM, the other embedders did not perform better. (Reworded on 2026-11-02, see E18.)
- Snapshots: the first chunker run crashed in the report writer after it had written its snapshots, and rule v1 left more; they are kept (append-only). Not committed.

## E18 — 2026-11-02 — O34 applied; embedder check (no API call); wording of the embedder result
- O34: the 11 notes accepted (status reviewed; `corpus/notes_v1/_review_2026-11-02_o34.py`); the SOP stays draft. R-0055 and R-0056 are `draft` again (the author checks the spans that the agent chose); golden set v2 = 54 reviewed + 2 draft. From now on every sweep and run loads reviewed items only (dev 24 answerable + 4 unanswerable; test 22 + 4). The earlier dev and test numbers included R-0055 and R-0056; they are recomputed without them in E19.
- Guards added: no OpenAI call and no paid embedder on this corpus again (code refuses); the one-run-on-test guard keys on the configuration name (the corpus status change had altered the index fingerprint).
- `reports/sweeps/embedder_check.md` (`gridsense.eval.embedder_check`, no API call): self-retrieval 1.000 for all four embedders (198 chunks each; no duplicate texts); all vectors normalized; the four indices hold the same chunks in the same order; prefixes equal the model cards and were used (stored vectors equal the recomputed ones to 1e-7); OpenAI vectors equal the on-disk cache. Paired bootstrap on dev (24 queries), MiniLM minus: bge-small Q +0.060 (−0.017 to +0.136), e5-small +0.044 (−0.055 to +0.163), OpenAI +0.082 (−0.005 to +0.190).
- Wording: "On this corpus, with settings tuned for MiniLM, the other embedders did not perform better." The chunk settings were selected with MiniLM before the embedder sweep. Changed in SUMMARY.md, embedder.md, WINNERS.json and in the E17 sentence above (E17 is reworded in place; this entry is the record).

## E19 — 2026-11-02 — Steps 3 to 5: reviewed-only check, precision, table rows (v2), labelling table
- Reviewed-only (`reports/sweeps/REVIEWED_ONLY.md`, from stored per-query results; no new run): without the draft R-0056 no sweep winner changes. Frozen v1 on test without the draft R-0055 (22 answerable): Recall@1 0.77, Recall@5 0.95, MRR@10 0.83; tau 0.490; 4 of 4 unanswerable flagged, 5 of 22 answerable flagged.
- Precision (`PRECISION.md`, `PRECISION_final_v2.md`; relevant = holds a golden span; M = relevant chunks in the index): v1 test P@1 0.77, normalized P@3 0.75, R-precision 0.66; v2 test P@1 0.73, normalized P@3 0.73, R-precision 0.64. Computed from stored lists: no new use of the test split.
- Table rows (`table_rows.md`): rule fixed before the run (v2 frozen if Q(v2) >= Q(v1) - 0.014). Dev: Q 0.875 → 0.899, Recall@1 0.79 → 0.88, MRR@10 0.85 → 0.90; 198 → 290 chunks; frozen as v2. Test, one run (`FROZEN_TEST_v2.md`, ledger row `final_v2`): Recall@1 0.73, Recall@5 0.91, MRR@10 0.80; against v1 Q −0.036 (−0.106 to +0.014). R-0036 fell from rank 1 to 5, R-0034 rose from 4 to 2. The dev gain did not repeat on test; nothing was changed; the choice between v1 and v2 is the author's (O41).
- Labelling table for the final configuration v2: `reports/layer1/labelling_table.csv` (162 rows), `LABELLING.md`, `gridsense.eval.graded_precision` (tested). Waits for the author's labels (O40).
- Guards: no API call and no paid embedder again; the one-test-run ledger keys on the configuration name.

## E20 — 2026-11-02 — Steps 6 and 7: near-miss queries and the S4 preparation (nothing started)
- 12 near-miss unanswerable queries added as drafts (R-0057 to R-0062 dev, R-0063 to R-0068 test): topic in the corpus, answer not; the script and a test re-check the absence patterns. Golden set v2 = 54 reviewed + 14 draft (O39).
- S4 prepared, not started: `S4_PLAN.md`; draft prompts `prompts/generator_v0.md` and `prompts/evidence_check_v0.md`; models and prices in `configs/eval.yaml` (generator gpt-6-luna 0.10 / 0.50, judge gpt-6.1-terra 1.00 / 5.00, Batch rates 50 %); names and prices read on the OpenAI pricing page and the luna model page on 2026-11-02 (no API call); the ledger prices Batch calls separately.
- Cost estimate (`reports/s4/COST_ESTIMATE.md`; tokens counted locally from the real prompts and DeepEval templates; fixtures for events): 1,088 calls; expected 1.29 USD at standard rates and 0.65 USD with Batch; ceiling 2.43 and 1.21 USD; limit 5 USD; spent 0.0006.
- Evidence decision for S4: tau is one input; the generator's "insufficient evidence" output is another; fail-closed OR; deterministic tests specified in `S4_PLAN.md` § 4. Manuscript chunks are 24 % of the passages in the top 3 (O37). No API call was made. Not committed yet at the time of this entry.

## E21 — 2026-11-02 — Decisions O35, O37, O39, O41 applied; golden events drafted (O36); S4 not started
- O41: v2 kept. The test difference (Q −0.036, interval −0.106 to +0.014) and the R-0036 cause are written as a known limitation in `reports/sweeps/SUMMARY.md` (sections 2 and 7); a diversity limit for row chunks is future work, to be selected on dev only. v1 or v2 is not chosen with test results.
- O37: manuscript chunks allowed in S4 prompts, corpus unchanged. O35: generator temperature 0 (`configs/eval.yaml`); if gpt-6-luna refuses a temperature, none is sent and the model default is recorded.
- O39: R-0055 and R-0056 → reviewed (golden v3, 56 reviewed + 12 near-miss drafts). Corpus check of three near-miss queries in `reports/layer1/NEAR_MISS_CHECK.md`: R-0064 and R-0067 are clean near-misses; R-0059 has only totals in the corpus (an estimate is possible, so I flag it for you). R-0060 rephrased to "Which five μPMUs does the targeted study remove when the fault is at bus 57?". No run was repeated: all sweep and test numbers still use the 54 reviewed queries of v2.
- O36: 60 draft events (`data/eval/golden_events.jsonl`, `reports/events/`): test split, w = 25, 12 per k (4 CONFIDENT right, 4 CONFIDENT wrong with p >= 0.9, 4 LOW_CONFIDENCE), 11 of 11 fault types, 25 of 25 fault buses. Built from stored Layer-0 votes and latents (seed 36, deterministic); masks replayed with the paper's function and the stored votes reproduced (largest difference 0.0). Reference points from the event object are `reviewed`, judgement points and passages `draft`. Tests: `tests/gridsense/test_events.py`.
- Not started: S4. It waits for O38 (rubrics), O42 (SOP) and the cost approval. No API call, no paid embedder.

## E22 — 2026-11-02 — Author decisions on the draft events; S4 event selection (recorded before any S4 run)
- R-0059 kept as a near-miss trap (expected output: "insufficient evidence"; field `expected_output`); R-0060 new wording accepted. Both set `reviewed`: golden v4 = 58 reviewed + 10 near-miss drafts. No run repeated.
- High probability = calibrated p >= 0.9 (accepted). The 180 copies of the 3 fixed judgement statements stay `draft` until the author approves each statement (`reports/events/fixed_statements.md`). The other 67 judgement points: `reports/events/judgement_points_review.csv`.
- **S4 selection (30 events).** Rule: 2 events per cell, seeded inside each cell, no hand selection. The author's "k x gate outcome" gives only 10 cells (no REJECT event exists in the pool), so the cell is k x stratum (confident_correct, high_conf_error, low_confidence) = 15 cells x 2 = 30. Cell seed = 360000 + 10 k + stratum index (0, 1, 2); `numpy.random.default_rng(seed).choice` on the sorted event IDs of the cell. File: `reports/events/s4_selection.json` (code `select_s4`; test `test_O36_s4_selection_...`). Result by cell:
  - k=0 confident_correct seed 360000: 55_CA_10p0_0034, 99_BC_0p1_0021
  - k=0 high_conf_error seed 360001: 53_AB_10p0_0009, 93_ABG_10p0_0008
  - k=0 low_confidence seed 360002: 29_BC_10p0_0034, 47_ABG_10p0_0011
  - k=1 confident_correct seed 360010: 108_ABG_1p0_0044, 86_BCG_10p0_0008
  - k=1 high_conf_error seed 360011: 18_BG_1p0_0017, 93_CAG_10p0_0040
  - k=1 low_confidence seed 360012: 101_BC_0p1_0025, 62_BC_10p0_0009
  - k=2 confident_correct seed 360020: 13_CA_0p1_0023, 62_AG_1p0_0046
  - k=2 high_conf_error seed 360021: 42_CA_1p0_0046, 93_ABG_1p0_0001
  - k=2 low_confidence seed 360022: 65_CAG_10p0_0044, 99_CG_10p0_0040
  - k=5 confident_correct seed 360050: 7_AB_10p0_0048, 97_CAG_0p1_0042
  - k=5 high_conf_error seed 360051: 47_ABC_1p0_0040, 57_BG_0p1_0035
  - k=5 low_confidence seed 360052: 21_CAG_10p0_0022, 80_ABCG_10p0_0011
  - k=10 confident_correct seed 360100: 13_ABCG_10p0_0039, 35_CG_0p1_0041
  - k=10 high_conf_error seed 360101: 55_BG_1p0_0028, 72_BC_0p1_0030
  - k=10 low_confidence seed 360102: 35_AB_10p0_0035, 99_CAG_10p0_0021
- The selection covers 11 of 11 fault types and 21 of 25 fault buses; 13 of 30 top-1 right; 33 of the 67 other judgement points belong to these 30 events. The events are still `draft`; S4 needs their judgement points reviewed.
- S4 not started. No API call.

## E23 — 2026-11-03 — Fixed statements approved; reporting rule; S4 judgement sheet (S4 not started)
- The 3 fixed judgement statements were approved. Statement 3 was changed to "The report does not claim that the result shows safety or performance on a real grid." The 180 copies are `reviewed` (`derived_from: author_approved_fixed_statement`). `golden_events.jsonl` was regenerated (the code is deterministic; only these fields changed).
- The S4 selection (k x stratum, 2 per cell, seed 360000 + 10 k + stratum index) was approved. The record in E22 and `s4_selection.json` is unchanged.
- Reporting rule added to `EVAL.md` § 11: S4 results per stratum or weighted by the stratum share in the test split (`reports/events/stratum_shares.json`); never an unweighted mean over the 30 events as a system score.
- Separate sheet for the 33 judgement points of the 30 S4 events: `reports/events/judgement_points_s4.csv` (with predicted bus, dark buses, nearest μPMUs, kNN agreement). The other 34 of the 67 stay in `judgement_points_review.csv`.
- Check of the 20 "nearest μPMU dark" points: the nearest-μPMU lists come from `feeder_fault_buses`, which uses the operated feeder (sw7 and sw8 open). Recomputed from the OpenDSS model (`feeder_distances`, open switches sw7 and sw8, sort by km then bus) for the 59 events that have a predicted bus: 0 differences. The as-built paper graph (switches closed) is not used for these points.
- S4 not started. No API call.

## E24 — 2026-11-03 — Reporting order for S4; commit rule (S4 not started)
- `EVAL.md` § 11 and `S4_PLAN.md`: the primary S4 result is the per-stratum table, with the CONFIDENT-wrong stratum (`high_conf_error`) first. The weighted figure is secondary and always labelled "over the covered strata". No unweighted mean over the 30 events.
- Commit rule (author): commit all GridSense files to `gridsense-eval` at the end of each round.
- No API call. S4 not started.

## E25 — 2026-11-11 — S4 code built and tested with mock models (no API call, no cost)
- Code `src/gridsense/s4/`; 36 tests (`test_s4.py`; all gridsense tests pass). The live client is refused in code without `s4_approval.json` from the author and `reviewed` rubrics; a fake OpenAI object tests pricing, cache, salt, temperature fallback (O35) and the budget.
- The generator never sees the truth, the run ID or the window ID (anonymous `event-NNN`; tested). Deterministic checks: schema, field consistency, neighbour counts, citation validity, no confirmation, no control command, no approval claim, no injection echo, evidence consistency.
- Dry runs (mock models): pilot (5), main (30), noise (10 x 2), evidence check (68 queries). Snapshots `*_s4_*_dry.json`; the dry numbers test plumbing only.
- Findings (free): (1) no Batch API: standard price, 1.29 USD expected, 2.43 ceiling. (2) One query + top 3 retrieves only 4 documents for the 30 events; the draft SOP is 39 of 90 passages; no dropout or neighbour note. `--queries multi` is an option. (3) tau alone flags 0 of the 12 near-miss queries (all 8 older unanswerable ones are flagged).
- DeepEval maps a 1-5 rubric to 0-1 as (score - 1) / 4: checked with mock scores 2 and 5.
- Not done: the price re-check on the OpenAI pages; the approval file. S4 not started.

## E26 — 2026-11-11 — Approvals recorded; first live S4 calls: evidence check (68) and pilot (5). Stopped before the main run
- Author approvals (chat, 2026-11-11): rubrics approved (status `approved`; O38), SOP **excluded** from S4 retrieval (O42), budget 2.43 USD, draft events allowed, single-query retrieval. `s4_approval.json` written by the agent from these messages. Prices re-read on developers.openai.com/api/docs/pricing: unchanged (luna 0.10/0.50, terra 1.00/5.00; Batch half).
- SOP exclusion: query-time filter only (index and fingerprint unchanged); the 30 events then retrieve 7 documents (note_21 24, note_16 24, note_24 22, note_22 10, note_25 4, note_29 4, note_09 2). tau (dev, 0.4639) is unchanged; the dense top-1 is now over the remaining documents.
- Models: luna refuses an explicit temperature (default used, recorded; O35 fallback). Both models reason and bill reasoning as output: 900 max tokens gave empty replies in the first pilot; set generator effort `low`, judge `none`, caps 4000 and 1500. Direct diagnostic calls (0.01 USD) are in the ledger as `s4_diagnostic`.
- Evidence check, live, final settings (snapshot `...T065018Z_s4_evidence_check_live`): unanswerable queries flagged insufficient 6 of 6 dev, 4 of 4 test (reviewed), all 10 near-miss (draft); tau alone flags 4 of 6, 4 of 4 and 0 of 10. Answerable queries wrongly flagged: dev 0.20 (tau 0.20), test 0.30 (tau 0.22). Dev and test answerable rates are on queries that tau was fitted on (dev).
- Pilot (5 events, one per k, seeded): 0 failures; see `reports/s4/PILOT_READING.md`. Defects fixed: judge context (adherence artefact), a too-strict check, a cache-key bug. First pilot snapshots kept (append-only; invalid for adherence).
- Ledger total 0.28 USD of 5. Main run and noise runs not started: stopped for the author to read the pilot.

## E27 — 2026-11-11 — S4 main run (30 events, live). Author said "approve" after the pilot
- Snapshot `eval/snapshots/20261111T091213Z_s4_main_live.json`, items `eval/items/20261111T091213Z_s4_main_live.jsonl`, table `reports/s4/S4_main_live.md`. 30 items, 0 failures, 0.732 USD (330 calls). Ledger total 1.013 USD of 5. Events are `draft` (author-approved for S4).
- Checker defect (mine, found after the run): `field_consistency` flagged "Dark uPMU buses: []" at k = 0 as a dark-sensor claim (5 reports; all 5 had this as their only error). Fixed in `checks.py` (empty list "[]" is not a claim) with a regression test. The table file still shows the old values (0.90 / 0.80 / 0.80); with the fix all 30 pass field_consistency. The fix was made after seeing the results; no re-call was needed.
- Real finding: `evidence_consistency` fails for 2 of 10 CONFIDENT-wrong events (53_AB_10p0_0009, 57_BG_0p1_0035): the generator says "insufficient" and still fills `recommended_operator_review`.
- Generator "insufficient" share: CONFIDENT-wrong 0.8, CONFIDENT-correct 0.6, LOW_CONFIDENCE 0.2. It follows the retrieved passages, not the gate state; it is not a measure of prediction risk.
- In CONFIDENT-wrong events the neighbours often agree with the wrong bus (10 of 10 in 5 of 10 events). The reports state the count correctly.
- Judge scores stay informational (O13). Noise runs not started.
