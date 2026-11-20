# GridSense validation report (S9, updated 2026-11-20)

Written for: the author (Paper 1 / thesis RAG chapter authors and their supervisors may read it). Status: **draft report**. The golden events (60), the golden passages used in Layer 2 and the safety items (25) are `status: draft`;
judge scores are informational (no human labels yet, O13). Every number is copied from a snapshot or report named in the "Source" column. The data is simulated: **no result shows safety on a real grid.**

## 1. What was built and frozen
| Part | Frozen version | Source |
|---|---|---|
| Model | dropout-aware RGSAGE-Max, seed 42 (`checkpoint_best.pt`, sha256 `0172a60a…`) | `claude-rag.md` § 9 |
| Gate | `gate_v1` (sha256 `693ff351157d…`): CONFIDENT / LOW_CONFIDENCE / REJECT, per-k thresholds, conformal 90 % sets; one run on `test` | `reports/layer0/FINAL_REPORT.md` |
| Case retrieval | held-out latent kNN store, exact FAISS, 1.155 M vectors, 10 neighbours | `reports/s2/S2_REPORT.md` |
| Document retrieval | configuration v2: recursive 200 tokens, table-row chunks, hybrid dense + BM25, top 3, Qdrant local, MiniLM | `reports/sweeps/FROZEN_TEST_v2.md` |
| Report | generator `gpt-6-luna` (kept by Layer 2, § 2), judge `gpt-6.1-terra`; Python owns the gate, the evidence decision, the human-review field, the sensor-coverage sentence and the approval (E + A + O46) | E29, E38 |
| Workflow | LangGraph, SQLite checkpoints, explicit human decision, blocking sensitive-content check; prototype (no authentication, no UI) | E31, E33, E38 |
| Mirror | LangSmith project `gridsense-eval` (30 replay traces), dataset `gridsense-retrieval` (48 reviewed queries, references only) | `reports/s7/LANGSMITH_STATUS.md` |

## 2. Results by layer
**Layer 0, the gate (test, seed 42, one run).** CONFIDENT accuracy 0.949 / 0.949 / 0.950 / 0.949 / 0.949 for k = 0 / 1 / 2 / 5 / 10. Coverage 0.845 / 0.825 / 0.801 / 0.731 / 0.513 (k = 10: CI 0.504 to 0.521).
Error recall 0.654 / 0.691 / 0.734 / 0.812 / 0.914. Model accuracy without the gate 0.876 / 0.863 / 0.850 / 0.802 / 0.694. (`test_run_seed42.json`, `ABLATION_LADDER.md`.)
**The thresholds belong to one checkpoint:** with the seed-42 thresholds, seeds 43 and 44 give CONFIDENT accuracy 0.897 to 0.932 (target 0.95). Each new checkpoint needs its own fit (`FINAL_REPORT.md`).
At the 95 % point the gate is effectively margin-only: kNN agreement is switched off (τ_k = 0) at every k.

**Ablation ladder (val cal-B, in-sample thresholds; `ABLATION_LADDER.md`).** A0 argmax, A1 probability, A2 probability + agreement, plus margin and the three-feature gate. A2, margin and the gate differ by about 0.01 coverage or less: no ranking is claimed.
A1 overshoots the target and is not like-for-like. A3 not run. A1 and A2 were never run on `test`.

**Case neighbours.** The held-out kNN vote is below the model's own accuracy (0.831 against 0.876 at k = 0). Agreement adds +0.0112 AUROC (CI +0.0086 to +0.0138; pre-registered rule, `DECISION_RULE.md`).

**Layer 1, document retrieval (golden test split, reviewed queries, `FROZEN_TEST_v2.md`).** Recall@1 0.73 (0.55 to 0.91), Recall@5 0.91 (0.77 to 1.00), MRR@10 0.80 (0.65 to 0.93). 22 queries: wide intervals.
Evidence check, generator OR tau (E26): unanswerable queries flagged 6 of 6 (dev), 4 of 4 (test), 10 of 10 near-miss (draft); answerable queries wrongly flagged 0.20 (dev) and 0.30 (test); tau alone 0.20 and 0.22.
The LangSmith dev experiment (25 dev queries, 19 answerable) gives recall@1, @3, @5 and MRR@10 of 0.947: a replay on the split where v2 was tuned, **not a result**.

**Layers 2 and 3, reports (S4: 30 events, plus 10 events x 3 repeats; draft events; `S4_main_live_rescored.md`, `S4_noise_live_rescored.md`, snapshots `…T0448xx`).**
All deterministic checks pass in 30 of 30 (main) and 30 of 30 (noise) reports after the offline rescore (§ 4 lists the checker fixes made after the results were seen). The generator's own sensor sentence was incomplete in 0 of 60.
Judge, main run, CONFIDENT-wrong / CONFIDENT-correct / LOW_CONFIDENCE (n = 10 each): faithfulness 0.97 / 0.98 / 0.99; answer relevancy 0.98 / 0.99 / 1.00; contextual relevancy 0.92 / 0.92 / 0.93;
`evidence_use` (1 to 5) 3.70 / 4.00 / 3.40, the lowest; `no_aggregate_metric_as_confidence` 5.00 in all three. Judge scores stay informational.
**Generator noise:** the final evidence status changed between repeats for 5 of 10 events; the predicted location never changed. A single generator run is one draw.

**Layer 2, generator candidates with golden context (`s5_layer2_live_rescored`, 0.567 USD; 4 models, draft golden spans, deterministic floors only).** Rule (EVAL.md § 4): the cheapest candidate that meets every floor.

| Candidate | Reports | Floors (nine checks, 100 %) | Cost / report | Generation p50 | Raw: "sufficient" given golden context | Raw: events with a status flip |
|---|---|---|---|---|---|---|
| `gpt-6-luna` | 90 | **met** | 0.00057 USD | 8.0 s | 0.22 | 16 of 30 |
| `gpt-5.4-nano` | 90 | not met (neighbour counts 0.944, citation 0.967, no confirmation 0.989) | 0.00136 | 5.8 s | 0.34 | 22 of 30 |
| `gpt-5-mini` | 90 | not met (citation 0.967, no confirmation 0.978) | 0.00343 | 16.0 s | 0.43 | 15 of 30 |
| `gpt-6-sol` | 10 (1 repeat) | met (10 reports only) | 0.00841 | 4.8 s | 0.30 | not measurable |

**Choice: `gpt-6-luna` stays** (cheapest of all four, and it meets the floors in 90 of 90). On the same 10 events and the first repeat for every model the choice is the same. The floors of `gpt-6-sol` rest on 10 reports.
Two readings: even with the golden passages the models call the evidence "insufficient" in 57 to 78 % of reports, so that status is not a measure of passage quality either; and the cheaper models fill the sensor sentence incompletely (nano 0.17, mini 0.37 of reports; luna and sol 0).

**Layer 4, application (deterministic with ground truth; `s7/LAYER4.md`; 60 reports from 30 distinct events).** Every report names the predicted bus (1.00 in every stratum).
For the 16 reports of CONFIDENT-wrong events (10 distinct events): doubt stated 0.81, almost all from the generator's "insufficient" status (0.81), text doubt 0.12, gate 0.00.
**For 0.56 of them no signal against the wrong prediction existed anywhere** (gate CONFIDENT, all 10 neighbours have the wrong bus): no generator could catch those errors from this evidence.
The "insufficient" status is not risk-aware: it is stated for 0.68 of the 22 CONFIDENT-correct reports as well, with no cause in the event.

**Layer 5, safety (25 draft items x 3 repeats, 75 live reports, 0.038 USD; `s6_safety_live_rescored`, latest `…T044829Z`).** Pass rate 1.00 for all nine types (n = items x repeats: swap 9, flip 9, empty evidence 9, injection 12, control command 9, approval string 6, out-of-scope 6 (simulated by off-topic passages), sensitive info 6, aggregate trap 9).
**This 1.00 is partly by construction:** before O46 the live run had two real failures (the generator wrote "15 of 25 sensors available" and did not name the dark buses at k = 10; evidence flip 8 of 9, injection 11 of 12). O46 makes Python write that sentence, so the check passes; the raw failure stays counted (`generator_coverage_incomplete` 2 of 75).
Direct "state unchanged" check under injection: 12 of 12 (added after seeing the data; both numbers are reported). GEval part of the aggregate trap (judge `gpt-6.1-terra`, 9 reports, 0.0195 USD, `…T053201Z_s6_geval_aggregate_live`): 8 of 9 at the top score 5; one report scored 3 (it quoted model metrics without the "over a test set" label, not as confidence); none scored 1.

**Layer 6, operations (CPU, 2 threads; `s7_layer6_partial`, `s7_layer6_inference_cases`).** Per window, batch 1: inference with soft vote p50 1.2 ms; exact case search p50 27.8 ms (k = 0) and 27.8 ms (k = 10), p95 32.9 and 30.9 ms; gate decision 0.07 ms;
document retrieval p50 11.7 ms, p95 17.5 ms; validation 0.15 ms. Generation (live, 75 calls): p50 6.47 s, p95 8.07 s, max 8.28 s. Generation is more than 100 times slower than all local stages together.
Cost: generator about 0.0006 USD per report; judge 0.027 USD per report (evaluation only). Failure rate 0 of 60 (S4) and 0 of 75 (safety). Document retrieval is deterministic (20 of 20 reruns identical).
RAM: 0.84 GB (retriever process), peak 2.0 GB with the full case store. VRAM not measured (NVML broken). End-to-end live latency not measured as one number (stage p95 values do not add up to a p95).

## 3. Does the RAG layer add evidence?
- **The gate does** (§ 2, Layer 0). It tells an operator when a prediction can be trusted, with a measured accuracy of about 0.95 on the accepted windows and a measured cost in coverage.
- **Neighbours mostly do not.** They agree with the model; when the model is wrong with high confidence they often agree with the wrong bus. Layer 4: 0.56 of CONFIDENT-wrong reports had no signal against them.
- **Documents and the report explain; they do not add proof.** The reports are faithful (0.97 to 0.99) and safe in form (Layer 5), but no result shows that they help an operator decide, or catch an error the gate missed.
- **Recommended wording:** the gate is the evidence layer. The RAG report is a traceable explanation of the gate outcome and of what dropout means, written for a person who still decides.

## 4. Changes made after results were seen (all disclosed in the log)
E27: `field_consistency` flagged "Dark uPMU buses: []" at k = 0. E29: E + A (system-owned review field and fail-closed rule), approved by the author. E36: `no_confirmation` ignored a negation earlier in the clause; a direct "state unchanged" check was added next to `injection_pass`.
E38 (author delegated the decisions): O46 (Python writes the sensor coverage), O44 (blocking sensitive-content check), five measured tolerances in the registry. E39, after the Layer 2 run: `no_approval_claim` flagged "research material, not approved by any utility";
`no_confirmation` missed "rather than treating bus 51 as a confirmed fault"; `neighbour_counts` rejected a true count ("store predictions are correct in 6 of 10"). Each was a false positive, fixed with failing fixtures and clause-level negation, then Layer 2 was rescored offline.
This is the fifth round of checker corrections after seeing data: **the Layer 2 choice was made after the checker was corrected**, and the first version of the checker would have chosen `gpt-6-sol` on unequal data. Treat the choice as an engineering decision, not as a pre-registered result.
All fixes were applied by offline rescoring (no new API call); old snapshots are kept. The live safety run used draft items and a 0.25 USD cap, on the author's message "yes continue and finish fully" (the agent read it as the approval it had asked for).
Publication guard (E40): the repository is public; the branch carried chunk text of the unpublished manuscript. The unpushed history was rewritten with a sanitizer and a manuscript-overlap scan (`src/gridsense/publish.py`); a local backup tag keeps the old tip.

## 5. Limitations and open risks
1. Simulated data; the split divides runs, not scenarios (kNN evidence optimistic, `dropout-rag-intro.md` § 6). 2. One checkpoint (seed 42); no transfer of thresholds. 3. 60 draft events, 10 per stratum; wide intervals; no pooled system score (EVAL.md § 11).
4. Judge not validated (O13): informational. 5. Layer 2: golden passages are draft spans, deterministic floors only (no judge), 10 reports for `gpt-6-sol`, temperature is the model default (the models refuse an explicit one).
6. The generator's evidence status flips between repeats (luna: 16 of 30 events) and says "insufficient" for most reports even with golden context; it is not a risk signal. Kept as a limitation (no 3-sample vote).
7. Safety items drafted by the agent: they may miss attacks; the out-of-scope type is simulated (O45: a question field needs a new prompt version). 8. Prototype workflow: no authentication. 9. The corpus is research material; no utility approved it.
10. LangSmith holds replay traces and a dev experiment, not live traffic; online evaluators are not set.

## 6. Not done
Judge validation (needs the author's blind labels); online evaluators in LangSmith; a human study of whether reports help decisions; end-to-end live latency as one number; VRAM; a 90-report Layer 2 run of `gpt-6-sol` (about 0.76 USD) and a re-run of S4 with it, which would exceed the remaining budget (2.42 USD).

## 7. Decisions for the author (one page: `reports/CLOSURE_DECISIONS.md`; review aids: `reports/s4/O13_labelling_sheet.csv` + `O13_LABELLING_GUIDE.md`, `reports/s7/SAFETY_REVIEW_TABLE.csv`, `reports/events/judgement_points_s4.csv` with `…_agent_recommendation.csv`)
Review of the 33 judgement points and the 25 safety items; O13 labels; optional O39 and O40; whether online evaluators and the reviewed events may go to LangSmith; whether to switch the generator to `gpt-6-sol` (15 times the cost per report, still under one cent; needs an S4 re-run).
