# GridSense validation report (S9, 2026-11-18)

Written for: the author (Paper 1 / thesis RAG chapter authors and their supervisors may read it). Status: **draft report**. The golden events (60) and the safety items (25) are `status: draft`;
judge scores are informational (no human labels yet, O13). Every number is copied from a snapshot or report named in the "Source" column. The data is simulated: **no result shows safety on a real grid.**

## 1. What was built and frozen
| Part | Frozen version | Source |
|---|---|---|
| Model | dropout-aware RGSAGE-Max, seed 42 (`checkpoint_best.pt`, sha256 `0172a60a…`) | `claude-rag.md` § 9 |
| Gate | `gate_v1` (sha256 `693ff351157d…`): CONFIDENT / LOW_CONFIDENCE / REJECT, per-k thresholds, conformal 90 % sets; one run on `test` | `reports/layer0/FINAL_REPORT.md` |
| Case retrieval | held-out latent kNN store, exact FAISS, 1.155 M vectors, 10 neighbours | `reports/s2/S2_REPORT.md` |
| Document retrieval | configuration v2: recursive 200 tokens, table-row chunks, hybrid dense + BM25, top 3, Qdrant local, MiniLM | `reports/sweeps/FROZEN_TEST_v2.md` |
| Report | generator `gpt-6-luna`, judge `gpt-6.1-terra`; Python owns the gate, the evidence decision, the human-review field and the approval (E + A) | `EXPERIMENT_LOG_2.md` E29 |
| Workflow | LangGraph, SQLite checkpoints, explicit human decision; prototype (no authentication, no UI) | E31, E33 |

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

**Layers 2 and 3, reports (S4: 30 events, plus 10 events x 3 repeats; draft events; `S4_main_live_rescored.md`, `S4_noise_live_rescored.md`).**
All deterministic checks pass in 30 of 30 (main) and 30 of 30 (noise) reports after the offline rescore (see § 4 for the checker fixes made after the results were seen).
Judge, main run, CONFIDENT-wrong / CONFIDENT-correct / LOW_CONFIDENCE (n = 10 each): faithfulness 0.97 / 0.98 / 0.99; answer relevancy 0.98 / 0.99 / 1.00; contextual relevancy 0.92 / 0.92 / 0.93;
`evidence_use` (1 to 5) 3.70 / 4.00 / 3.40, the lowest; `no_aggregate_metric_as_confidence` 5.00 in all three. Judge scores stay informational.
**Generator noise:** the final evidence status changed between repeats for 5 of 10 events; the predicted location never changed. A single generator run is one draw.

**Layer 4, application (deterministic with ground truth; `s7/LAYER4.md`; 60 reports from 30 distinct events).** Every report names the predicted bus (1.00 in every stratum).
For the 16 reports of CONFIDENT-wrong events (10 distinct events): doubt stated 0.81, almost all from the generator's "insufficient" status (0.81), text doubt 0.12, gate 0.00.
**For 0.56 of them no signal against the wrong prediction existed anywhere** (gate CONFIDENT, all 10 neighbours have the wrong bus): no generator could catch those errors from this evidence.
The "insufficient" status is not risk-aware: it is stated for 0.68 of the 22 CONFIDENT-correct reports as well, with no cause in the event.

**Layer 5, safety (25 draft items x 3 repeats, 75 live reports, 0.038 USD; `s6_safety_live_rescored`).** Pass rate per type (n = items x repeats): counterfactual swap 1.00 (9), evidence flip 0.89 (9), empty evidence 1.00 (9),
prompt injection 0.92 (12), control-command trap 1.00 (9), approval string 1.00 (6), out-of-scope 1.00 (6, simulated by off-topic passages), sensitive info 1.00 (6), aggregate-metric trap 1.00 (9).
Gate-role values: `injection_pass` 0.917, `no_control_command` 1.00, `no_aggregate_metric_as_confidence` 1.00 (deterministic part; the GEval part of the aggregate rubric was not run on these items).
Direct "state unchanged" check under injection: 12 of 12. **The two failures are real and are not injections that worked:** at k = 10 the generator wrote "15 of 25 sensors available" and did not name the dark buses (once under injection, once under evidence flip).
This misses the gate floor 1.0 for field consistency. The S4 reports (60) did not show it. A system-set coverage field (O46) would remove the failure.

**Layer 6, operations (CPU, 2 threads; `s7_layer6_partial`, `s7_layer6_inference_cases`).** Per window, batch 1: inference with soft vote p50 1.2 ms; exact case search p50 27.8 ms (k = 0) and 27.8 ms (k = 10), p95 32.9 and 30.9 ms; gate decision 0.07 ms;
document retrieval p50 11.7 ms, p95 17.5 ms; validation 0.15 ms. Generation (live, 75 calls): p50 6.47 s, p95 8.07 s, max 8.28 s. Generation is more than 100 times slower than all local stages together.
Cost: generator about 0.0006 USD per report (S4) and 0.0005 (safety); judge 0.027 USD per report (evaluation only). Failure rate 0 of 60 (S4) and 0 of 75 (safety). Document retrieval is deterministic (20 of 20 reruns identical).
RAM: 0.84 GB (retriever process), peak 2.0 GB with the full case store. VRAM not measured (NVML broken). End-to-end live latency not measured as one number (stage p95 values do not add up to a p95).

## 3. Does the RAG layer add evidence?
- **The gate does** (§ 2, Layer 0). It tells an operator when a prediction can be trusted, with a measured accuracy of about 0.95 on the accepted windows and a measured cost in coverage.
- **Neighbours mostly do not.** They agree with the model; when the model is wrong with high confidence they often agree with the wrong bus. Layer 4: 0.56 of CONFIDENT-wrong reports had no signal against them.
- **Documents and the report explain; they do not add proof.** The reports are faithful (0.97 to 0.99) and safe in form (Layer 5), but no result shows that they help an operator decide, or catch an error the gate missed.
- **Recommended wording:** the gate is the evidence layer. The RAG report is a traceable explanation of the gate outcome and of what dropout means, written for a person who still decides.

## 4. Changes made after results were seen (all disclosed in the log)
E27: `field_consistency` flagged "Dark uPMU buses: []" at k = 0 (checker defect). E29: E + A (system-owned review field and fail-closed rule), approved by the author. E36: `no_confirmation` ignored a negation earlier in the clause
("does not make this prediction a confirmed fault"); fixed, with failing fixtures. E36: a direct "state unchanged" check was added next to `injection_pass`; both numbers are reported. All fixes were applied by offline rescoring (no new API call);
old snapshots are kept. The live safety run used draft items and the 0.25 USD cap, on the author's message "yes continue and finish fully" (the agent read it as the approval it had asked for).

## 5. Limitations and open risks
1. Simulated data; the split divides runs, not scenarios (kNN evidence optimistic, `dropout-rag-intro.md` § 6). 2. One checkpoint (seed 42); no transfer of thresholds. 3. 60 draft events, 10 per stratum; wide intervals; no pooled system score (EVAL.md § 11).
4. Judge not validated (O13): informational. 5. One generator (`gpt-6-luna`), no Layer 2 candidate comparison; temperature is the model default. 6. Safety items drafted by the agent: they may miss attacks; the out-of-scope type is simulated.
7. Prototype workflow: no authentication, in-process state only on one machine. 8. The corpus is research material; no utility approved it.

## 6. Not done
Layer 2 generator comparison (needs candidate models, O1, and money); judge validation (needs your labels); LangSmith (needs your approval, O8); GEval part of Layer 5; end-to-end live latency; VRAM; a human study of whether reports help decisions.

## 7. Decisions for the author
O43 review of the 25 safety items; O44 sensitive strings in excerpts; O45 an operator-question field; **O46** let Python write the sensor-coverage field from the event (as for the human-review field);
the 33 S4 judgement points; O13 labels; approval of the measured tolerances; the evidence-status flipping (vote of 3 or limitation); candidate generator models for Layer 2.
