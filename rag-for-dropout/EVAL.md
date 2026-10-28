# EVAL.md — GridSense evaluation design (component level and end to end)

Written 2026-10-28 (S0). Source: `rag-eval.md` (`spec/eval_01–04.md`) and `implementation-plan.md` §13
(`spec/impl_08–09.md`), adapted to the repo and to the author decisions in `claude-rag.md` § 3.
Order of work: free deterministic metrics first, then paid LLM-judge metrics on approved configurations only.
Values marked **(P)** are proposals. The author approves them (`OPEN_ITEMS.md` O12) before a reported run.

## 1. Common protocol
- Unit: the window. Dropout grid k ∈ {0, 1, 2, 5, 10}; 3 mask seeds. System seed 42; Layer 0 repeated for seeds
  43 and 44 if each takes < 30 min of CPU (O5).
- Confidence intervals: bootstrap over **runs** (all windows of a run stay together), 2,000 resamples (P), 95 %.
- Seed comparisons: paired t-test over 3 seeds, Holm per family (root `CLAUDE.md`). Trends stay trends.
- Splits: `val` for every choice; `test` once per frozen configuration. Golden sets have their own `dev` / `test`.

## 2. Layer 0 — prediction, gate, case evidence (free)
| Metric | Notes |
|---|---|
| Top-1 accuracy, macro-F1 (26 classes) | Must equal the paper numbers at each k before anything else (`IMPLEMENTATION.md` § 4) |
| ECE (15 equal-mass bins (P)), Brier on top-1 correctness | Calibrated top-1 vote share (vote / 25; Platt vs isotonic fit on cal-A, compared on cal-B), per k. Argmax never changes |
| Coverage, selective risk, AURC | Risk-coverage curve; τ selected on cal-B; selective risk at the coverage of the chosen τ |
| kNN-vote accuracy, k_nn = 5/10/20 | Run-grouped LOO for `val`; `test` queries the train store |
| Error-detection AUROC and AUPRC | Signals: calibrated top-1, top-1 − top-2 vote-share margin, kNN agreement, neighbour error rate, combined (fit on `val`) |
| Candidate-set coverage and mean size | LOW_CONFIDENCE windows; conformal coverage vs nominal |
| Error recall | Wrong predictions sent to LOW_CONFIDENCE or REJECT |
| Report-window choice (O3) | Top-1 accuracy per window index 21–39 on `val` (w − 20 post-fault samples); earliest index where it stops rising |
| Latent method (O4) | kNN-vote accuracy on `val`: concatenation (1,600-d) vs mean + max pooling (128-d) |
| Informational | Event-level aggregation (40 windows of a run); accuracy by window position; scenario-held-out control; leakage sanity run |
Questions to answer: does ECE hold as k grows? Does kNN agreement degrade more slowly than the soft-vote score?

## 3. Layer 1 — retrieval, component sweeps (free first)
Golden set: `golden_retrieval.jsonl` (`data/eval/SCHEMA.md`). Metrics: Recall@k, Precision@k, MRR, nDCG@k (k = 1, 3,
5, 10), exact-identifier recall, no-answer behaviour (the sufficiency check says "insufficient"), latency p50/p95.
Then, for approved configurations only: DeepEval contextual precision and contextual recall.

Sweep order (one variable at a time; the winner of each step is fixed for the next; `rag-component-sweep`):
| Step | Axis | Candidates | Fixed during the step |
|---|---|---|---|
| 1 | Chunker | fixed-token, recursive, structure-aware (LaTeX section / md heading), sentence, semantic, parent-child | MiniLM, dense, FAISS-flat, top_k 5 |
| 2 | Chunk size and overlap | 3 sizes inside the embedder window; overlap 0 / 10–15 % | winner of 1 |
| 3 | Embedder | `all-MiniLM-L6-v2` baseline; 2 small open retrieval models (model card checked: window, prefixes, licence); optional OpenAI embedding (paid, O1) | winners of 1–2; re-chunk if the window differs |
| 4 | Retrieval mode | dense; BM25; hybrid RRF (k_rrf = 60 (P), swept on `dev`) | winners of 1–3 |
| 5 | Reranker | off; one small cross-encoder on the top 20 | winner of 4 |
| 6 | top_k, candidate counts | top_k 3/5/8; candidates 20/50 | winner of 5 |
| 7 | Vector store | FAISS-flat (reference), Chroma, Qdrant local mode | winner of 6 |
Rules: never compare chunk sizes above the embedder input window (log truncation per chunk). Prefixes are applied in
one function. Each step writes a snapshot and one row per candidate to `reports/sweeps/<axis>.md`.

**Vector-store step.** With exact search, every store returns the same ranking for the same vectors, so retrieval
quality does not choose the store. Measure: ranking parity with FAISS-flat (must be 100 % for exact mode), ANN recall
vs exact (HNSW settings), p50/p95 query latency, build time, peak RAM, disk size, metadata filtering, hybrid support,
persistence and re-open time. Choose by these operational numbers. Record the choice and the reason.

## 4. Layer 2 — generator on golden context
Input: golden event object + golden case neighbours + golden passages (no live retrieval).
| Metric | Type |
|---|---|
| Faithfulness, answer relevancy | DeepEval, judge model |
| Field consistency (bus, score, gate outcome, candidate set, k) | Deterministic, must be 100 % |
| Schema validity | Deterministic (Pydantic), must be 100 % |
| Citation validity (`document_id` exists, excerpt is a substring) | Deterministic, must be 100 % |
Candidates: 2–3 OpenAI generator models (O1), temperature as configured, 3 repeats each; report mean and spread,
tokens, cost per report. Choose the cheapest model that meets the floors, not the highest mean.

## 5. Judge LLM selection and validation
1. Candidates: 2 OpenAI models that differ from the chosen generator (O1). Temperature 0. Pinned snapshot names.
2. Human labels: about 50 reports, labelled by the author for faithfulness and for each rubric (`golden_human_labels.jsonl`).
3. Per judge and metric: Cohen's κ, % agreement, confusion table. Also a position/verbosity probe: the same report
   with sections reordered and with padding; the score must not change by more than the measured noise.
4. Pick the judge with the highest κ. If κ < 0.6 (P) for a metric, that metric is informational and the human labels are
   reported. 
5. Cache key: (judge model, rubric hash, input hash). Never pay twice for one judgment.

## 6. Layer 3 — end-to-end RAG triad (live retrieval)
Contextual relevancy, faithfulness, answer relevancy on the golden events, with live gate, case and document
retrieval. Per event record: event ID, STGNN output, gate outcome, neighbours, evidence score, retrieval query,
passages (`document_id`, section, hash), report, each metric with the judge reason.

## 7. Layer 4 — application checks
| Check | Type |
|---|---|
| Location correctness (report names the predicted bus; CONFIDENT vs truth) | Deterministic |
| Uncertainty correctness (wrong → doubt stated; right → no doubt without cause) | Deterministic with ground truth |
| Prediction is a candidate, not a confirmed fault | GEval `rubrics/prediction_vs_confirmed.md` |
| Completeness, adherence | GEval `rubrics/completeness.md`, `rubrics/adherence.md` |
| Evidence use (neighbours used; no general LLM knowledge as procedure) | GEval `rubrics/evidence_use.md` |
| No aggregate metric as confidence | GEval `rubrics/no_aggregate_metric_as_confidence.md` + regex trap |

## 8. Layer 5 — safety
Counterfactual swap, evidence flip, empty evidence, prompt injection (state unchanged, deterministic), no control
command (regex + GEval), approval string, out-of-scope question, sensitive information (regex), aggregate-metric trap
(model-card passage with "0.839 macro-F1" in context; the report must not present it as this window's confidence),
toxicity (small set). Pass rate per test over ≥ 3 runs, with n.

## 9. Layer 6 — operations
Latency p50/p95 per stage (inference, gate, case retrieval, document retrieval, generation, validation); tokens per
generation and judge call; cost (prices with a date, from the author); failure rate; retries; determinism (Layers 0–1
identical on rerun); peak RAM and VRAM.

## 10. Ablation ladder (at every k, same test windows)
A0 STGNN argmax → A1 + calibrated threshold → A2 + kNN agreement → A3 + silent-failure check (NOT RUN unless approved)
→ A4 + retrieval and LLM report. Report coverage, selective risk, AURC, false acceptance, error recall, latency.

## 11. Golden datasets
| Set | Size | Content | Split |
|---|---|---|---|
| `golden_retrieval.jsonl` | 40–60 | paraphrases, exact identifiers, dropout/observability, ≈15 % unanswerable | dev / test |
| `golden_events.jsonl` | 60–100 | first fault window per run; all k, all 11 fault types; correct CONFIDENT; high-confidence errors; LOW_CONFIDENCE | test |
| `golden_safety.jsonl` | 20–30 | swaps, flips, empty evidence, injection, control-command, aggregate-metric, out-of-scope | test |
| `golden_human_labels.jsonl` | ≈50 | author labels for judge validation | — |
LLM drafts → `status: draft` → author review → `reviewed`. Version per set; version in every snapshot. Event
candidates are chosen by a seeded stratified sampler (code), not by hand.

## 12. Registry, tolerance, comparison
`configs/registry.yaml`: name, direction, role (`gate` / `guardrail` / `informational`), floor, tolerance,
`tolerance_source`. Tolerance = 2 × std over 3–5 repeat runs (`measured`); before that `provisional`.
Compare (`rag-compare`): FAIL if a gate metric fails, a floor is not met, or snapshots are not comparable (golden
version, split, k grid); REVIEW if a guardrail regresses more than its tolerance, CIs overlap, or a tolerance is
provisional; PASS otherwise. The baseline must meet its floors too.

## 13. LangSmith (offline experiments and online evaluation)
- Datasets: each reviewed golden set → one LangSmith dataset, version tag = golden version. Drafts are never pushed.
- Offline experiments: the same evaluators as the local suite (DeepEval metrics wrapped as LangSmith evaluators, plus
  the deterministic checks). The experiment URL goes into the local snapshot. The local snapshot stays the source of truth.
- Tracing: one project per environment; each LangGraph node is a run; metadata = model, index and policy versions.
- Online evaluators on sampled production traces (sample rate in config): LLM judge for faithfulness and
  no-aggregate-confidence; code evaluators for field consistency and the control-command pattern. Feedback scores
  per trace; a dashboard per metric; an alert when a gate-role metric fails.
- Traffic: there is no live grid. A replay stream of `val`/`test` events stands in for it. Say so in every result.
- Redaction: no keys, local paths or env values in traces. Ask before the first upload (O8).
