# PLAN.md — GridSense RAG track: staged plan

Written 2026-10-28 (stage S0). This file is the stage-2 plan that `rag-eval.md` §12–13 asks for. Rules: `claude-rag.md`.
Design details: `IMPLEMENTATION.md`. Metrics: `EVAL.md`. Each stage ends with a **stop**: report to the author, add an
entry to `EXPERIMENT_LOG.md`, update `DONE.md`, and wait for approval. Do not start a stage before the author approves.

## Overview
| Stage | Content | Cost | Blocked by |
|---|---|---|---|
| S0 | Support files, skills, config skeleton, stage-1 inspection | 0 | — (done 2026-10-28) |
| S1 | Author closes the open items that S2–S3 need | 0 | — (done 2026-10-28) |
| S2 | `rag` dependency group; prediction wrapper; latents; case store; Layer 0 | 0 (CPU) | done 2026-10-30 (frozen gate, one test run, seeds 43/44): `reports/layer0/FINAL_REPORT.md` |
| S3 | Corpus ingestion; retrieval golden drafts; free component sweeps (Layer 1, deterministic) | 0 (CPU) | **done 2026-11-02:** O30 review applied (golden v1), 8 sweep axes and the vector store run on dev, configuration frozen and run once on the golden test split (`reports/sweeps/SUMMARY.md`). Open: O34 (12 draft documents). Next stage S4 needs O1 (generator and judge models, prices). SUPERSEDED: session B baseline (E11) |
| S4 | Cost estimate; judge-validation pilot; DeepEval contextual metrics on the top configs | paid | **prepared 2026-11-02, not started** (`S4_PLAN.md`, `reports/s4/COST_ESTIMATE.md`: expected 1.29 USD, 0.65 with Batch; ceiling 2.43); needs the author's approval, O35 to O38 and the golden events |
| S5 | Generator, LangGraph workflow, report schema; Layers 2 and 3 | paid | S4 |
| S6 | Layers 4, 5, 6: application, safety, operations; ablation ladder A0–A4 | paid | S5 |
| S7 | Registry, snapshots, compare command, regression tests, measured tolerances | small | S6 |
| S8 | LangSmith: datasets, experiments, tracing, online evaluators, replay stream | small | O8, S7 |
| S9 | Validation report: actual results, limitations, decisions | 0 | S8 |

## S0 — Support files (this session)
Write the docs in `rag-for-dropout/`, the verbatim `spec/` split, six `rag-*` skills, `configs/eval.yaml` and
`configs/registry.yaml` (skeletons), GEval rubric drafts, `data/eval/SCHEMA.md`, `reports/stage1_inspection.md`,
`src/gridsense/__init__.py`, `tests/gridsense/test_docs.py`, a deny rule for the frozen sources, a 2-line import in
the root `CLAUDE.md`. **Exit:** `uv run pytest tests/gridsense -q` passes; every new md file < 200 lines.

## S1 — Close the open items
The author answers `OPEN_ITEMS.md` O2 (calibration input), O3 (report trigger), O4 (latent layer and pooling),
O5 (case-store seed), O9 (cost limit), O10 (git), O11 (paper version in the corpus). O1, O6, O7, O8 can wait until S4–S8.
**Exit:** each answer is in `claude-rag.md` § 3 or in `configs/eval.yaml`, with the date.

## S2 — Prediction wrapper, case store, Layer 0 (CPU, free)
1. Add a `rag` dependency group (faiss-cpu, rank-bm25, sentence-transformers, chromadb, qdrant-client, deepeval,
   langgraph, langchain-core, langsmith, openai, pydantic). Ask first. Check the torch pin does not move.
2. `gridsense.model.predictor`: load `checkpoint_best.pt` read-only, run the frozen model, apply dropout with
   `fault_localization.dropout.masking` (same seeds as `dropout/evaluator.py`), return scores, top-1, latent.
3. Latent extraction on CPU through `pinns.cpu_guard`, nice 19, measured RAM. Train split → case store; val and test →
   query sets. Store pooled 64-d latents (float16 if RAM needs it) with run ID, label, prediction, k, mask hash.
4. Split `val` by run into cal-A and cal-B. `gridsense.verification.calibration`: vote share = vote / 25; Platt and
   isotonic maps on the top-1 share, fit on cal-A, compared on cal-B (ECE, Brier on top-1 correctness). Record the
   top-1 − top-2 margin. Assert that argmax predictions equal the paper predictions. Thresholds on cal-B.
4b. Report window (O3): top-1 accuracy per window index 21–39 on `val`; earliest index where it stops rising.
4c. Latent method (O4/O15): (a) concat, (b) mean + max, (c) concat + PCA-256, on one 10 % train-run subsample.
4d. Identity gate: k = 0 test predictions = `predictions_test.csv`; val macro-F1 = `best_validation_graph_f1`.
5. Layer 0 metrics (`EVAL.md` § Layer 0) at k ∈ {0,1,2,5,10}, seed 42; seeds 43 and 44 if each < 30 min CPU (O5).
6. Leakage: assert no test run in the store; LOO excludes the same run; sanity run with test in the store
   (kNN accuracy should rise to near 1.0); informational scenario-held-out control.
**Decision at the stop:** does kNN agreement add error-detection AUROC over the soft-vote score? If not, simplify the
gate (`spec/impl_12.md` §22 step 1a). **Exit:** Layer 0 snapshot; deterministic rerun gives identical numbers.

## S3 — Corpus, retrieval golden set, free component sweeps
1. Ingest Paper 1 (`paper/overleaf.tex`; which version: O11) and `results/RESULTS.md`
   as `model_card` documents with section paths and content hashes. Parsers: LaTeX sections, md headings.
2. Draft 40–60 retrieval queries (`rag-golden-review`): paraphrases, exact identifiers (bus, k, table), dropout and
   observability questions, about 15 % unanswerable, split `dev` / `test`. **Stop** for the author's review.
3. Run `rag-component-sweep` on `dev` with deterministic metrics only, one variable at a time:
   chunker → chunk size → embedder → retrieval mode → reranker → top_k → vector store (operational).
4. Freeze the top 2–3 configurations. Run them once on `test`. **Exit:** sweep table in `reports/`, snapshots saved.

## S4 — Judges and paid retrieval metrics
1. `rag-cost-check` with the author's prices (O1). The whole track has **5 USD** (O9); every paid stage must fit in
   the remaining budget in `eval/cost_ledger.jsonl`. **Stop** for approval.
2. Pilot: 5 items per layer. Then judge validation: about 50 human-labelled items; Cohen's κ and % agreement per judge
   candidate and metric. Pick the judge with the best agreement; it must differ from the generator.
3. DeepEval contextual precision / recall on the frozen retrieval configurations. **Exit:** judge choice recorded.

## S5 — Generator and workflow (Layers 2–3)
1. `gridsense.agent`: LangGraph states of `spec/impl_08.md` §12.1 (three gate outcomes); Pydantic report;
   citation validator; insufficient-evidence route; human-approval node (no automatic approval).
2. Draft 60–100 golden events (first fault window per run, test split, all k, all fault types, many high-confidence
   errors). **Stop** for review. Iterate prompts on `val` events only.
3. Layer 2 with golden context for 2–3 OpenAI generator candidates, 3 repeats each. Layer 3 live RAG triad.
**Exit:** generator chosen by Layer 2 metrics at a stated cost; field consistency, schema and citation validity = 100 %.

## S6 — Application, safety, operations (Layers 4–6)
Safety set of 20–30 items (counterfactual swap, evidence flip, empty evidence, injection, control-command trap,
aggregate-metric-as-confidence trap, out-of-scope, approval string). Ablation ladder A0–A4 at every k; A3 = NOT RUN
unless the Stage F silent-failure check is recorded as usable in `PC-STGNN/07_*`. Latency p50/p95, tokens, cost,
failure rate, RAM. **Exit:** all gate-role metrics pass; pass rates over ≥ 3 runs.

## S7 — Regression machinery
Registry with roles and floors; 3–5 repeat runs on one commit → tolerance = 2 × std (`measured`); compare command
(PASS / REVIEW / FAIL, Markdown + JSON); configuration comparison table; pytest for each item of `rag-eval.md` §10.1
with mocked LLM and embedder. **Exit:** baseline snapshot accepted by the author and listed in `DONE.md`.

## S8 — LangSmith online evaluation
Ask before the first upload (O8). Push the reviewed golden sets as LangSmith datasets (versioned). Run offline
experiments with the same evaluators. Trace the pipeline in a project. Online evaluators on sampled traces: LLM judge
(faithfulness, no-aggregate-confidence) and code (field consistency, control-command pattern). A replay stream of
val/test events stands in for live traffic; state this in every result. **Exit:** experiment URLs in snapshots.

## S9 — Validation report
`reports/validation_report.md`: actual results per layer, chosen configuration with its evidence, limitations
(`dropout-rag-intro.md` § 6), open risks. No result is upgraded beyond its statistics.

## Risks
| Risk | Effect | Control |
|---|---|---|
| Shared scenarios between train and test | Latent kNN looks too good | Scenario-held-out control; leakage sanity run |
| Small golden sets | Wide confidence intervals | Bootstrap CIs; no claim from a tiny set |
| Judge bias or drift | Wrong quality signal | κ against human labels; pinned model snapshot; cache |
| RAM (about 4 GB free) and busy GPU | Jobs fail or slow the S4.4 queue | CPU guard; pooled float16 latents; batches |
| Corpus is the paper itself | Retrieval questions become trivial or circular | Golden queries written from operator needs, not from section titles |
| First fault window is hard | Report-level numbers look worse than the paper | Window-position accuracy in Layer 0 |
