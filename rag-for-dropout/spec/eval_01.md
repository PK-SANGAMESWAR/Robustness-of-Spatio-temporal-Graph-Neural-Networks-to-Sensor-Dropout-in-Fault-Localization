<!-- Source: rag-eval.md lines 1-63, verbatim. Do not edit. Index: 00_INDEX.md -->

# rag-eval.md — End-to-end evaluation framework for GridSense

**Type:** Prompt for Claude Code
**Language:** ASD-STE100 (Simplified Technical English)
**Project:** GridSense — STGNN fault prediction, deterministic gate, case retrieval, document retrieval, LLM report
**Reference:** `GridSense_PLAN.md` revision 2, sections 8.6, 11, 12, 13.7 to 13.13

---

## 1. Your task

Make an end-to-end evaluation framework for the GridSense pipeline.
The framework must measure each layer of the pipeline and the full pipeline.
The framework must use the real pipeline code. Do not make a copy of the pipeline for the evaluation.

Do the work in stages (see section 12).
After stage 2, stop. Show the plan to me. Wait for my approval.
Do not change files before I approve the plan.
Do not start a paid or long evaluation before I approve the cost.

---

## 2. Rules that apply to all stages

1. Read the repository before you write code.
2. Do not make changes that are not related to the evaluation.
3. Do not change the behavior of the application. If the evaluation needs a hook in the application, tell me first.
4. Use the existing code, configuration and libraries when they can do the task. Do not add a dependency that is not necessary.
5. Use DeepEval for LLM-judged metrics, unless the repository already has a different evaluation framework. If it does, tell me and use that framework.
6. Do not do expensive work during import. Load models, indexes and API clients only when a function needs them.
7. Load environment variables in one location only (for example, one `settings` module that reads `.env`).
8. Do not hide errors. If a metric fails, record the failure and the cause. Do not record the failure as a score of 0 or 1.
9. Do not put API keys, tokens or secrets in code, logs, snapshots or Git.
10. After each stage, do these steps:
    - Tell me what you changed and why.
    - Run the inexpensive tests that apply to the stage.
    - Add an entry to the experiment log (see section 11).

---

## 3. Stage 1 — Inspect the repository

Find and record these items. Give the file path and the function name for each item.

1. The STGNN inference entry point, the checkpoint, and the preprocessing code.
2. The output of the STGNN: logits or probabilities, the class definition (bus, section or fault type), and any ambiguity set.
3. The calibration code (for example, temperature scaling), if it exists.
4. The deterministic gate: the thresholds, the reason codes and the outcomes (`CONFIDENT`, `LOW_CONFIDENCE`, `REJECT`).
5. The case store: which layer gives the latent vector, which split is in the store, and the vector database.
6. The document corpus: the sources, the parser, the chunk size, the overlap, the embedding model, its input-length limit, and its prefixes.
7. The retriever: dense, lexical, hybrid, reranker, `top_k`, and the metadata filters.
8. The LangGraph workflow: the nodes, the state fields and the routes.
9. The report generator: the LLM model, the prompts, the output schema and the validation.
10. The existing tests, evaluation scripts, metrics and result files.
11. The configuration files and the location of each setting.
12. The data splits: train, calibration and test. Find how the code keeps them apart.

**Important:** Some components in `GridSense_PLAN.md` possibly do not exist yet.
Do not assume that a component exists. If you cannot find a component, write "not found".
Do not make a component during this stage.

---

