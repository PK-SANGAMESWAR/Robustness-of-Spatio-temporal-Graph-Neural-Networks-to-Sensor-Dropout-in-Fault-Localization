<!-- Source: implementation-plan.md lines 1512-1655, verbatim. Do not edit. Index: 00_INDEX.md -->

## 17. Security, privacy, and auditability

-   Keep API keys in environment variables or a secret manager, never
    source code.
-   Treat document text as untrusted input: retrieved text may contain
    instructions or prompt-injection content. The agent must treat it as
    evidence, not as executable instructions.
-   Restrict ingestion to approved sources and preserve document
    revision metadata.
-   Enforce access controls independently of semantic retrieval.
-   Log model version, policy version, index version, retrieved chunk
    IDs, gate results, report version, and human decision.
-   Minimize retention of raw telemetry and sensitive operational
    information.
-   Sanitize file uploads and validate file type/size.
-   Limit network access and external tool permissions for the report
    agent.
-   Do not give the agent any tool that directly operates grid
    equipment.
-   Make all retries, overrides, and human decisions auditable.
-   Ensure a document cannot grant itself approval by including text
    that says it is approved.

------------------------------------------------------------------------

## 18. Performance and the 8 GB NVIDIA T1000

The T1000's 8 GB VRAM budget is shared with the STGNN and any other GPU
workloads. Do not assume every model should be loaded onto the GPU.

Recommended approach: 1. Keep the STGNN on the GPU if it needs the
acceleration. 2. Benchmark the small embedding model on CPU first. 3.
Generate document embeddings in batches. 4. Use CPU for retrieval and
orchestration initially. 5. Use a small reranker on CPU or evaluate it
separately before deploying it alongside the STGNN. 6. Measure peak
VRAM, RAM, latency, and throughput for each component. 7. Avoid
simultaneously loading multiple embedding/reranking/LLM models onto the
GPU without measuring memory use. 8. If using a local LLM, treat it as
an optional report-drafting dependency and test its resource impact
independently.

The embedding index itself does not need to reside in GPU memory for a
small manual corpus. Start simple and optimize only when measurements
identify a bottleneck.

------------------------------------------------------------------------

## 19. Implementation milestones

### Milestone 0 --- Repository audit

-   Identify the existing STGNN entry point, model checkpoint,
    preprocessing code, inference requirements, and evaluation scripts.
-   Run the existing model and capture baseline metrics.
-   Identify available grid manuals and their approval status.
-   Record Python and dependency versions.
-   Document existing assumptions and known failure cases.

**Exit condition:** A reproducible baseline prediction can be generated
from a known event.

### Milestone 1 --- Stable event and prediction schemas

-   Create typed schemas.
-   Wrap inference behind a predictable interface.
-   Preserve sensor masks, timestamps, and model version.
-   Add input validation and unit tests.

**Exit condition:** Invalid inputs are rejected, and valid inputs
produce reproducible structured outputs.

### Milestone 2 --- Document ingestion and dense retrieval baseline

-   Build the source manifest.
-   Parse approved documents.
-   Compare initial chunking configurations.
-   Generate embeddings with a small local model.
-   Index chunks with provenance metadata.
-   Create a labelled query benchmark.
-   Measure Recall@k, MRR, nDCG@k, latency, and unanswerable-query
    behavior.

**Exit condition:** Retrieval results can be traced to the original
source pages/sections and are quantitatively evaluated.

### Milestone 3 --- Hybrid retrieval and optional reranking

-   Add lexical/BM25 retrieval.
-   Fuse results with RRF.
-   Evaluate hybrid versus dense-only retrieval.
-   Test a cross-encoder reranker if justified.
-   Select a configuration based on the validation benchmark.

**Exit condition:** The chosen retriever improves or meets the target
retrieval quality at acceptable latency.

### Milestone 4 --- Verification gate

-   Create a held-out calibration dataset.
-   Fit calibration without using the training set.
-   Evaluate calibration and selective risk.
-   Define sensor-coverage and unsupported-pattern policies.
-   Add fail-closed behavior and reason codes.
-   Evaluate coverage versus error rate.

**Exit condition:** Every gate decision is reproducible, tested, and
supported by validation results.

### Milestone 5 --- LangGraph and report generation

-   Build typed workflow state.
-   Implement retrieval and report nodes.
-   Add source-citation metadata.
-   Validate reports against a schema.
-   Add insufficient-evidence behavior.
-   Add human review and approval persistence.

**Exit condition:** The agent creates a traceable draft only for cases
that pass verification and evidence checks.

### Milestone 6 --- CI and end-to-end evaluation

-   Add unit and integration tests.
-   Add fixed regression datasets.
-   Add retrieval and report evaluation scripts.
-   Add CI workflow and evaluation summaries.
-   Test failure cases and approval invariants.

**Exit condition:** A failing mandatory check blocks the build; test
results are reproducible.

### Milestone 7 --- Demo and documentation

-   Add a Streamlit review UI if useful.
-   Demonstrate a passing case and at least three failure cases.
-   Document installation, ingestion, evaluation, and replay commands.
-   Record limitations and non-production status.
-   Capture latency, memory, and evaluation metrics.

**Exit condition:** Another developer can reproduce the demo and
understand the evidence and limitations behind each result.

------------------------------------------------------------------------

