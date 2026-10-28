<!-- Source: implementation-plan.md lines 1337-1511, verbatim. Do not edit. Index: 00_INDEX.md -->

## 14. CI and regression testing

### 14.1 Unit tests

Test: - Schema validation. - Probability conversion and calibration
application. - Sensor-mask calculations. - Each verification reason
code. - Source metadata propagation. - Chunk IDs and idempotent
ingestion. - Citation existence and excerpt validation. - Report schema
validation. - Approval-state transitions.

### 14.2 Integration tests

Test the full path with fixed fixtures: - Valid prediction passes to
retrieval. - Low-confidence prediction goes to manual review. - Critical
sensor missing goes to manual review. - No relevant evidence leads to
abstention/review. - Invalid citation causes report validation to
fail. - Human rejection prevents approval/export. - An LLM response
cannot alter verification state. - A transient retrieval failure is
retried within limits. - Replaying the same event does not create
duplicate reports. - **[Rev 2]** A low-confidence event produces a
candidate-set report, never a single-location one. - **[Rev 2]** No
test-split case ID is present in the case store. - **[Rev 2]** A case
store built from a different `model_version` is refused at query
time. - **[Rev 2]** Report fields equal state-object fields (location,
probability, gate outcome). - **[Rev 2]** The counterfactual-swap and
injection fixtures (§13.10) pass.

### 14.3 Regression tests

Keep a small deterministic set of events and retrieval queries. Track: -
STGNN metrics. - Gate decisions. - Retrieval metrics. - Citation
validation outcomes. - Latency and resource use.

Set thresholds after obtaining a baseline. Avoid brittle tests that
require exact free-form LLM wording; test schema, claims, sources, and
required invariants instead.

### 14.4 Example CI stages

``` text
1. Lint / formatting / type checks
2. Unit tests
3. Ingestion and metadata tests
4. Verification policy tests
5. Retrieval evaluation on a small fixed benchmark
6. Report schema and citation tests
7. End-to-end workflow tests with mocked LLM/retriever
8. Optional scheduled full benchmark
9. Build artifact and publish evaluation summary
```

Use mocked LLM and embedding components for most unit tests. Run a
smaller number of real-model tests separately because they are slower
and can be nondeterministic.

------------------------------------------------------------------------

## 15. Suggested repository structure

Adapt this structure to the existing repository rather than moving all
current files at once.

``` text
gridsense/
├── README.md
├── PLAN.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── retrieval.yaml
│   ├── chunking.yaml
│   ├── verification_policy.yaml
│   └── evaluation.yaml
├── src/
│   └── gridsense/
│       ├── schemas/
│       │   ├── event.py
│       │   ├── prediction.py
│       │   └── report.py
│       ├── model/
│       │   ├── predictor.py
│       │   └── calibration.py
│       ├── verification/
│       │   ├── gates.py
│       │   ├── sensor_coverage.py
│       │   └── policy.py
│       ├── ingestion/
│       │   ├── parsers.py
│       │   ├── chunking.py
│       │   ├── metadata.py
│       │   └── index_builder.py
│       ├── retrieval/
│       │   ├── embeddings.py
│       │   ├── dense.py
│       │   ├── lexical.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       ├── agent/
│       │   ├── graph.py
│       │   ├── state.py
│       │   ├── prompts.py
│       │   └── report_validation.py
│       ├── review/
│       │   ├── approval.py
│       │   └── audit.py
│       └── api/
│           └── app.py
├── data/
│   ├── source_manifest.example.json
│   └── eval/
│       ├── retrieval_queries.jsonl
│       └── expected_gate_cases.jsonl
├── scripts/
│   ├── ingest_documents.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_gates.py
│   └── replay_event.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── .github/
    └── workflows/
        └── ci.yml
```

Keep raw approved documents, credentials, private telemetry, and
generated production data out of Git unless the repository's data policy
explicitly permits them.

------------------------------------------------------------------------

## 16. API boundaries

Even if the first version is a single Python process, keep interfaces
clear.

### Prediction interface

Input: - Event ID. - Time window or reference to a validated measurement
window. - Sensor availability mask. - Model version.

Output: - Predicted location/class. - Raw logits or probabilities. -
Calibration output, if applicable. - Model version and inference
metadata.

### Verification interface

Input: - Prediction object. - Sensor mask and measurement quality. -
Versioned verification policy.

Output: - Pass/fail for downstream decision support. - Individual check
results. - Reason codes. - Policy version. - Diagnostic metadata.

### Retrieval interface

Input: - Structured query. - Approved-source filters. - Retrieval
configuration version.

Output: - Chunk IDs and text. - Document and section metadata. -
Retrieval/reranking scores. - Index version.

### Report interface

Input: - Verified event. - Retrieved evidence. - Report schema and
prompt version.

Output: - Structured draft. - Evidence/citations. - Validation
results. - Pending human approval status.

These interfaces make it possible to test each component without
requiring the entire stack to be running.

------------------------------------------------------------------------

