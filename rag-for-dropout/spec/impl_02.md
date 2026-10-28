<!-- Source: implementation-plan.md lines 128-283, verbatim. Do not edit. Index: 00_INDEX.md -->

## 2. Goals, non-goals, and success criteria

### 2.1 Goals

-   Accept versioned grid measurements and produce a structured fault
    prediction.
-   Preserve the sensor-availability mask and relevant provenance
    throughout the pipeline.
-   Reject invalid or insufficiently supported predictions before they
    reach the report-generation agent.
-   Retrieve approved documents with stable source and page/section
    references.
-   Generate a report whose factual claims can be traced to model output
    or retrieved evidence.
-   Require explicit human approval before any operational
    recommendation is treated as approved.
-   Evaluate the full pipeline reproducibly.
-   Run on a development machine with an NVIDIA T1000 8 GB GPU, while
    allowing CPU fallback for smaller components.

### 2.2 Non-goals for the first version

-   Autonomous switching, breaker control, or direct control of grid
    infrastructure.
-   Treating LLM-generated reasoning as proof that a prediction is
    correct.
-   Training a new embedding model from scratch.
-   Building a distributed production platform before single-machine
    behavior is validated.
-   Claiming real-world safety from simulation-only experiments.
-   Using the STGNN's aggregate macro-F1 as an individual prediction's
    confidence.

### 2.3 Proposed acceptance criteria

These are **engineering gates to define and validate**, not claims about
current performance:

-   Every prediction has a schema-validated output and model version.
-   Every gate produces a machine-readable result, reason codes, and a
    traceable decision.
-   Every report claim that relies on a manual includes a source
    citation.
-   A failed verification gate cannot be bypassed by the LLM.
-   A missing source, invalid input, or low-confidence case follows a
    defined fallback route.
-   Retrieval is evaluated against a labelled query-document set before
    choosing a final model or chunking configuration.
-   CI fails when required unit tests, regression thresholds, or safety
    invariants fail.
-   No physical action is executed by the prototype.

------------------------------------------------------------------------

## 3. Proposed architecture

``` text
Grid telemetry / replayed dataset
              |
              v
      Input validation layer
              |
              v
    Existing STGNN inference API
              |
              v
    Structured prediction object
              |
              v
  Deterministic verification service
  - schema and data-quality checks
  - calibrated confidence
  - sensor coverage / missingness checks
  - policy and out-of-distribution checks
  - [Rev 2] LOO physics residual (silent sensor failure)
       /               |                     \
   CONFIDENT      LOW CONFIDENCE         INVALID / CHECK FAILED
   (single         (top-k / conformal         |
    location)       candidate set)            v
       \               /               Manual-review queue
        v             v                (no report drafted)
     LangGraph workflow
        |
        v
  Query construction from structured event
        |
        +--------------------------------+
        v                                v
  [Rev 2] Case retrieval:          Document retrieval:
  latent kNN over past             dense + lexical/hybrid
  simulated faults (§8.6)          over curated notes
        |                                |
        v                                |
  Deterministic evidence score           |
  (kNN agreement, margin,                |
   physics residual)                     |
        +---------------+----------------+
                        |
        v
  Optional reranking
        |
        v
  Evidence validation and context assembly
        |
        v
  Report drafting with source references
        |
        v
  Output validation / policy checks
        |
        v
  Human review and explicit approval
        |
        v
  Export approved report (no equipment control)

CI evaluates each layer and the end-to-end workflow.
```

### Component responsibilities

  -----------------------------------------------------------------------
  Component               Owns                    Must not own
  ----------------------- ----------------------- -----------------------
  STGNN service           Fault-location          Operational
                          prediction and model    authorization
                          metadata                

  Verification service    Deterministic gate      Free-form LLM judgment
                          decisions               

  Document ingestion      Parsing, chunking,      Editing source manuals
                          metadata, embeddings    

  Retriever               Candidate evidence      Claiming retrieved text
                          retrieval               is automatically
                                                  correct

  Reranker                Reordering retrieved    Creating new evidence
                          candidates              

  LangGraph               Workflow routing and    Bypassing failed gates
                          state transitions       

  Report generator        Drafting a sourced      Executing grid controls
                          report                  

  Human reviewer          Reviewing and approving Unlogged approval
                          the draft               

  CI/evaluation           Reproducible quality    Replacing engineering
                          checks                  validation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

