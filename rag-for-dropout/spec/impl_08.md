<!-- Source: implementation-plan.md lines 1015-1192, verbatim. Do not edit. Index: 00_INDEX.md -->

## 12. LangGraph workflow design

LangGraph should orchestrate the process; it should not replace
deterministic validation.

### 12.1 Suggested graph states

``` text
START
  |
  v
validate_event
  | invalid
  +----------------------> manual_review
  |
  v
run_or_load_prediction
  |
  v
verify_prediction                      [Rev 2: three outcomes]
  | REJECT
  +----------------------> manual_review
  |
  | CONFIDENT  -> report_mode = "single_location"
  | LOW_CONF   -> report_mode = "candidate_set"
  v
build_retrieval_query
  |
  v
retrieve_cases (latent kNN, §8.6)      [Rev 2]
  |
  v
compute_evidence_score (Python)        [Rev 2]
  |
  v
retrieve_evidence (documents)
  |
  v
rerank_if_enabled
  |
  v
check_evidence_sufficiency
  | insufficient
  +----------------------> manual_review
  |
  v
draft_report
  |
  v
validate_report_and_citations
  | fail
  +----------------------> revise_once_or_manual_review
  |
  v
human_approval
  |
  +---- reject ----------> closed_with_feedback
  |
  +---- approve ---------> export_approved_report
```

### 12.2 State fields

Use a typed state object containing: - `event_id` - `input_reference` -
`prediction` - `verification_result` - `retrieval_query` -
`retrieved_chunks` - `reranked_chunks` - `evidence_sufficiency` -
`draft_report` - `citation_validation` - `approval_status` -
`trace_id` - `errors` - `model_version`, `policy_version`, and
`index_version`

Avoid placing unnecessary raw telemetry or sensitive information in
long-lived workflow state. Store large artifacts separately and keep
references.

### 12.3 Retries and idempotency

-   Retry transient retrieval/service errors a limited number of times.
-   Do not retry a deterministic failed verification as though it were a
    network error.
-   Give each event a stable ID.
-   Make report export idempotent to prevent duplicates.
-   Persist graph state when human review may happen later.
-   Log human decisions and their timestamps.
-   Do not resume a paused workflow using an obsolete policy without
    checking policy version.

### 12.4 Human-in-the-loop

The operator must be able to: - View the original event and
prediction. - See why the verification gate passed or failed. - Open the
exact cited manual pages/sections. - Approve, reject, or request
clarification. - Add a review note. - See whether the report is a draft
or approved artifact.

Approval should be an explicit authenticated action recorded in an audit
log. A model-generated string such as `"approval_status": "approved"`
must never count as a human approval.

------------------------------------------------------------------------

## 13. Evaluation plan

Do not evaluate only the final generated report. Evaluate each layer
separately and then end-to-end.

### 13.1 Build a labelled retrieval benchmark

Create a small, reviewed set of operator-style queries. Each query
should be linked to: - Relevant document IDs. - Relevant page/section or
chunk IDs. - Whether the corpus contains a sufficient answer. - Critical
terms or identifiers that must be preserved. - Whether an incorrect
passage could be misleading.

Use realistic paraphrases, exact IDs, abbreviations, missing-sensor
cases, and unanswerable questions. Split development and test queries so
retrieval tuning does not leak into the final test.

### 13.2 Retrieval metrics

-   **Recall@k:** Does at least one relevant passage appear in the top
    k?
-   **Precision@k:** How many of the top k results are relevant?
-   **MRR:** How early does the first relevant result appear?
-   **nDCG@k:** Does the ranking place more relevant passages higher,
    including graded relevance?
-   **Source coverage:** Are the necessary approved sources present?
-   **Latency:** How long does retrieval take?
-   **Unanswerable-query handling:** Does the system abstain instead of
    supplying unrelated text?

Use human-labelled relevance. Similarity scores alone are not evaluation
labels.

### 13.3 Compare retrieval variants

Run the same benchmark for: 1. Dense retrieval only. 2. Lexical/BM25
only. 3. Hybrid retrieval with RRF. 4. Hybrid retrieval plus reranking.
5. Alternative embedding model(s). 6. Alternative chunk
sizes/strategies.

Change one factor at a time where practical. Keep the test set fixed and
report the model, corpus, chunking, and index versions.

### 13.4 Model and gate metrics

Evaluate the STGNN across: - Sensor-dropout levels. - Different
missing-sensor patterns. - Relevant fault classes and locations. - Noise
and other conditions present in the experimental design.

Report: - Macro-F1 and localization accuracy. - Calibration metrics. -
Coverage (fraction of cases allowed through the gate). - Selective risk
(error among accepted cases). - False acceptance rate. - False rejection
rate. - Results by dropout condition.

A gate that rejects every event has low accepted-case error but is not
useful. Always report risk together with coverage.

### 13.5 Report quality

Create a small manually reviewed set of generated reports. Score: -
Citation existence and correctness. - Whether claims are supported by
cited passages. - Correct distinction between predicted and confirmed
fault. - No fabricated document/page/revision. - Completeness of
required fields. - Appropriate abstention when evidence is
insufficient. - No unauthorized instruction or claim of completed
physical action.

Automated LLM-as-judge scoring can help triage cases, but it should not
be the sole evidence of safety or factual correctness.

### 13.6 Statistical reporting

Use a fixed evaluation protocol. Where comparisons are made, report the
number of test queries/events and confidence intervals where
appropriate. Do not claim a statistically meaningful improvement from a
tiny hand-picked set. Keep the final test set untouched until
configuration selection is complete.

