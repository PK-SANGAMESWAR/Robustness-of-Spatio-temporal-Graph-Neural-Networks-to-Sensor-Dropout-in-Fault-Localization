<!-- Source: implementation-plan.md lines 841-1014, verbatim. Do not edit. Index: 00_INDEX.md -->

## 10. RAG generation and source-grounded reporting

### 10.1 Context assembly

After retrieval/reranking: - Remove duplicates. - Preserve section
ordering when it matters. - Include source title, revision, page, and
section. - Limit context to passages relevant to the event. - Preserve
warnings and prerequisites with the steps they qualify. - Record
retrieval scores for diagnostics, but do not present them as
probabilities of correctness.

### 10.2 Prompt design

The report-generation prompt should instruct the model to: - Use only
the supplied event fields for the prediction and sensor status. - Use
retrieved approved documents for procedure-related claims. - Cite each
material procedural claim. - Clearly distinguish model prediction from
confirmed fault. - State when evidence is insufficient or conflicting. -
Never invent missing manual sections, document revisions, measurements,
or approval status. - Never issue a direct control command. - Produce a
structured report that can be schema-validated.

The prompt is one control, not the entire safety system. Enforce
critical rules in Python after generation as well.

### 10.3 Suggested report schema

``` json
{
  "event_id": "event-identifier",
  "predicted_location": "BUS-45",
  "prediction_status": "candidate_not_confirmed",
  "verification_status": "passed",
  "verification_reasons": [],
  "sensor_coverage_summary": "summary derived from telemetry",
  "evidence": [
    {
      "claim": "Brief claim supported by a source",
      "document_id": "grid-sop-001",
      "revision": "revision-if-known",
      "section": "section-if-known",
      "page": 12,
      "quote_or_excerpt": "Short supporting excerpt"
    }
  ],
  "recommended_operator_review": [
    "Review the cited approved procedure.",
    "Confirm the relevant conditions using authorized processes."
  ],
  "limitations": [],
  "approval_status": "pending_human_review"
}
```

Validate output using Pydantic or an equivalent schema. Reject or
regenerate malformed output; never silently accept invalid fields.

### 10.4 Citation validation

For every evidence item: - Verify that `document_id` exists. - Verify
the cited revision matches the retrieved record. - Verify the
section/page is present when available. - Verify that the quoted excerpt
is a substring or traceable span from the source text. - Ensure the
source was approved at retrieval time. - Record the exact retrieved
chunk IDs used for the report.

A substring check does not prove the claim is logically supported. Add
human review and a separate entailment/faithfulness evaluation for
important claims.

### 10.5 Handling insufficient evidence

If retrieval produces no adequate passage: - Do not fabricate a
procedure. - Mark the evidence status as insufficient. - Ask for human
review. - Include the reason and retrieval diagnostics in the trace. -
Do not present generic LLM knowledge as an approved grid procedure.

------------------------------------------------------------------------

## 11. Deterministic verification layer

Keep this component separate from the LLM and the retrieval system.

### 11.1 Checks to implement

**Input validity** - Expected fields are present. - Values are finite
and within expected ranges. - Timestamp/order and sample-window
requirements are met. - The sensor mask has the expected shape and valid
values. - The model and preprocessing versions are known.

**Prediction reliability** - Convert logits to probabilities
correctly. - Fit calibration on data independent of model training. -
Evaluate calibration by fault class and dropout condition where data
allows. - Define an acceptance policy on a validation set. - Record
confidence, calibration method, and policy version.

**Sensor coverage** - Calculate overall availability. - Check critical
sensors, phases, or regions defined by engineering policy. - Evaluate
whether the observed missingness pattern is represented in validation. -
Consider spatial distribution of missing sensors, not only the total
count. - Route unknown or unsupported patterns to review.

**[Rev 2] Silent sensor failure** - The availability mask only catches
sensors that are *missing*. A sensor that reports a wrong value with m
= 1 passes every mask-based check. - When the Stage F leave-one-out
physics residual is available, add it as a gate input: flag events
where any present sensor's residual exceeds the threshold validated in
Stage F (S5.1). - Report this check's detection AUROC per corruption
type; if it does not meet the Stage F bar, do not use it as a gate and
say so in the limitations.

**[Rev 2] Ambiguity set** - If the model outputs an ambiguity set
(candidates indistinguishable under the current sensor mask), a
single-location report is not allowed for that event: route it to the
low-confidence branch with the ambiguity set as the candidate set.

**Out-of-distribution and data quality** - Detect obviously malformed
inputs. - Flag conditions outside the tested data range. - Record
distribution-shift indicators if implemented. - Do not claim that a
generic OOD detector guarantees detection of every unfamiliar condition.

### 11.2 Confidence calibration

A common method is temperature scaling, which applies a learned
temperature to logits before softmax. Fit it on a held-out calibration
dataset. Do not fit it on the training set and report the resulting
confidence as unbiased.

Evaluate: - Reliability diagrams. - Expected calibration error (ECE),
with the binning scheme documented. - Brier score or log loss where
appropriate. - Calibration by sensor-dropout rate and important fault
classes. - Selective risk as a function of coverage.

If the model is not well calibrated, the gate must not pretend its
probability threshold has a trustworthy interpretation.

### 11.3 Gate policy

The gate should return structured reason codes, for example: -
`INVALID_INPUT` - `MODEL_VERSION_UNKNOWN` -
`LOW_CALIBRATED_CONFIDENCE` - `CRITICAL_SENSOR_MISSING` -
`UNSUPPORTED_DROPOUT_PATTERN` - `OUT_OF_DISTRIBUTION_WARNING` -
`VERIFICATION_PASSED`

Use explicit policy configuration with version control. The threshold
and sensor rules should be determined through evaluation and engineering
review, not embedded as undocumented magic numbers.

**[Rev 2] Three outcomes, not two.** The gate returns one of:

-   `CONFIDENT`: calibrated probability ≥ τ_p **and** kNN agreement ≥
    τ_k **and** no failed check. A single-location report is drafted.
-   `LOW_CONFIDENCE`: input is valid but the confidence condition fails,
    or an ambiguity set is present. An "uncertain" report is drafted
    with a candidate set (top-k, or a conformal set at a stated coverage
    such as 90%, fitted on the calibration split).
-   `REJECT`: invalid input, unknown version, unsupported dropout
    pattern, or a failed silent-failure check. No report; manual
    review.

τ_p and τ_k are chosen on the calibration split from the
accuracy-vs-coverage curve to meet a stated target (for example ≥ 95%
accuracy on `CONFIDENT` cases), never fixed in advance. Check whether
one threshold holds across dropout levels; if not, use a
dropout-dependent threshold and report it.

### 11.4 Fail-closed behavior

When required checks cannot be completed---for example, missing
calibration metadata or an unknown policy version---route to manual
review. Do not treat a failed or unavailable check as a pass.

------------------------------------------------------------------------

