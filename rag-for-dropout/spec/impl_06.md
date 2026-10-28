<!-- Source: implementation-plan.md lines 674-840, verbatim. Do not edit. Index: 00_INDEX.md -->

## 8. Dense, lexical, and hybrid retrieval

### 8.1 Dense semantic retrieval

Dense retrieval embeds the query and searches for nearby document
vectors. It is useful when the question uses different wording from the
manual.

Example: - Query: "What should happen if measurements disappear during a
fault?" - Manual wording: "When a measurement unit becomes unavailable,
apply the prescribed fallback procedure."

The wording differs, but the intended meaning may be similar.

### 8.2 Lexical/BM25 retrieval

Lexical retrieval is important for exact identifiers: - `BUS-45` -
`SOP-7.2` - relay identifiers - equipment IDs - standard names and
section numbers - exact error codes

A dense model may find semantically similar content but miss an exact
identifier. Lexical retrieval compensates for this.

### 8.3 Hybrid retrieval

Run dense and lexical/sparse retrieval, then merge candidate results.
Reciprocal Rank Fusion (RRF) is a sensible first fusion method because
it combines ranks without requiring the dense and lexical scores to
share a scale.

Recommended first retrieval flow:

1.  Validate and normalize the query without deleting technical
    identifiers.
2.  Apply access and approval filters.
3.  Retrieve the top 20--50 candidates from dense search.
4.  Retrieve the top 20--50 candidates from lexical/BM25 search.
5.  Merge candidates with RRF.
6.  Deduplicate overlapping chunks.
7.  Optionally rerank the top 10--20 candidates.
8.  Return a small set of high-quality passages with source metadata.

The values above are starting points for experimentation, not fixed
requirements. Tune candidate counts on the evaluation set.

### 8.4 Query construction

Construct the retrieval query from structured fields rather than sending
only an unconstrained natural-language summary. Example:

``` text
Event type: suspected feeder fault
Predicted location: BUS-45
Relevant condition: several measurements unavailable
Task: find approved procedures covering verification, escalation,
and operator review for this situation.
```

Do not let an LLM invent the location, event type, or sensor status.
Populate those fields from the validated event object.

### 8.5 Metadata filtering

Apply filters before or during retrieval where supported: - Approved
document status. - Relevant document type. - Effective revision. -
Applicable grid/feeder context, if represented in the metadata. -
Language or jurisdiction, if needed.

Never rely on the embedding model to infer whether a document is current
or approved. These are metadata/policy checks.

### 8.6 [Rev 2] Case-based retrieval over STGNN latent embeddings

Document retrieval explains *what a fault type means and what to do*. It
cannot say whether *this* prediction is right. For that, retrieve past
simulated faults that the STGNN itself represents as similar.

**Embedding.** Use the STGNN's penultimate representation (the vector
fed to the classification head), pooled to one vector per event. Do not
use a text embedding of the event: the model's latent space is what
determines its prediction, so similarity there is the relevant
similarity. L2-normalize and use cosine distance.

**What to store per case:**

``` json
{
  "case_id": "stable-id",
  "split": "train",
  "latent": [0.01, -0.12, "..."],
  "true_location": "section id",
  "predicted_location": "section id",
  "pred_correct": true,
  "fault_type": "LG",
  "fault_resistance_ohm": 1.0,
  "load_multiplier": 1.0,
  "dropout_rate": 0.2,
  "sensor_mask_hash": "sha256",
  "model_version": "exact checkpoint",
  "latent_layer": "name of layer"
}
```

Store the model's *wrong* predictions as well, tagged `pred_correct =
false`. A query whose neighbours are mostly past errors is a direct
warning.

**Which split goes in the store.** Training split only. If calibration
or validation cases are also stored, evaluate them leave-one-out (a case
must never retrieve itself). Test-split cases are **never** stored; see
§13.9.

**Evidence signals (computed in Python, not by the LLM):**

-   *kNN agreement*: fraction of the k nearest cases whose true location
    equals the current prediction (optionally distance-weighted).
-   *kNN vote*: the location most common among neighbours, and whether
    it matches the STGNN's top-1.
-   *Neighbour error rate*: fraction of neighbours with `pred_correct =
    false`.
-   *Distance to nearest case*: a simple out-of-distribution indicator.

These signals feed the deterministic evidence score and the gate
(§11). The neighbours themselves (location, type, dropout level) go
into the report as concrete precedents: "8 of the 10 most similar
simulated faults were at section S12."

**Re-embedding rule.** The latent space changes every time the STGNN is
retrained. Rebuild the case store for each checkpoint and record
`model_version` and `latent_layer` in the collection manifest. A query
latent from checkpoint B must never be searched against a store built
from checkpoint A.

**Store choice.** Use a separate collection, e.g.
`gridsense_cases_<model_version>`, in the same vector database. A FAISS
flat index is also enough at this scale, since exact search over tens of
thousands of vectors is fast on CPU.

------------------------------------------------------------------------

## 9. Reranking: when and how

A reranker is a second-stage model that scores a query against each
candidate passage jointly. Unlike a bi-encoder, which embeds query and
document separately, a cross-encoder can consider their interaction in
more detail. It is slower, so it is normally applied only to the
candidate shortlist.

### Recommended approach

1.  Establish dense-only retrieval as the baseline.
2.  Establish hybrid retrieval.
3.  Evaluate the top results.
4.  Add a small cross-encoder reranker if the ranking metrics or manual
    inspection show a meaningful weakness.
5.  Measure the quality improvement and added latency.
6.  Keep the reranker only if the improvement justifies its cost.

Reranking can improve relevance but does not create evidence and does
not validate a source. A highly ranked passage can still be irrelevant,
outdated, or inapplicable.

Use a separate evaluation set to avoid tuning the reranker on the same
examples used for final reporting.

------------------------------------------------------------------------

