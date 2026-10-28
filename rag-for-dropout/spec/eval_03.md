<!-- Source: rag-eval.md lines 178-321, verbatim. Do not edit. Index: 00_INDEX.md -->

## 5. Golden datasets

### 5.1 General rules

1. Keep each golden set small and reviewed by a person.
2. A generated example is a draft. Mark it `status: draft`. Do not use a draft in a reported result.
3. I review each draft. After my review, the status changes to `status: reviewed`.
4. Give each golden set a version number. Record the version in each snapshot.
5. Do not identify a golden context by a chunk ID only. Chunk IDs change when the chunk size changes.
   Identify a golden context by these items:
   - `document_id`,
   - section path,
   - a short text span from the source,
   - the content hash of the normalized span.
   A retrieved passage matches when it contains the span (normalized whitespace and case).
6. Do not put events from the test split in the case store. Assert this in code.

### 5.2 Retrieval golden set (`data/eval/golden_retrieval.jsonl`)

About 40 to 60 queries. Include these types:
- paraphrased operator questions,
- questions with exact identifiers (section, bus, device ID),
- questions about sensor dropout and observability,
- questions with no answer in the corpus (about 15 %).

Each item has: `query_id`, `query`, `golden_contexts` (as in 5.1), `reference_answer`, `answerable`, `critical_terms`, `split` (`dev` or `test`), `status`.

### 5.3 Event golden set (`data/eval/golden_events.jsonl`)

About 60 to 100 events from the test split. Include these types:
- correct `CONFIDENT` predictions,
- incorrect predictions with high probability (the most important type),
- `LOW_CONFIDENCE` events and ambiguity sets,
- each dropout level,
- each fault type.

Each item has: `event_id`, `true_location`, `dropout_rate`, `fault_type`, `golden_passages`, `reference_report_points` (the facts that a correct report must contain), `split`, `status`.

### 5.4 Safety set (`data/eval/golden_safety.jsonl`)

About 20 to 30 items: counterfactual swaps, evidence flips, empty evidence, injection chunks, out-of-scope questions and control-command traps.

### 5.5 Human labels for judge validation

Label about 50 reports by hand for faithfulness and correctness.
Use these labels to measure the agreement between the judge and a person (Cohen's κ and percent agreement).
If the agreement is low, report the human labels and mark the judge metric as informational only.

---

## 6. Judge and generation configuration

### 6.1 Judge

1. Use one judge model and one judge configuration for the full suite in a snapshot.
2. Use a judge model that is different from the generation model, if possible.
3. Set the judge temperature to 0. Record the exact model snapshot name.
4. Record the DeepEval version and the GEval rubric text in the snapshot.
5. Cache each judge response with a key from: the judge model, the rubric hash, the input hash. Do not pay twice for the same judgment.

### 6.2 Generation

1. Put the generation model behind one interface, so that a local model (Ollama) and an API model (OpenAI) can be used with the same code.
2. Record the exact model name and the prompt hash.
3. Develop and run the inexpensive tests with the local model or with a mocked model.

### 6.3 GEval rubrics

Write each rubric in a separate file in `eval/rubrics/`. Each rubric must have:
- the criteria, as a numbered list,
- the score scale and the meaning of each score,
- one good example and one bad example.

---

## 7. Configuration

Put all evaluation settings in one file (for example, `configs/eval.yaml`), with these items:

- embedding model, revision, input-length limit, prefixes,
- chunk size, overlap, chunking strategy,
- retrieval mode, `top_k`, candidate counts, reranker,
- case store: latent layer, k, model version,
- gate thresholds and policy version,
- generation model, temperature, prompt version,
- judge model, temperature, metric thresholds,
- dataset versions,
- number of repeat runs,
- the cost limit for one run.

**Index invalidation.** Make a fingerprint (hash) from the embedding model, the revision, the prefixes, the chunk size, the overlap, the chunking strategy, the parser version and the corpus version.
Store the fingerprint with the index.
If the fingerprint is different, rebuild the index or stop with an error. Do not use a stale index.
Do the same for the case store with the STGNN checkpoint and the latent layer.

---

## 8. Snapshots and metric registry

### 8.1 Snapshot

Each run makes one snapshot file (JSON) in `eval/snapshots/`. The snapshot contains:

- the Git commit and a flag that shows uncommitted changes,
- the experiment label,
- the timestamp (UTC),
- the effective configuration (after defaults),
- the index fingerprint and the case-store fingerprint,
- the prompt hashes and the rubric hashes,
- the dataset versions,
- the judge model and the generation model,
- the flattened metric results (for example, `layer3.faithfulness.mean`),
- the number of items, the number of failures, and the failure causes,
- the token usage and the cost,
- the per-item records, or a path to them.

### 8.2 Metric registry (`eval/registry.yaml`)

For each metric, write these items:

| Field | Meaning |
|---|---|
| `name` | Flattened name, for example `layer0.selective_risk` |
| `direction` | `higher_is_better` or `lower_is_better` |
| `role` | `gate`, `guardrail` or `informational` |
| `floor` | The absolute minimum (or maximum) value. A baseline that does not meet the floor is not healthy. |
| `tolerance` | The largest regression that is acceptable |
| `tolerance_source` | `measured` (from repeat runs) or `provisional` |

Suggested roles (I approve the final list):
- **Gate:** field consistency = 100 %, schema validity = 100 %, citation validity = 100 %, prompt injection pass = 100 %, no control commands = 100 %, no leakage assertion = pass, selective risk on `CONFIDENT` events, error recall.
- **Guardrail:** faithfulness, answer relevancy, contextual precision, contextual recall, Recall@k, ECE, p95 latency, cost per event.
- **Informational:** toxicity, MRR, nDCG, mean candidate-set size, token counts.

### 8.3 Tolerance from measured noise

1. Run the suite 3 to 5 times on the same commit and configuration.
2. For each metric, calculate the standard deviation over the runs.
3. Set the tolerance to about 2 × the standard deviation, or to a larger value that I approve.
4. Until you measure the noise, mark the tolerance as `provisional`.
5. Compare metric means, not only pass rates. Report bootstrap confidence intervals over the items.

---

