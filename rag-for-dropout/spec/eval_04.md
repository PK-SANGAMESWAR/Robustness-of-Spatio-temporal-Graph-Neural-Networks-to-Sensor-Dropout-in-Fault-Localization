<!-- Source: rag-eval.md lines 322-417, verbatim. Do not edit. Index: 00_INDEX.md -->

## 9. Baseline-versus-candidate comparison

Make a command that compares a candidate snapshot with a baseline snapshot.
The result is one of three outcomes:

| Outcome | Condition |
|---|---|
| **FAIL** | A gate metric fails, or a metric does not meet its absolute floor, or the two snapshots are not comparable (different dataset version or test split). |
| **REVIEW** | A guardrail regresses more than its tolerance, or the confidence intervals overlap so the result is not clear, or a tolerance is still `provisional`. |
| **PASS** | All gates pass, all floors are met, and no guardrail regresses more than its tolerance. |

Rules:
1. Check the floors on the baseline also. If the baseline does not meet a floor, tell me. Do not accept the baseline as healthy.
2. Show the reason for each outcome, metric by metric.
3. Write the result as a table in Markdown and as JSON.

### 9.1 Configuration comparison

Make a command that runs two or more configurations on the same golden sets.
Show the trade-offs in one table: retrieval quality, generation quality, Layer 0 metrics, latency and cost.
Change one setting at a time when possible.

---

## 10. Tests and cost control

### 10.1 Inexpensive tests (run before any paid evaluation)

Write pytest tests for these items. Use mocked LLM and embedding components.

1. Metric calculations: Recall@k, MRR, nDCG, AUROC, AURC, ECE, selective risk, against values that you calculate by hand.
2. Golden-context matching: a passage that contains the span matches; a passage with a different chunk ID but the same span also matches.
3. The comparison logic: one test for PASS, REVIEW and FAIL each, and one test for a baseline that does not meet a floor.
4. Snapshot serialization: write, read, and get the same values.
5. Configuration consistency: one pipeline instance and one judge configuration for the full run.
6. Index invalidation: a change to the chunk size changes the fingerprint and stops the use of the old index.
7. Leakage: the test detects a test-split event in the case store.
8. Known failures: a mocked generator that gives a wrong location must fail field consistency; a mocked generator that obeys an injection must fail the injection test.
9. A metric exception is recorded as a failure, not as a score.

### 10.2 Cost estimate

Before the first paid run, calculate and show me:
- the number of judge calls and generation calls for each layer,
- the estimated input and output tokens for each call,
- the estimated cost for one full run, and for the repeat runs in section 8.3,
- the cost for one run with only the deterministic metrics (this must be 0).

Use prices that I give you. Do not use a price that you remember.
Add a cost limit to the configuration. Stop the run if the limit is reached.
Run a small pilot (about 5 items for each layer) before a full run, after my approval.

---

## 11. Documentation and experiment log

Write these files:

- `eval/README.md` — the architecture, the commands, the metric definitions, the golden-set rules and the known limitations.
- `eval/EXPERIMENT_LOG.md` — one entry for each stage and each run: date, label, commit, configuration change, result, cost, and the decision.

Known limitations to state:
- The data is simulated. The results do not show safety on a real grid.
- The corpus is curated for research. It is not approved by a utility.
- An LLM judge can be wrong. The judge metrics need human validation.

---

## 12. Stages

1. Inspect the repository (section 3).
2. Propose the plan: the architecture, the metrics for each layer, the golden-set design, the file changes, the regression method and the cost estimate. **Stop. Wait for my approval.**
3. Make the pipeline wrapper. Make Layer 0 and Layer 1.
4. Make Layer 2 and Layer 3.
5. Make Layers 4, 5 and 6.
6. Make the suite runner, the registry, the snapshots, the baseline management and the comparison.
7. Make the configuration comparison, the tests and the documentation.
8. Validate the framework. Report the actual results and the limitations.

Before each paid run, show the cost estimate and wait for my approval.

---

## 13. Your first response

Do stages 1 and 2 only. Give a short but complete plan with these items:

1. The existing pipeline: the components that exist, with file paths, and the components that do not exist.
2. The evaluation layers and the metrics that you will use for each layer, adapted to the code.
3. The golden-set design and the number of items.
4. The file changes and the new files.
5. The regression method and the initial metric registry.
6. The estimated API cost and compute time.
7. The risks and the questions for me.

Then stop. Do not change files. Wait for my approval.