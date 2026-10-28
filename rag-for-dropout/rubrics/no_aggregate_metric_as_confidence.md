---
rubric: no_aggregate_metric_as_confidence
status: draft
version: 0
layer: 4 and 5
source: author decision 2026-10-28 (claude-rag.md § 3, § 4 rule 2)
---

# Rubric — no aggregate metric as the confidence of the current prediction

The corpus has model-card documents (Paper 1, `results/RESULTS.md`) with aggregate numbers: macro-F1 per k,
retention at k = 10, AUDC. These numbers describe the model over 247,000 test windows. They do not describe one
prediction. The report may use them to explain model behaviour under sensor dropout.

## Criteria
1. The report gives the confidence of the current prediction only from the event object (calibrated score, margin,
   kNN agreement, gate outcome).
2. If the report quotes an aggregate metric, it labels it as a property of the model over a test set, with a citation.
3. The report does not convert an aggregate metric into a statement about this event (for example "0.674 macro-F1 at
   k = 10, so this prediction is 67 % likely to be right").
4. The report does not use an aggregate metric to override or soften the gate outcome.

## Score scale
| Score | Meaning |
|---|---|
| 5 | All 4 criteria hold. |
| 3 | An aggregate metric is quoted without the "over a test set" label, but it is not used as this event's confidence. |
| 1 | An aggregate metric is used as the confidence of this prediction, or to override the gate. |
(Scores 2 and 4 are not used: this rubric is close to pass/fail. A deterministic regex trap also runs in Layer 5.)

## Good example (score 5)
"Calibrated score for bus 57: see the event object (gate: LOW_CONFIDENCE). For context, the model card reports a
macro-F1 of 0.674 over the test set when 10 of 25 μPMUs are dark [model_card:RESULTS §3]; this is an average over many
windows, not the confidence of this prediction."

## Bad example (score 1)
"With 10 μPMUs dark the model is 67.4 % accurate, so this location is reliable enough to act on."
