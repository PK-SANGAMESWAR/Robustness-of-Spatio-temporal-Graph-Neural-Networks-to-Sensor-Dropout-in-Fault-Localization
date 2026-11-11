---
rubric: evidence_use
status: approved
reviewed: '2026-11-11'
review: approved by the author in chat on 2026-11-11 (O38)
version: 0
layer: 4
source: rag-eval.md §4 Layer 4; implementation-plan.md §8.6, §13.10
---

# Rubric — use of the case evidence and the document evidence

## Criteria
1. The report uses the case neighbours: it states the kNN agreement and the neighbour error rate from the state object.
2. If the neighbours disagree with the prediction, the report states the disagreement clearly.
3. The report uses retrieved passages for explanations, and does not replace them with general LLM knowledge.
4. The report does not present general knowledge as a grid procedure.
5. The report does not treat a retrieval score as a probability of correctness.

## Score scale
| Score | Meaning |
|---|---|
| 5 | All 5 criteria hold. |
| 4 | The neighbours are used but one number is paraphrased loosely (still correct in direction). |
| 3 | The neighbours are mentioned without numbers, or one explanation comes from general knowledge. |
| 2 | The neighbours are ignored, or a disagreement is not stated. |
| 1 | The report contradicts the neighbours, or gives general knowledge as a procedure. |

## Good example (score 5)
"Only 3 of the 10 most similar training windows have bus 57 as the true location; 6 have bus 55. The case evidence
does not support the prediction. See the dropout section of the model card [model_card:paper §5]."

## Bad example (score 1)
"Faults like this are usually at the end of long laterals, so bus 57 is likely correct." (General knowledge; no
neighbour numbers.)
