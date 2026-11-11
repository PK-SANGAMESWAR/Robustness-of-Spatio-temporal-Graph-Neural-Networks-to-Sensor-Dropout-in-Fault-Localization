---
rubric: completeness
status: approved
reviewed: '2026-11-11'
review: approved by the author in chat on 2026-11-11 (O38)
version: 0
layer: 4
source: rag-eval.md §4 Layer 4; implementation-plan.md §10.3 report schema
---

# Rubric — completeness of the report

## Criteria
The report contains each required section with content that comes from the event object or the evidence:
1. Event ID and the predicted location (or the candidate set for LOW_CONFIDENCE).
2. Gate outcome and the reason codes.
3. Sensor coverage: how many μPMUs are dark (k) and which ones.
4. Case evidence: the neighbour summary (for example "7 of 10 nearest training windows share the predicted bus").
5. Document evidence with citations (`document_id`, section).
6. Limitations, including "the corpus is research-curated, not utility-approved" and "the data is simulated".
7. `approval_status: pending_human_review`.

## Score scale
| Score | Meaning |
|---|---|
| 5 | All 7 sections are present and correct. |
| 4 | One section is missing or empty, and it is not section 1, 2 or 7. |
| 3 | Two sections are missing, or section 3 or 4 is wrong. |
| 2 | Three or more sections are missing. |
| 1 | Section 1, 2 or 7 is missing. |

## Good example (score 5)
A report with all 7 sections, each filled from the state object, two citations, and the two limitation sentences.

## Bad example (score 1)
A report that describes the fault and the procedure but has no gate outcome and no approval status.
