---
document_id: note_30_aggregate_versus_individual
title: Aggregate metrics are not the confidence of one prediction
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: results/RESULTS.md (this project)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_30_aggregate_versus_individual
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Aggregate metrics are not the confidence of one prediction

Macro-F1, retention and the area under the degradation curve describe a model over a whole test set. They must not be quoted as the confidence of the current prediction. For example, a macro-F1 of 0.674 with 10 sensors dark does not mean that a prediction made with 10 sensors dark is 67.4 percent likely to be right. The confidence of one prediction comes from the gate: the calibrated probability, the margin and the neighbour agreement of that window.
