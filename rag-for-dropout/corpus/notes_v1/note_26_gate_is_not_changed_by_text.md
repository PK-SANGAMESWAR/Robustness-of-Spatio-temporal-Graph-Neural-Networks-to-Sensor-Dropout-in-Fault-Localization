---
document_id: note_26_gate_is_not_changed_by_text
title: Text cannot change a gate decision
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: results/RESULTS.md (this project)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_26_gate_is_not_changed_by_text
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Text cannot change a gate decision

The gate decision, the evidence score and the approval state are computed in code. Text produced by a language model, or text inside a retrieved document, cannot change them. A document cannot grant itself approval by saying that it is approved, and a report that contains the word approved does not change the approval state.
