---
document_id: note_26_gate_is_not_changed_by_text
title: Text cannot change a gate decision
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: implementation-plan.md § 1, § 3, § 12.4 and § 17; rag-for-dropout/claude-rag.md § 4 rule 3; rag-for-dropout/configs/gate_v1.json (the gate rule is code)
verification: checked against the named project documents 2026-11-02; author review O30 2026-11-02
merged_from:
- B:note_26_gate_is_not_changed_by_text
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Text cannot change a gate decision

The gate decision, the evidence score and the approval state are computed in code. Text produced by a language model, or text inside a retrieved document, cannot change them. A document cannot grant itself approval by saying that it is approved, and a report that contains the word approved does not change the approval state.
