---
document_id: note_25_prediction_is_not_confirmation
title: A prediction is not a confirmed fault
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: implementation-plan.md § 10.2 (prompt design), § 10.3 (report schema, pending_human_review), § 12.4 (human review) and § 13.5 (report quality)
verification: checked against the named project documents 2026-11-02; author review O30 2026-11-02
merged_from:
- B:note_25_prediction_is_not_confirmation
- A:N26
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# A prediction is not a confirmed fault

The model predicts a candidate fault bus. It does not confirm a fault. A person must confirm the location with the processes that the utility authorises before any action. Every report is a draft with the status pending human review, and the report must never state that the fault is confirmed or that any action has been taken.

A report is not an instruction to operate equipment.
