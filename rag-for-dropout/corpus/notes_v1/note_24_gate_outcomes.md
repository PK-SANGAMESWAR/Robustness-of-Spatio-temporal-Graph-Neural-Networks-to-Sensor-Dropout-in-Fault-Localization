---
document_id: note_24_gate_outcomes
title: Gate outcomes and reason codes
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: reports/layer0/FINAL_REPORT.md; configs/gate_v1.json
verification: unverified paraphrase; author review pending
merged_from:
- B:note_24_gate_outcomes
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Gate outcomes and reason codes

The gate returns CONFIDENT, LOW_CONFIDENCE or REJECT. LOW_CONFIDENCE means the input is valid but a condition failed; the reason codes are LOW_CALIBRATED_CONFIDENCE, LOW_MARGIN and LOW_KNN_AGREEMENT. REJECT means no report is drafted: UNSUPPORTED_DROPOUT_PATTERN for a number of dark sensors other than 0, 1, 2, 5 or 10 or a pattern other than random, and INVALID_INPUT for malformed votes. A passed check has the code VERIFICATION_PASSED.
