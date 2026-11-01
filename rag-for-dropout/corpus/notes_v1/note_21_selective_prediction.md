---
document_id: note_21_selective_prediction
title: Selective prediction and coverage
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: Geifman and El-Yaniv (2017); reports/layer0/FINAL_REPORT.md; reports/layer0/FINAL_REPORT.md section 2
verification: unverified paraphrase; author review pending
merged_from:
- B:note_21_selective_prediction
- A:N15
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Selective prediction and coverage

Selective prediction accepts a window only when the confidence is high enough and sends the rest to a person. Coverage is the share of windows accepted; selective risk is the error rate among the accepted windows. For the frozen gate_v1 at its 95 percent target, coverage on the test split is 0.845, 0.825, 0.801, 0.731 and 0.513 with 0, 1, 2, 5 and 10 sensors dark, and the accuracy of the CONFIDENT windows is 0.949 to 0.950.

At the 98 percent operating point coverage is 0.772 with no sensor dark and 0.376 with 10 dark. More dark μPMUs means more LOW_CONFIDENCE reports.
