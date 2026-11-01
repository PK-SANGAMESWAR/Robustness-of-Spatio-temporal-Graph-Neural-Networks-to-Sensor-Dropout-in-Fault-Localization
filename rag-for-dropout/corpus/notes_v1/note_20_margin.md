---
document_id: note_20_margin
title: The top-1 minus top-2 margin
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: Geifman and El-Yaniv (2017); reports/layer0/FINAL_REPORT.md; reports/layer0/FINAL_REPORT.md section 1
verification: unverified paraphrase; author review pending
merged_from:
- B:note_20_margin
- A:N14
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# The top-1 minus top-2 margin

The margin is the vote share of the best class minus the vote share of the second-best class. It is a simple measure of how clearly one class wins. On the test split its error-detection AUROC is 0.915 with no sensor lost and 0.887 with 10 lost, slightly above the isotonic probability (0.911 and 0.885). At the 95 percent operating point of the frozen gate the margin is the only threshold that is used.

The margin threshold grows with dropout, from 0.1459 with no sensor dark to 0.2518 with 10 dark. At this operating point the probability threshold is 0 with 0, 1, 5 and 10 sensors dark, and the kNN-agreement threshold is 0 at every k.
