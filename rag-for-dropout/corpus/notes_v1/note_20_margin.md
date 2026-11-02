---
document_id: note_20_margin
title: The top-1 minus top-2 margin
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Geifman and El-Yaniv (2017)
project_sources: reports/layer0/FINAL_REPORT.md § 1 and § 4; rag-for-dropout/configs/gate_v1.json
verification: project numbers checked against the named files 2026-11-02; general background not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_20_margin
- A:N14
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02); wording corrected by session A
---
# The top-1 minus top-2 margin

The margin is the vote share of the best class minus the vote share of the second-best class. It is a simple measure of how clearly one class wins. On the test split its error-detection AUROC is 0.915 with no sensor lost and 0.887 with 10 lost, slightly above the isotonic probability (0.911 and 0.885). At the 95 percent operating point of the frozen gate the margin threshold is set at every k, the probability threshold only at k = 2, and the kNN-agreement threshold at no k.

The margin threshold is 0.1459 with 0 and 1 sensors dark, 0.1368 with 2, 0.1633 with 5 and 0.2518 with 10 dark. At this operating point the probability threshold is 0 with 0, 1, 5 and 10 sensors dark and 0.5424 with 2 dark, and the kNN-agreement threshold is 0 at every k.
