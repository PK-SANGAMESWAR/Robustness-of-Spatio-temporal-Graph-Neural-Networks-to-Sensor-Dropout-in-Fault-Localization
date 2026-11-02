---
document_id: note_14_standard_training_collapse
title: Why standard training fails when sensors are lost
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project)
verification: checked against the named project documents 2026-11-02; author review O30 2026-11-02
merged_from:
- B:note_14_standard_training_collapse
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Why standard training fails when sensors are lost

A model trained with every sensor present meets an input at inference that it never saw in training when some sensors are dark. For the standard RGSAGE-Max model the macro-F1 falls from 0.846 with no sensor lost to 0.048 when 10 of the 25 sensors are dark; that is 5.7 percent of its clean macro-F1. The most accurate standard model was also the one that degraded fastest, but this ordering is only a trend at three seeds (RESULTS.md sections 2 and 8).
