---
document_id: note_29_gate_belongs_to_one_checkpoint
title: A gate belongs to one checkpoint
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: reports/layer0/FINAL_REPORT.md
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_29_gate_belongs_to_one_checkpoint
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# A gate belongs to one checkpoint

The thresholds, the isotonic maps and the conformal quantiles of gate_v1 were fitted for the seed-42 checkpoint. Applied unchanged to the seed-43 and seed-44 checkpoints, the accuracy of the CONFIDENT windows at the 95 percent point was 0.897 to 0.932, below the target. Each new checkpoint needs its own calibration, its own gate fit and its own frozen test run.
