---
document_id: note_13_dropout_aware_training
title: Dropout-aware training
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project)
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_13_dropout_aware_training
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# Dropout-aware training

Dropout-aware training masks random μPMU nodes while the model is trained. For each training batch the number of masked nodes k is drawn from {0, 1, 2, 5}, and the model learns that the mask pattern (0,0,0) means a dark sensor. The architecture, the number of parameters (95,258) and the inference latency do not change; the training time per epoch rises by about 2.7 percent (RESULTS.md, runtime summary).
