---
document_id: note_15_what_dropout_aware_training_recovers
title: What dropout-aware training recovers
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: results/RESULTS.md (this project)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_15_what_dropout_aware_training_recovers
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# What dropout-aware training recovers

For RGSAGE-Max, dropout-aware training gives macro-F1 values of 0.839, 0.827, 0.815, 0.774 and 0.674 when 0, 1, 2, 5 and 10 sensors are dark. With 10 sensors dark it keeps 80.3 percent of the clean macro-F1, against 5.7 percent for standard training, and the area under the degradation curve rises from 0.402 to 0.913. The cost in clean accuracy for RGSAGE-Max is 0.008 and is not significant. These are averages over a test set; they do not describe the reliability of one prediction.
