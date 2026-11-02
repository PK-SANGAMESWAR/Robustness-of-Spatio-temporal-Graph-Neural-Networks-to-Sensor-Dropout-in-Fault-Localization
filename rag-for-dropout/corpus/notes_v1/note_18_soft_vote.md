---
document_id: note_18_soft_vote
title: The soft vote
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project); fault_localization/models/voting.py
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_18_soft_vote
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# The soft vote

The model classifies each of the 25 nodes separately. The decision for a window sums the 25 node softmax vectors and takes the class with the largest sum. The summed vector adds up to 25, not to 1. Dividing by 25 gives a vote share, but the vote share is not a calibrated probability of being correct.
