---
document_id: note_12_missing_data_representation
title: How a dropped sensor is represented
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project); fault_localization/dropout/masking.py
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_12_missing_data_representation
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# How a dropped sensor is represented

A dropped μPMU has its three voltage channels set to 0 and its three mask channels set to 0, giving the mask pattern (0,0,0). A voltage of 0 can be a valid reading, so the mask is the authoritative indicator of a missing sensor. The mask channels of a working node are static and encode how many phases the μPMU measures (1, 2 or 3 phases); 14 of the 25 nodes are three-phase. Standard training never sees the pattern (0,0,0).
