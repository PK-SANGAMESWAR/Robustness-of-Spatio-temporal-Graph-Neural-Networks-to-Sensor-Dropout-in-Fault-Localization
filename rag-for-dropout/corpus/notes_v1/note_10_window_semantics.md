---
document_id: note_10_window_semantics
title: Window semantics and the report window
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project); reports/s2/S2_REPORT.md; fault_localization/models/common.py (unlabelable_window_mask)
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_10_window_semantics
- A:N04
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# Window semantics and the report window

A window has 20 time steps of 1 ms with stride 1. Windows w0000 to w0019 end before the fault and are labelled no fault. Window w0020 is labelled with the fault bus but holds no post-fault sample. A window w of 21 or more holds w minus 20 post-fault samples. The report window is index 25 (5 post-fault samples); it was chosen on the validation split as the earliest index within one percentage point of the best accuracy.

Because window w0020 has no post-fault sample, no model can classify it from its content.
