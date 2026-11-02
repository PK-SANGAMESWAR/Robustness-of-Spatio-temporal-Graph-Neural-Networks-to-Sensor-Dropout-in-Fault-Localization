---
document_id: note_08_partial_observability
title: Partial observability on the feeder
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project); fault_localization/config/constants.py; results/RESULTS.md section 8 (trivial reference); CLAUDE.md key facts (exactly 50 % no-fault)
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_08_partial_observability
- A:N03
- A:N05
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# Partial observability on the feeder

Only 25 buses of the feeder carry a μPMU. There are 25 candidate fault buses, and only 6 of them (13, 18, 25, 35, 47, 97) are μPMU buses. The other 19 candidate fault buses are not measured directly, so the model infers their location from the voltage disturbance seen at other buses. Localization is therefore a classification over 25 fault buses plus a no-fault class, based on remote measurements.

Half of the windows in the corpus are no-fault windows. A classifier that always answers no fault scores a 26-class macro-F1 of 0.0256, so a macro-F1 near this value means that the model has lost its ability to locate faults.
