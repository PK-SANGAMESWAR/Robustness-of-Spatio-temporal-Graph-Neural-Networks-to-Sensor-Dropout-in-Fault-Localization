---
document_id: note_28_simulation_limits
title: Limits of the simulated data
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: results/RESULTS.md (this project); CLAUDE.md; implementation-plan.md section 5.1
verification: unverified paraphrase; author review pending
merged_from:
- B:note_28_simulation_limits
- A:N29
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Limits of the simulated data

All data come from OpenDSS simulations of the IEEE 123-bus feeder; the corpus has no measurement noise unless noise is added in a separate study. The train, validation and test splits share all 825 scenarios and differ in their runs. Results therefore do not show safety on a real grid, and the documents in this corpus were curated for research and are not approved by a utility.

Every report states that no utility-approved procedure was used.
