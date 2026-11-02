---
document_id: note_05_fault_resistance
title: Fault resistance and detectability
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Anderson, P. M. (1995). Analysis of Faulted Power Systems. IEEE Press
project_sources: results/RESULTS.md § 1 (macro-F1 by fault resistance, standard RGSAGE-Max)
verification: project numbers checked against the named files 2026-11-02; general background not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_05_fault_resistance
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Fault resistance and detectability

Fault resistance limits the fault current and reduces the voltage depression that remote sensors see. A high-resistance fault therefore leaves a weaker signature. This project simulates three fault resistances: 0.1, 1 and 10 Ω. The 10 Ω faults are the hardest to localize: for the standard RGSAGE-Max model the macro-F1 is 0.949 at 0.1 Ω, 0.893 at 1 Ω and 0.676 at 10 Ω (RESULTS.md section 1).
