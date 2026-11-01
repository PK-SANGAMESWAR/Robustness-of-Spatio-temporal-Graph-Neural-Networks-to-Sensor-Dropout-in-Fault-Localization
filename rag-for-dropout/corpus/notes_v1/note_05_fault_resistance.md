---
document_id: note_05_fault_resistance
title: Fault resistance and detectability
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: Anderson, P. M. (1995). Analysis of Faulted Power Systems. IEEE Press; results/RESULTS.md (this project)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_05_fault_resistance
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Fault resistance and detectability

Fault resistance limits the fault current and reduces the voltage depression that remote sensors see. A high-resistance fault therefore leaves a weaker signature. This project simulates three fault resistances: 0.1, 1 and 10 Ω. The 10 Ω faults are the hardest to localize: for the standard RGSAGE-Max model the macro-F1 is 0.949 at 0.1 Ω, 0.893 at 1 Ω and 0.676 at 10 Ω (RESULTS.md section 1).
