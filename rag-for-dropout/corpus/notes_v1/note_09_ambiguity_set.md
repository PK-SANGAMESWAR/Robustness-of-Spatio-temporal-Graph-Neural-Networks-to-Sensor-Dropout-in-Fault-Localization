---
document_id: note_09_ambiguity_set
title: Ambiguity sets
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: Phadke, A. G. and Thorp, J. S. Synchronized Phasor Measurements and Their Applications. Springer; corpus/feeder_v1/feeder_fault_buses.md (nearest μPMUs)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_09_ambiguity_set
- A:N20
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Ambiguity sets

An ambiguity set is a group of candidate fault locations that produce nearly the same measurements at the available sensors, so the data cannot separate them. Ambiguity grows when sensors are missing. When the evidence is ambiguous a report should show the candidates and not a single bus. The current gate does not output a physical ambiguity set; for LOW_CONFIDENCE windows it outputs a conformal candidate set (about three to four classes).

Two fault buses that are close to the same μPMUs give similar voltage patterns. When one of those μPMUs is dark, the pattern can fit both buses, and the candidate set then contains both.
