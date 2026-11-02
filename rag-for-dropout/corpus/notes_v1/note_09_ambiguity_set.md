---
document_id: note_09_ambiguity_set
title: Ambiguity sets
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: implementation-plan.md § 11.1 (ambiguity set) and § 11.3 (gate outcomes); rag-for-dropout/configs/gate_v1.json and configs/candidate_sets_v1.json; reports/layer0/CANDIDATE_SETS_REPORT.md; corpus/feeder_v1/feeder_fault_buses.md (nearest μPMUs)
verification: checked against the named project documents 2026-11-02; author review O30 2026-11-02
merged_from:
- B:note_09_ambiguity_set
- A:N20
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02); wording corrected by session A
---
# Ambiguity sets

An ambiguity set is a group of candidate fault locations that produce nearly the same measurements at the available sensors, so the data cannot separate them. Which candidates are indistinguishable depends on which sensors are available. When the evidence is ambiguous a report should show the candidates and not a single bus. The current gate does not output a physical ambiguity set; for LOW_CONFIDENCE windows it outputs a conformal candidate set (about three to four classes).

Two fault buses that are close to the same μPMUs give similar voltage patterns. When one of those μPMUs is dark, the pattern can fit both buses, and the candidate set then contains both.
