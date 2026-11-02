---
document_id: note_22_conformal_sets
title: Conformal candidate sets
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Vovk, Gammerman and Shafer (2005)
- Angelopoulos and Bates (2021)
project_sources: reports/layer0/CANDIDATE_SETS_REPORT.md; reports/layer0/FINAL_REPORT.md § 0 and § 3; rag-for-dropout/claude-rag.md § 9 rule 4
verification: project numbers checked against the named files 2026-11-02; general background not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_22_conformal_sets
- A:N17
- A:N18
- A:N19
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Conformal candidate sets

A conformal prediction set contains the classes whose score passes a threshold fitted on held-out data, so that the true class is inside the set with a chosen probability, for example 90 percent. The guarantee is marginal: it holds over the population used to fit the threshold, not for every subgroup. A set fitted on all windows covered only 55 to 81 percent of the LOW_CONFIDENCE windows. candidate_sets_v1 is fitted on LOW_CONFIDENCE windows only and covers about 90 percent of them with 3.1 to 4.3 classes on average.

The 90 percent is a property of the method over many LOW_CONFIDENCE windows; one event's set can miss the true location. A CONFIDENT report names one bus, a REJECT report shows no bus and no set, and only a LOW_CONFIDENCE report shows a candidate set.
