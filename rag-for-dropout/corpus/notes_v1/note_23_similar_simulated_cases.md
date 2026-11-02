---
document_id: note_23_similar_simulated_cases
title: Similar simulated cases (kNN evidence)
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md (this project); reports/layer0/STEP1_REPORT.md; reports/layer0/FINAL_REPORT.md section 4; reports/layer0/DECISION_RULE.md; reports/layer0/CANDIDATE_SETS_REPORT.md
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_23_similar_simulated_cases
- A:N21
- A:N22
- A:N24
- A:N25
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# Similar simulated cases (kNN evidence)

The case store holds latent vectors of simulated training windows with their true fault bus. The nearest cases to a new window are shown as similar simulated cases. Because every scenario appears in both train and test, about 83 percent of the 20 nearest neighbours of a window come from its own scenario when nothing is held out. The held-out store removes the scenario of the query; its kNN vote is then 0.831 on the test split with no sensor lost, below the 0.876 of the model itself. Neighbours are precedents, not proof that a prediction is correct.

kNN agreement is the fraction of neighbours whose true bus equals the prediction; the neighbour error rate is the fraction of neighbours that the model itself got wrong. On test, kNN agreement detects wrong predictions with AUROC 0.748 to 0.826, lower than the margin. It stayed in the gate by a pre-registered rule with a small effect (AUROC +0.011 on validation), and at the 95 percent operating point its threshold is 0. Neighbours cannot show conditions outside the simulation or a wrong reading from a present μPMU.
