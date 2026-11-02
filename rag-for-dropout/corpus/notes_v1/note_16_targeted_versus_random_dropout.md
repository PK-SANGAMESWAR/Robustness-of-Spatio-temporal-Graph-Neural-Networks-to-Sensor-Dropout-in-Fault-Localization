---
document_id: note_16_targeted_versus_random_dropout
title: Targeted dropout is harder than random dropout
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: results/RESULTS.md § 5; results/dropout_study_targeted/targeted_curves_standard.csv and targeted_curves_dropout_aware.csv (the +0.307 to +0.473 range is derived from these two files); reports/layer0/FINAL_REPORT.md § 1
verification: checked against the named project documents 2026-11-02; author review O30 2026-11-02
merged_from:
- B:note_16_targeted_versus_random_dropout
- A:N09
- A:N28
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02); wording corrected by session A
---
# Targeted dropout is harder than random dropout

Targeted dropout removes the μPMUs closest to the fault. With one sensor lost, the standard model falls to 0.448 under targeted loss against 0.736 under random loss. The dropout-aware model is also penalised by targeted loss, but keeps 0.307 to 0.473 macro-F1 over the standard model under the same targeting. With one sensor lost, the dropout-aware model under targeted loss (0.755) is not distinguishable from the standard model under random loss (0.736; difference +0.018, p = 0.33 over three seeds). From two sensors lost, the dropout-aware worst case (0.695) is above the standard average case (0.590). Windows without a fault fall back to random selection, so these numbers are a conservative bound.

Under targeting the area under the degradation curve of the dropout-aware model falls from 0.913 to 0.747, and its macro-F1 with 10 sensors dark is 0.521. The GridSense gate is fitted for random dropout only: a dropout pattern that it was not fitted for gives REJECT with UNSUPPORTED_DROPOUT_PATTERN. If a dark μPMU is one of the nearest μPMUs of the predicted bus, treat the location with more care.
