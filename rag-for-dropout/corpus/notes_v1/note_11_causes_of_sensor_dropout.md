---
document_id: note_11_causes_of_sensor_dropout
title: Why sensors drop out
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Phadke, A. G. and Thorp, J. S. Synchronized Phasor Measurements and Their Applications. Springer
verification: general background; not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_11_causes_of_sensor_dropout
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Why sensors drop out

A μPMU can stop delivering usable data because of communication loss, loss of power, loss of its time reference, or a hardware failure. For the localization model the effect is the same in every case: some inputs are missing at the moment of the fault. This is different from permanently placing fewer sensors, which is known when the model is built.
