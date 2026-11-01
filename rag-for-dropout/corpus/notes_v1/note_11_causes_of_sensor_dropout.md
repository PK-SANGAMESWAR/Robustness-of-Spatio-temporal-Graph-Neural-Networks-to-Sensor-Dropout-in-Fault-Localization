---
document_id: note_11_causes_of_sensor_dropout
title: Why sensors drop out
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: Phadke, A. G. and Thorp, J. S. Synchronized Phasor Measurements and Their Applications. Springer
verification: unverified paraphrase; author review pending
merged_from:
- B:note_11_causes_of_sensor_dropout
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Why sensors drop out

A μPMU can stop delivering usable data because of communication loss, loss of power, loss of its time reference, or a hardware failure. For the localization model the effect is the same in every case: some inputs are missing at the moment of the fault. This is different from permanently placing fewer sensors, which is known when the model is built.
