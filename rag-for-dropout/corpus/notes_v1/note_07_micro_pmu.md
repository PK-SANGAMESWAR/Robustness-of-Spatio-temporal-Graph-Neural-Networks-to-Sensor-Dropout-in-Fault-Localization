---
document_id: note_07_micro_pmu
title: What a μPMU measures
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Phadke, A. G. and Thorp, J. S. Synchronized Phasor Measurements and Their Applications. Springer
- IEEE Std C37.118.1
project_sources: fault_localization/config/constants.py (NUM_PMU_NODES = 25, NUM_PHASES = 3, NUM_MASK_CHANNELS = 3, WINDOW_TENSOR_SHAPE = (25, 20, 6)); dataset_fullscale_50/processed/node_mapping.json (25 buses)
verification: project numbers checked against the named files 2026-11-02; general background not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_07_micro_pmu
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# What a μPMU measures

A micro-phasor measurement unit (μPMU) measures voltage and current phasors at distribution level with precise time stamps from a common time reference, at a high reporting rate. IEEE Std C37.118.1 defines the synchrophasor measurement requirements. In this project each of the 25 μPMUs provides three RMS phase voltages per time step to the model, together with three availability-mask channels.
