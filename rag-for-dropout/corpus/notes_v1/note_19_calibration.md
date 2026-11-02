---
document_id: note_19_calibration
title: Calibration and isotonic regression
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: general background, not checked against a specific page
further_reading:
- Guo et al. (2017) On Calibration of Modern Neural Networks
- Zadrozny and Elkan (2002)
project_sources: reports/layer0/STEP2_REPORT.md; reports/layer0/FINAL_REPORT.md § 4
verification: project numbers checked against the named files 2026-11-02; general background not checked against a specific page; author review O30 2026-11-02
merged_from:
- B:note_19_calibration
- A:N13
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O30 review (2026-11-02)
---
# Calibration and isotonic regression

A confidence is calibrated when predictions made with confidence p are right about a fraction p of the time. Expected calibration error (ECE) measures the gap; this project uses 15 equal-mass bins. Isotonic regression fits a monotone map from the vote share to the probability of being correct. The map is fitted separately for each number of dark sensors, because the vote scale changes with dropout: a map fitted without dropout had ECE 0.007 at k = 0 but 0.048 at k = 10, while per-k maps give ECE of 0.006 to 0.009 on the test split.

The values 0.007 and 0.048 are from the validation half cal-B (post-fault windows). On test the raw vote share has an ECE of 0.121 to 0.173.
