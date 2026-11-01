---
document_id: note_17_noise_versus_dropout
title: Measurement noise is not sensor dropout
document_type: technical_note
status: draft
approval_status: curated_for_research
version: 0
source_ref: results/RESULTS.md (this project); rag-for-dropout/EVAL.md section 10 (A3 not run)
verification: unverified paraphrase; author review pending
merged_from:
- B:note_17_noise_versus_dropout
- A:N10
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: null
---
# Measurement noise is not sensor dropout

Noise leaves the availability mask at 1, so the pattern that dropout-aware training learns to recognise never appears. Dropout-aware training therefore does not help against measurement noise. For the standard model at no dropout the macro-F1 falls from 0.846 to 0.786, 0.579 and 0.345 as the noise standard deviation grows from 0.0005 to 0.002 and 0.01 per unit. Noise and dropout together can push the result below the always-no-fault baseline of 0.0256.

A wrong value from a present μPMU (a silent failure) also leaves the mask at 1. The GridSense gate has no silent-failure check (ablation step A3 is not run), so a report cannot exclude a wrong reading from a present μPMU.
