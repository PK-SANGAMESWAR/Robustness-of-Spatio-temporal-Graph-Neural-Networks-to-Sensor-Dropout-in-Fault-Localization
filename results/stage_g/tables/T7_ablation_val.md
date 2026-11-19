# T7 — core ablation, validation split (248,000 windows)

| arm | n_seeds | k0 σ0 (H6) | 10 Ω k0 σ0 (H1) | k10 σ0 (H2) | k0 σ1e-2 (H4) | k10 σ1e-2 |
|---|---|---|---|---|---|---|
| R0 (Paper 1, no noise aug.) | 3 | 0.845 ± 0.009 | not scored | not scored | not scored | not scored |
| R0ⁿ | 3 | 0.839 ± 0.008 | 0.641 ± 0.013 | 0.655 ± 0.012 | 0.724 ± 0.011 | 0.578 ± 0.014 |
| R0ⁿ − R0 (noise augmentation alone) | 3 | -0.005 (1/3 seeds > 0, p=0.316) |  |  |  |  |
| R1 (type head) | 1 | 0.674 | 0.491 | 0.502 | 0.585 | 0.441 |
| R2 (learned decoder) | 3 | 0.856 ± 0.002 | 0.668 ± 0.007 | 0.645 ± 0.007 | 0.743 ± 0.005 | 0.571 ± 0.011 |
| R3 (λ₁ only) | 1 | 0.727 | 0.348 | 0.564 | 0.639 | 0.506 |
| R4 (λ₂ only) | 3 | 0.794 ± 0.007 | 0.518 ± 0.031 | 0.586 ± 0.019 | 0.693 ± 0.008 | 0.520 ± 0.020 |
| R5 (PC-STGNN) | 3 | 0.898 ± 0.003 | 0.741 ± 0.011 | 0.857 ± 0.001 | 0.748 ± 0.008 | 0.679 ± 0.009 |
| R5-γ0 (no aux. loss) | 1 | 0.882 | 0.688 | 0.858 | 0.732 | 0.670 |
| R6ⁱ (shuffled table, inference) | 3 | 0.085 ± 0.001 | 0.150 ± 0.004 | 0.067 ± 0.004 | 0.220 ± 0.011 | 0.162 ± 0.008 |
| R8 (no noise aug.) | 1 | 0.887 | 0.735 | 0.843 | 0.487 | 0.423 |
| B2ᵃ (regressed R̂; epoch-1 checkpoint) | 1 | 0.709 | 0.558 | 0.506 | 0.562 | 0.425 |

- Fused soft-voted macro-F1 (Gate-4 scorer), mean ± sample std over the seeds listed; 3 offsets per cell.
- R0: only the training-loop validation F1 at k = 0, σ = 0 exists (R0 was not run through the noise grid).
- R0ⁿ − R0 is what noise augmentation alone buys (12_CHECKLISTS § A, pairing trap).
- R6 (trained) collapsed twice and is not a row; R6ⁱ replaces it (14, Amendment 5). R7 dropped (D6).
- 1-seed rows (R1, R3, R5-γ0, R8, B2ᵃ): no significance claim.
- B2ᵃ is the epoch-1 checkpoint of an unstable run (see T5 notes); its row is not evidence for or against a regressed R̂.
