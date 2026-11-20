# A4 (B2ᵃ on OOD R): post hoc definition and result `[Recorded 2026-11-20]`

**Post hoc.** `14_PREREGISTRATION.md` Amendment 5 left "B2ᵃ collapses on OOD R" undefined. The sealed set was opened on 2026-11-18, and B2ᵃ's sealed F1 (0.231) and validation F1 (0.709)
were already in T7/T8 before this rule was written. The author approved the rule below on 2026-11-20 (chat answer to the question "which definition of A4 do you approve"). It is not a
pre-registered test. It uses no new sealed-set access: only existing scoring files. 14 is not edited.

## Rule (approved)
A4 **holds** if the pooled sealed F1 of B2ᵃ (k = 0, σ = 0) is at least 50 % below its validation F1 (k = 0, σ = 0). R5's drop is reported beside it. One seed (42) for B2ᵃ. No test, no p-value.

## Result: A4 holds under this rule
Sources: `results/stage_g/val/b2a/seed_00042_scoring.json`, `results/stage_g/ood/b2a_seed00042_scoring.json`, `results/stage_g/ood/r5_seed0004{2,3,4}_scoring.json`, `results/stage_g/tables/T7_ablation_val.md` (R5 validation).

| Model | Validation F1 (k=0, σ=0) | Sealed F1 (k=0, σ=0) | Relative drop |
|---|---|---|---|
| B2ᵃ (seed 42, epoch-1 checkpoint) | 0.7089 | 0.2313 | **−67.4 %** (rule met: ≥ 50 %) |
| R5 (3 seeds; sealed 0.5299 / 0.5363 / 0.5369, mean 0.5344) | 0.898 | 0.5344 | −40.5 % |

Descriptive, from the same B2ᵃ scoring file (σ = 0, k = 0): by resistance 3 Ω 0.344, 5 Ω 0.368, 20 Ω 0.052, 30 Ω 0.047; the network branch alone 0.064; PSM alone 0.223.
R̂ itself was not stored, so the "three-way classifier" picture of `06_STAGE_E_pcstgnn.md` (a head trained on {0.1, 1, 10} Ω) is not checked here. That would need B2ᵃ rescored on the sealed set (a second opening).

## How to word it
- Say: "Under a post hoc rule (≥ 50 % relative F1 drop from validation to the sealed set), B2ᵃ drops 67 % (one seed) and R5 drops 40 %."
- Do not say: "A4 was confirmed" or "regressed resistance fails off-grid" as a general result. B2ᵃ is the epoch-1 checkpoint of an unstable run (best validation F1 0.709 at epoch 1; later epochs 0.46 to 0.67; train loss about 100 against 2.6 for R5), so the drop does not separate "a regressed R̂ fails" from "this run was unstable".
- A4 is not in a Holm family and no claim rests on it. Limitation L3 (no stable B2ᵃ retrain) stays.
