# PC-STGNN Project — File Index

Extension of Paper 1 (*Dropout-Aware Training for Sensor-Robust Fault Localization with
STGNNs*, IEEE 123-bus, 25 µPMUs). This directory supersedes `../NEW-STRATEGY.md`, which is the
earlier single-file draft of the same plan and is kept for reference only.

**Reviewed against the repo on 2026-09-22 — read `REVIEW-2026-09-22.md` first.** Edits made in
that pass are tagged `[Corrected 2026-09-22]` inside each file.

## Read order

`01` is the frozen specification and is referenced by everything. Stage files `02–09` run in
sequence with hard gates. Files `10–14` are cross-cutting registries used by every stage.

| File | Contents | When |
|---|---|---|
| `REVIEW-2026-09-22.md` | Repo check of this plan: what was wrong, what already exists, what changed | First |
| `01_SPEC.md` | Frozen method specification. Single source of truth. | Always |
| `02_STAGE_A_audit.md` | Verify Paper 1 assets. Gate 0. | Week 1 |
| `03_STAGE_B_operator.md` | Build + validate the physics operator. Gate 1. | Weeks 2–4 |
| `04_STAGE_C_gonogo.md` | Training-free science. Gate 2 — the decision. | Weeks 4–6 |
| `05_STAGE_D_infra.md` | Injectors, baselines, OOD sets. Gate 3. | Weeks 5–9 |
| `06_STAGE_E_pcstgnn.md` | PC-STGNN training + core ablation. Gate 4. | Weeks 8–13 |
| `07_STAGE_F_gate.md` | Trust gate / silent failures. Gate 5. | Weeks 12–16 |
| `08_STAGE_G_eval.md` | Full evaluation + model mismatch. | Weeks 15–19 |
| `09_STAGE_H_writing.md` | Analysis, paper structure, release. | Weeks 19–26 |
| `10_FIGURES_TABLES.md` | Every figure and table, with its source step. | Cross-cutting |
| `11_METRICS.md` | Exact definition of every metric. | Cross-cutting |
| `12_CHECKLISTS.md` | Leakage, fairness, reproducibility. | Before every table |
| `13_BUDGET_TIMELINE.md` | Compute budget, schedule, fallbacks. | Planning |
| `14_PREREGISTRATION.md` | Fill in before opening the test split. | End of Stage C |

## The five guiding rules

1. **One variable at a time.** Data, splits, epochs, optimiser, checkpoint selection stay
   exactly as Paper 1.
2. **Physics is fixed** — never learned, never fitted on test events.
3. **Use physics only where electrical laws add information the network lacks:** unseen
   resistance, missing sensors, wrong sensors.
4. **Say "physics-consistent".** Never "PINN". Never claim KCL or power-flow residuals.
5. **Cheap training-free checks before expensive training runs.**

## Gate map

G0 assets → G1 fidelity → G2 go/no-go → G3 infrastructure → G4 PC-STGNN → G5 trust gate.

## What already exists in the repo (do not redo)

See `REVIEW-2026-09-22.md` §2. In short: all five dropout-aware baselines at 3 seeds, the
random / persistent / contiguous / targeted / noise injectors and their curves, the
neighbour-mean imputation baseline, the classical baselines, a rank-1 physics matched filter
(F1 0.162), a physics-in-the-loss pilot (`results/pignn_p3_full`, null), and the significance
and runtime tooling.
