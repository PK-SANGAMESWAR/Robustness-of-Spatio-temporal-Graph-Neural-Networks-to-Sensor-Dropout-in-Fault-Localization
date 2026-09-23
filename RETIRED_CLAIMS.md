# Retired claims

Claims this project made, believed, or wrote into a spec — and the measurement that killed them.
One line each. Started 2026-09-23, after measurement overturned three plausible stories in two
days.

**Why this file exists.** Each of these was reasonable when written. What retired them was not a
better argument but a bigger sample or a direct test, and the pattern across them is more useful
than any one entry: *cheap checks were run on convenience samples, and their conclusions were then
written down as population facts.* That pattern is now a rule (`PC-STGNN/12_CHECKLISTS.md` §0).

This file is also the raw material for the paper's limitations section, and a reviewer who sees it
reads the rest of the work differently.

| # | Claim | Whose | What killed it | What replaced it |
|---|---|---|---|---|
| 1 | "σ_ε is set a priori from the snapshot-vs-Daily equivalence error measured in S0.4 (order 1e-3 p.u.)" | `01_SPEC.md` §4 | S0.4 measured that error at **1.02e-12 p.u.** — the operator is the corpus's own generator, not an approximation of it | σ_ε from S1.2 grid discretisation, declared *numerical* on the nominal table, with a stated physical floor (µPMU accuracy class) |
| 2 | "`reuse` mode is identical to `daily`, bit for bit, so the fast build path is validated" | `AUDIT-NOTE.md` S0.4, on **5 events** | The same comparison on **1,000 events**: the standing-circuit path disagrees with the corpus by > 1e-3 p.u. on **43.3 %** of them, because OpenDSS carries regulator taps across solves | `build_table` recompiles per solve (`mode="daily", reuse_limit=1`); 1.8 % residual disagreement, documented |
| 3 | "Post-inception samples are constant to < 1e-6 p.u." | `02_STAGE_A_audit.md` S0.3 | Full-corpus scan: **1.44e-5**, from two-solve settling | Restated as "< 1e-6 from step 42 onward", with the settling bounded and asserted |
| 4 | "Solver settling is ≤ 1.4e-5 p.u." | `AUDIT-NOTE.md` S0.3, on **400 events** | All **41,250** events: ≤ 1.98e-5 for 41,247, and **three** control-driven outliers at 6.4–7.1e-3 the sample never saw | `SETTLING_BOUND_PU = 2e-5` plus an enumerated `SETTLING_OUTLIERS` set, both asserted |
| 5 | "A non-monotonic sag almost always means a regulator or cap-bank control interacted with the fault" | `03_STAGE_B_operator.md` S1.3 | Re-solving nine non-monotone cells with `ControlMode=OFF`: **4 of 9 stay non-monotone** | Largely **intrinsic to ungrounded faults** on this feeder; 83 % of violations are ungrounded types. Controls stay as the corpus had them |
| 6 | "The three settling outliers and the non-monotone cells are almost certainly the same mechanism" | Claude, Stage A commit `10eebdc` | The outlier cells are **monotone** in R, and most non-monotone cells survive controls being disabled. `55_BC_1p0_0007`'s settling *is* control-driven (7.09e-3 with controls, 1.4e-5 without) — but that is one event, not a shared mechanism | Two distinct phenomena that share only "ungrounded fault" |
| 7 | "The λ₂ term divides by \|D\|, so an empty D gives NaN on every clean batch" | Author review, 2026-09-23 | Direct test: `rho_heldout` with an all-zero `d_mask` returns **exact zeros**, and `pc_loss`'s λ₂ term contributes 0 | No bug — the shifted/halved ρ of `01_SPEC.md` §4 removed both the `/\|A\|` and `/\|D\|` normalisations. Now pinned by four adversarial tests so it cannot come back |
| 8 | "S0.1 passes only on exact match at every k" | `02_STAGE_A_audit.md` S0.1 | Mask realisation depends on a **batch size nobody recorded**; two valid batch sizes differ from each other by up to 9.3e-4 macro-F1 | Exact at k = 0; all 33 Paper-1 arms re-evaluated inside Paper 2's own pipeline as the paired reference; `batch_size` now recorded in every curve CSV |

## Standing corrections that are not yet claims

- **Paper 1's reported 2.5 M windows** is not reconstructable from the 825-scenario grid (1.65 M).
  Already a documented deviation; restated here so Paper 2 never repeats the figure.
- **RGATv2's top relative-AUDC position** was a normalisation artefact (0.626 relative → 0.324
  absolute, 4th of 5) and is not claimed as robustness.
