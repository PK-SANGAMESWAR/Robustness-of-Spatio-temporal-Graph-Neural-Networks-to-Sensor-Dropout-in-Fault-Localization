# PC-STGNN — independent pre-registration (blind draft)

Status: **DRAFT, not approved.** Written 2026-11-18. The author must approve before use.

## 0. Provenance and limits (read first)
- Written without opening any file in the repository: no results, no CSV/JSON, no `PC-STGNN/` spec files,
  no `14_PREREGISTRATION.md`, no code.
- The only input was the project summary that loads automatically into every session (`CLAUDE.md`). That
  summary states the *outcome* of Gate 4 (sign rule passed for R5 vs R0ⁿ). So this draft is **blind to the
  result files, but not fully blind to the Gate 4 outcome.**
- Results for Gate 4 and the Stage G queue already exist on disk. This draft is therefore **not** a
  pre-registration for those comparisons. Use it (a) as an independent check of the existing plan in
  `14_PREREGISTRATION.md`, or (b) as the pre-registration for analyses that have **not yet been run**.
- Hypothesis labels here use the prefix **B** (B1, B2, …) so they are never confused with H1–H6 in the
  existing plan.
- No result numbers appear in this file. All thresholds below are design choices for the author to approve.

## 1. Research question
Does fusing a **fixed, zero-parameter physics operator** (signature table over a hypothesis grid, and its
residual ρ against the observed μPMU window) into the Paper-1 STGNN backbone improve fault localization on
the IEEE-123 feeder, compared with the same backbone without the operator?

Three regimes: (i) in-distribution, all sensors present; (ii) sensor dropout; (iii) out-of-distribution (OOD)
conditions held in the sealed set.

Wording: "physics-consistent". Never "PINN". The operator is fixed. It is never learned and never fitted on
validation or test events.

## 2. Arms
| Arm | Definition | Purpose |
|---|---|---|
| **A0** control | Paper-1 backbone (GRU encoder → GNN, depth 4 → node head → soft vote), no operator input | Baseline |
| **A1** fused | A0 plus the fixed operator features fused in. Only this one variable differs from A0 | Treatment |
| **A2** shuffled-table | A1 with the signature table permuted across hypotheses (fixed permutation, fixed seed). Same parameter count and input shape as A1 | Separates physics content from added capacity |
| **A3** operator-only | No learned part: predict the hypothesis with the lowest residual, map it to a bus | Zero-parameter reference |

A2's permutation is generated once, its seed and hash are written in §9 before any A2 training.

## 3. Fixed across all arms (one variable at a time)
- Corpus `dataset_fullscale_50/` (frozen), same event-level 70/15/15 splits, same leakage check.
- Same CLI flags: `--gnn-layers 4 --epochs 30 --grad-clip-norm 1.0 --warmup-epochs 3`. Same optimiser,
  learning rate, batch size, loss (unweighted cross-entropy), sampler.
- Checkpoint rule: best validation graph macro-F1. Never chosen on test.
- Seeds: 42, 43, 44 for every arm. No extra seeds, no seed swaps.
- Training dropout regime: **one** regime per comparison family, chosen and written in §9 before training
  (standard or dropout-aware with k ∈ {0, 1, 2, 5}). All arms in a family use the same regime.
- Evaluation dropout: `zero_messages` (the Paper-1 protocol), k ∈ {0, 1, 2, 5, 10} of 25 sensors, same mask
  seeds for every arm.
- Operator files (`operator/mu.npy`, `operator/mu0.npy`) and the ρ cache: SHA-256 written in §9 before the
  first training run. A hash change after that voids the comparison.

## 4. Operator integrity (checked before scoring)
1. The operator has zero trainable parameters and receives no gradient (unit test plus a parameter count
   printed in each run log).
2. The signature table is built only from the network model and the training-range hypothesis grid. No
   validation, test or OOD event enters its construction.
3. The ρ cache for val and test is computed with the same frozen operator as for train.

## 5. Hypotheses
### Confirmatory (primary)
- **B1 — in-distribution.** A1 > A0 in test graph macro-F1 at k = 0.
- **B2 — dropout robustness.** A1 > A0 in test graph macro-F1 at each k ∈ {1, 2, 5, 10}.
- **B3 — OOD.** A1 > A0 in graph macro-F1 on the sealed OOD set, separately for each OOD condition listed
  in §9 (for example: fault resistance outside the training range; load outside the 0.5–1.3 training range).
  The list is fixed before the set is opened.
- **B4 — physics content.** A1 > A2 at k = 0 and at k = 10. If A1 ≈ A2, any A1 gain over A0 is attributed to
  capacity, not to physics.

### Secondary (reported, not used for the main claim)
- Area under the dropout-degradation curve (AUDC) over k ∈ {0, 1, 2, 5, 10}, A1 vs A0.
- Macro-F1 on the subset of faults at the 19 **unmeasured** fault buses vs the 6 measured ones.
- Localization distance error (graph hops from predicted to true bus), node-diagnostic F1, calibration (ECE).
- A3 vs A0 and A3 vs A1, all regimes.

### Exploratory
Anything not listed above. It is labelled "exploratory" in every report and never enters a Holm family.

## 6. Statistical plan
- **Unit of analysis:** one trained model (one seed). n = 3 per arm. Pairs are formed by seed.
- **Statistic:** paired difference d_s = F1(A1, seed s) − F1(A0, seed s).
- **Test:** one-sided paired t-test on d_s (H₀: mean d ≤ 0). α = 0.05.
- **Families (Holm within each):**
  - F-A: B1 + B2 → 5 cells (k = 0, 1, 2, 5, 10).
  - F-B: B3 → one cell per OOD condition in §9.
  - F-C: B4 → 2 cells (k = 0, 10).
  Families are fixed here. No cell is added, removed or moved after any result is seen.
- **Sign rule (reported beside every test):** count of seeds with d_s > 0 (0/3 … 3/3).
- **Secondary uncertainty:** within-seed bootstrap over test **runs** (not windows; all windows of a run stay
  together), 10,000 resamples, 95 % percentile CI. Reported, not used for the decision.
- **Effect size:** mean d with the three per-seed values listed. No standardized effect size from n = 3.

## 7. Decision rules (wording of the claim)
| Outcome per cell | Claim allowed |
|---|---|
| Holm-adjusted p < 0.05 **and** sign 3/3 | "A1 improves on A0" for that cell |
| Sign 3/3, Holm fails | "consistent trend, not significant" |
| Sign 2/3 or 1/3 | "no evidence of a difference" |
| Sign 0/3 **and** Holm-adjusted p < 0.05 on the reverse test | "A1 is worse than A0" (reported in full) |

- The headline claim "PC-STGNN improves localization" needs **B1 or B2** to pass **and** B4 to pass at the
  same k. If B4 fails, the text says the gain is not shown to come from the physics operator.
- B3 is reported per OOD condition. No pooled OOD claim.
- A trend is never upgraded to a result in any later text (paper, thesis, slides).

## 8. Validity checks and exclusions (fixed now)
- Before any score is used: confirm in `metrics_by_epoch.csv` that the best-val epoch is not epoch 1 and that
  the loss is finite in all epochs.
- **Diverged run** (non-finite loss, or best-val macro-F1 at the trivial always-"no fault" level): counted as a
  failure of that arm for that seed. It is reported. It is **not** rerun with another seed. One rerun with the
  **same** seed is allowed only for a documented infrastructure fault (process killed, disk full), and the
  log of the failed attempt is kept.
- No early stopping other than the checkpoint rule in §3. No hyper-parameter change for any single arm.
- Any arm-specific hyper-parameter that A1 needs (for example a fusion weight) is chosen on **validation only**,
  before the test scoring, and the same search budget is given to A0 (and A2) where it applies.

## 9. Values to fix before training / opening (author fills in, then dates)
| Item | Value | Date |
|---|---|---|
| Training dropout regime per family | | |
| A1 fusion design (where ρ enters the network) | | |
| A2 permutation seed and SHA-256 of the permuted table | | |
| SHA-256 of `operator/mu.npy`, `operator/mu0.npy`, ρ cache | | |
| OOD conditions in F-B (list) | | |
| Sealed set hash matches `data/ood_test.sha256` (yes/no) | | |

## 10. Sealed OOD set protocol
1. All arms trained, checkpoints frozen, hashes recorded, this document approved.
2. Author gives the go-ahead in writing.
3. Hash check against `data/ood_test.sha256`; the opening is written in `14_PREREGISTRATION.md` §6
   "Sealed-set log" (date, who, which checkpoints).
4. One scoring pass. No model, threshold or preprocessing change afterwards. A second opening is a deviation.
5. `data/ood_pilot/` numbers are never reported.

## 11. Reporting commitments
- All cells in F-A, F-B and F-C are reported, whatever the outcome, with per-seed values.
- Every deviation from this document is logged with date, reason and effect, and the affected analysis is
  labelled "deviates from pre-registration".
- Analyses added after any result is seen are labelled "post hoc".

## 12. Comparison with the existing plan (to do, by a reader who may open files)
Compare B1–B4, the families in §6 and the rules in §7–8 with `14_PREREGISTRATION.md` and Gate 4 / Stage G
records. Note each difference. Where the existing plan is weaker (for example, a family defined post hoc),
say so; do not change the existing plan to match this draft after its results are known.
