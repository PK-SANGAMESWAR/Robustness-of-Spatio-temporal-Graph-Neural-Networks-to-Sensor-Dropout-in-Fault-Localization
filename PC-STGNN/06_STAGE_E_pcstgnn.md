# Stage E — PC-STGNN (Weeks 8–13)

**Purpose.** Train the core model and prove the gain comes from physics, not from capacity or
auxiliary tasks.
**Compute.** `[Corrected 2026-09-22]` ≈ 7 h per 30-epoch run on this backbone
(`results/runtime/runtime_training.csv`), so 130–175 GPU-h depending on the seed plan in
`13_BUDGET_TIMELINE.md`.

## Context you need

`01_SPEC.md` §6 for the loss; §5 for the architectural rule; §8 for the gradient audit table.
λ₁ = λ₂ = β = 0 must recover R0 exactly. The local precedent for "physics in the loss" is a null
(`results/pignn_p3_full`, 0.847 vs 0.846 at 3 seeds, rank-1 template); the B1 → B2 gap from
Stage C is your evidence that this operator is different in kind.

## S4.1 — The null test — run before anything else

**Do.** Train the new model with λ₁ = λ₂ = β = 0, gate off, seed 42.
`scripts/run_pcstgnn_null_test.py`.

**Check.** Loss curve and per-epoch validation F1 against the existing R0 run
(`results/dropout_aware/rgsage_max/seed_00042/metrics_by_epoch.csv`).

**Pass.** Exact reproduction. (`tests/test_pignn.py` already asserts byte-identity of the PIGNN
model with all switches off against RGSAGE-Max — copy that pattern.)

`[Corrected 2026-09-30]` **benchmark_version workaround, so this isn't rediscovered as a mystery
later.** The null-test run trains under `--benchmark-version dropout_aware_null`
(`results/dropout_aware_null/`), not R0's `dropout_aware` — its own manifest check would otherwise
fail against R0's frozen `results/dropout_aware/manifest.json` (Aug 2), which predates
`dataset_fullscale_50/splits/val_sample.txt` (a Stage C S2.3/S2.6 sample added 2026-09-25, read by
`run_b1_deltav_baseline.py`/`run_inference_fusion.py`). Diffed the two manifests file-by-file:
exactly one file added, zero files changed — and `WindowGraphDataset` (`data/dataset.py:36`) reads
`splits/{split}.txt` by explicit name only, never a glob, so `val_sample.txt` was never reachable
from the training path either way. R0's frozen manifest is left untouched as evidence; only the
null-test run's own manifest is fresh. **What is actually being compared does not change**:
`scripts/run_pcstgnn_null_test.py` still diffs its output against R0's original
`results/dropout_aware/rgsage_max/seed_00042/metrics_by_epoch.csv`.

**Branch on failure.** You have an unintended change in the data path, seeding, or graph
construction. Find it now. Every attribution claim in the paper rests on this single property —
it is what lets you say "λ = 0 recovers the Paper-1 model exactly."

`[Corrected 2026-09-30]` **PASSED — exact reproduction.** All 30 epochs of
`results/dropout_aware_null/pcstgnn/seed_00042/metrics_by_epoch.csv` are byte-identical to
`results/dropout_aware/rgsage_max/seed_00042/metrics_by_epoch.csv` on `train_loss`, `val_graph_f1`
and `val_node_f1` (max abs diff 0.0 on every epoch, per `scripts/run_pcstgnn_null_test.py`'s own
check); `best_validation_graph_f1` also matches exactly (0.8554909744946672). "λ = β = 0 recovers
the Paper-1 model exactly" holds.

## S4.2 — Gradient audit — guards the bug that was in the first draft

**Do.** `[Corrected 2026-09-23]` Finite-difference check on **three batches**, not one. A single
batch cannot exercise the paths that differ, and two of the three are the ones that break:

1. **A batch containing w = 20 windows**, so the L₁/L₂ loss mask is active and non-trivial.
2. **A batch at k = 10**, so |A| is small and |D| is large — the λ₂ term's intended regime.
3. **A batch at k = 0 with no synthetic drops**, so **|D| = 0**. This is the commonest case in
   training and the one the first-draft `r̃/|D|` formulation would have turned into NaN on every
   step. The λ₂ term must contribute exactly 0 here and the gradient must stay finite.

**Check.**
- ∂L₁/∂w ≠ 0 for network weights w
- ∂L₂/∂w ≠ 0
- β, α, c receive gradients
- μ, μ₀, σ_ε are not leaves with `requires_grad`; they receive no gradient

**Pass.** All four.

**Why this exists.** A reconstruction loss of the form ‖μ[f̂,…] − s_true‖² has zero gradient — f̂
is an arg-min index, constant in the weights. Under that formulation R4 would be silently
identical to R3 and you would lose weeks finding out. Two hours here prevents that.

`[Corrected 2026-09-30]` **PASSED, 12/12 applicable confirmations.** `scripts/run_pcstgnn_gradient_audit.py`
→ `results/pcstgnn/audit/s4_2_gradient/gradient_audit.json`. On all three batches: ∂L₁/∂w and
∂L₂/∂w computed directly from `forward_pc()`'s undetached `p_graph`/`rho_A`/`rho_D` via
`torch.autograd.grad` (not `pc_loss`'s reported `"l1"`/`"l2"`, which are `.detach()`d for logging
only — backward on those raises; `"total"` folds in CE and would contaminate the isolation), then
cross-checked against a genuine finite-difference estimate at the weight element with the largest
gradient magnitude (a fixed `[0,0]` landed on a coincidentally-zero element on the first attempt —
caught and fixed before trusting the result). β receives a gradient on all three batches
(4.7 / −42.0 / 116.1). μ (`phys_table`), μ₀ (folded into the same table) and σ_ε are confirmed
`requires_grad=False` and absent from `model.parameters()` on all three.

- **w = 20 batch**: ∂L₁/∂w ≠ 0 (norm 0.753, FD diff 4.9e-5); k = 0 here too, so ∂L₂/∂w = exactly 0
  (both autograd and numeric) — the batch's *own* stated expectation, not a failure. Loss mask
  independently confirmed non-vacuous: masked L₁ = 2.5018 vs. unmasked 2.4498 on the same batch.
- **k = 10 batch**: ∂L₁/∂w ≠ 0 (0.650, FD diff 2.2e-4) and ∂L₂/∂w ≠ 0 (0.874, FD diff 5.5e-6);
  `|D|_total = 480 = 10 dropped nodes × 3 channels × 16-sample batch`, arithmetic sanity-checked.
- **k = 0 batch (no w = 20 window)**: ∂L₁/∂w ≠ 0 (0.428, FD diff 1.65e-4); ∂L₂/∂w = exactly 0,
  finite — correct, since |D| = 0.

**α, c: N/A on the current model — not a skipped check, a scope finding.** `01_SPEC.md` §8's
gradient-audit table is the union of Stage E's fusion (β) and Stage F's PCRG-STGNN trust gate
(α, c, §7); `fault_localization/models/pc_stgnn.py` (224 lines, read in full) implements only β —
no `alpha`, no `c`, no gate mechanism exists yet. Re-check α/c when Stage F builds PCRG-STGNN
(`07_STAGE_F_gate.md`), not before.

## S4.3 — Tuning

**Do.** Single seed, 15-epoch budget (as in the Paper-1 depth study and the PIGNN screen). Sweep
λ₁ ∈ {0.1, 0.3, 1}, then λ₂ ∈ {0.1, 1} at the best λ₁. β is learned. ≈ 5–6 half-cost runs.

**Check.** Select on validation F1 averaged over k ∈ {0,1,2,5,10}. Never test. Never OOD.
(Checkpoint selection *within* a run stays validation F1 at k = 0, as in Paper 1 — two different
rules, both stated in the paper.)

**Branch.** Best λ₁ at a grid boundary → extend one step in that direction.

**Cost.** 6 × 3.5 h ≈ 21 GPU-h.

**Figure produced.** F-E1 — validation F1 vs λ₁ (appendix).

`[Corrected 2026-10-04]` **S4.3 RUN — RESULT PROVISIONAL AND NOT VALID FOR S4.4.**
`scripts/run_pcstgnn_tuning.py` → `results/pcstgnn_tuning/lambda_selection.json`, F-E1 at
`paper/figures/fe1_lambda1_sweep.png`.

Setup:
- Seed 42, 15 epochs per arm
- `use_fusion=True`, β learned (init 0), σ_ε = 7.2555e-4
- Dropout-aware training on (k ∈ {0,1,2,5})
- **Noise-free**: S4.3ᵇ was read at the time as scoping noise augmentation to S4.4 only. That reading is superseded below
- k = 0 ρ(A) served from `cache/rtilde_train.npy`, verified equal to the live path on real batches from this loader (max relative error 1.6e-5, `scripts/verify_pcstgnn_rho_cache.py`)

Run count: 6, not 5. λ₁ = 0.1 won on the grid's lower edge, so the branch rule added λ₁ = 0.0333.

| λ₁ | λ₂ | val F1, mean over k | val F1 at k=0 (checkpoint rule) | β |
|---|---|---|---|---|
| **0.0333** | **0** | **0.8501** | 0.8533 | 0.0222 |
| 0.1 | 0 | 0.8277 | 0.8323 | 0.0378 |
| 0.3 | 0 | 0.8026 | 0.8049 | 0.0508 |
| 1.0 | 0 | 0.8218 | 0.8293 | 0.0795 |
| 0.0333 | 0.1 | 0.8421 | 0.8452 | 0.0292 |
| 0.0333 | 1.0 | 0.8200 | 0.8176 | 0.0482 |

Compute was ≈ 74 GPU-h measured (≈ 10.8 h per arm plus ≈ 1.5 h λ-selection scoring), plus ≈ 8.6 h
lost to a host reboot.

**The S4.3ᵇ degeneracy was observed in every arm.** Source:
`results/pcstgnn_tuning/fusion_decomposition_val_k0.json`. Measured on the full validation split
(248,000 windows, k = 0), using each arm's own `checkpoint_best.pt`:
- **The network branch collapsed in all six arms.** With the physics term removed it predicts
  "no fault" on 99.94–100 % of windows. Its 26-class macro-F1 is 0.0256, exactly the trivial
  always-no-fault score.
- **On fault windows the fused prediction is PSM.** It agrees with argmin ρ on 87.7–98.0 % of
  fault windows. For the selected arm it agrees on 95.9 %, with accuracy 85.6 % vs PSM's 85.8 %.
- **The gain over PSM comes from no-fault windows.** For the selected arm, fused accuracy on
  no-fault windows is 87.0 % vs PSM's 49.8 %. The network contributes a constant "no fault" push,
  nothing else.

This is exactly the prediction of S4.3ᵇ: noise-free training gives β nothing to learn, and
PC-STGNN becomes "PSM with extra steps". It is recorded here as evidence for that prediction.

**Why λ₁ = 0.0333, λ₂ = 0 must not be carried into S4.4:**
- They were selected in the collapsed regime.
- The winner again sits on a grid edge, and λ₁ = 0 was never run, so the sweep cannot show that
  the λ₁ term helps at all.
- λ₂ = 0 won, but R4 needs a nonzero λ₂.

S4.3 is rerun under noise augmentation, gated on S4.3ᶜ.

## S4.3ᶜ — Collapse diagnostic and pre-registered collapse guard `[Added 2026-10-04, written before any noise-augmented run was started or viewed]`

### Diagnostic (step before the re-sweep)

**Arms.** λ₁ ∈ {0.0333, 0}, λ₂ = 0, β learned, `use_fusion=True`. λ₁ = 0 is the missing control:
fusion only, no physics-consistency loss.

**Training.**
- Seed 42, 5 epochs
- Noise augmentation exactly as S4.3ᵇ:
  - σ is drawn per window from {0, 5e-4, 2e-3, 1e-2} p.u. via `dropout/noise.py` (unchanged)
  - σ₀ in the residual equals the σ injected into that window
- Dropout augmentation stays on
- The k = 0 ρ cache is **off**: it was built noise-free and cannot serve a noisy window

`scripts/run_pcstgnn_noise_diagnostic.py`.

**Logged every epoch, full validation split, k = 0:**
- Network-only (physics term removed) 26-class macro-F1 and fraction of "no fault"
  predictions, on clean inputs
- Network-only macro-F1 at σ = 1e-2
- The standard fused validation F1

**Decision rule, read on each arm's final epoch:**
- **Fixed:** network-only F1 > 0.10 and no-fault fraction < 0.95.
- **Collapsed:** no-fault fraction > 0.99.
- **Unresolved:** anything else.

| Outcome | Action |
|---|---|
| ≥ 1 arm fixed | Proceed to the noise-augmented S4.3 rerun (below). |
| Both arms collapsed | The fault is in the fusion design, not the training data. Do **not** rerun S4.3 or start S4.4. Redesign the fusion first (an auxiliary network-only loss, PSM dropout during training, or a constraint on β). |
| Otherwise | Report; the author decides. |

**Why the 0.10 threshold.** It is ≈ 4× the trivial 0.0256. It is far below anything a learning
network shows by epoch 5: Paper-1 R0 seed 42 has val F1 0.641 at epoch 5 and 0.211 at epoch 1.

`[Corrected 2026-10-05]` **DIAGNOSTIC RUN: verdict UNRESOLVED for both arms; the author decides.**
Sources: `results/pcstgnn_noisediag/diagnostic_summary.json`, per-epoch CSVs in
`results/pcstgnn_noisediag_lam1_*/`. Measured on the full validation split, k = 0.

| Epoch | λ₁=0: fused F1 | λ₁=0: net-only F1 | λ₁=0: net "no fault" | λ₁=0.0333: fused F1 | λ₁=0.0333: net-only F1 | λ₁=0.0333: net "no fault" |
|---|---|---|---|---|---|---|
| 1 | 0.824 | 0.041 | 66.6 % | 0.836 | 0.036 | 75.2 % |
| 2 | 0.857 | 0.035 | 77.7 % | 0.799 | 0.040 | 65.8 % |
| 3 | 0.867 | 0.046 | 58.7 % | 0.823 | 0.043 | 58.8 % |
| 4 | 0.869 | 0.069 | 55.9 % | 0.819 | 0.053 | 57.3 % |
| 5 | **0.879** | **0.078** | **54.5 %** | 0.824 | 0.061 | 55.9 % |

- **Noise broke the hard collapse.** The network-only "no fault" share fell from 99.9–100 %
  (noise-free) to 54.5–55.9 %, close to the true base rate of 50 %. That clears the "collapsed"
  bar (> 0.99) by a wide margin.
- **The network still does not localize.** Network-only F1 is 0.078 (λ₁ = 0) and 0.061
  (λ₁ = 0.0333), below the 0.10 "fixed" bar. It rises about +0.01 per epoch. For scale,
  standalone Paper-1 R0 reached 0.641 by epoch 5. Localization is still offloaded to the physics
  term. Network-only F1 at σ = 1e-2 is about the same (0.064 / 0.061).
- **Under the collapse guard, both arms would be ineligible** (network-only F1 ≤ 0.10).
- **λ₁ > 0 is worse than λ₁ = 0 on both measures:** fused 0.824 vs 0.879, network-only 0.061 vs
  0.078. This matches the noise-free trend. It is a single seed at 5 epochs, recorded as an
  observation only.

### Test-split outputs written by the runner — quarantined `[Added 2026-10-05 08:37 IST]`

The runner always evaluated the test split at the end of training. So all six noise-free S4.3 runs
and both S4.3ᶜ runs wrote test-split outputs, although S4.3 and S4.3ᶜ are validation-only. Each run
wrote three files: `predictions_test.csv`, `confusion_matrix.csv` (test split) and `metrics.json`
(`test_graph`, `test_node_diagnostic`). All 24 files are now in
`results/_quarantine_test_outputs/`, with the original folder structure. Nothing was deleted.
The README there lists each file, its original path and its modification time.

**Use check (2026-10-05).** The check covered the code, the run logs, the documents and all 23
Claude Code session transcripts. It looked only for key names, never for values.
- **No decision used a test score.**
  - λ selection (`run_pcstgnn_tuning.py`) reads only `best_validation_graph_f1` and `beta_final`
    from `metrics.json`. Its curves and `fusion_decomposition_val_k0.json` use the validation split.
  - The S4.3ᶜ verdict (`run_pcstgnn_noise_diagnostic.py`) reads only `metrics_by_epoch.csv`.
  - `lambda_selection.json`, `fusion_decomposition_val_k0.json`, `diagnostic_summary.json` and both
    run logs contain no test field.
  - The CLI prints the run directory, not the metrics.
  - The audit scripts that read `predictions_test.csv` scan only the Paper-1 result folders.
- **No Claude session saw test content.** One `cat` of an S4.3 `metrics.json` on 2026-09-30 ran
  before that file existed, and its output holds no test field. The other transcript hits are file
  listings. The quarantine work itself read key and column names only.
- **Not fully attributable.** File access times show automated reads of the small files
  (`metrics.json`, `confusion_matrix.csv`). These happened in batches at 2026-10-04 21:04 and
  2026-10-05 08:17 / 08:27 IST, and skipped the 37 MB prediction files. The pattern matches a
  background editor or indexer process, and the output appears in no log or transcript. It cannot be
  shown to be that, so "no person saw them" is not claimed beyond the points above.

**Rule from now on.** Every tuning and diagnostic run must set `training.skip_test_eval: true`.
The test split is evaluated only in runs the protocol designates as test runs. S4.3ᵈ already sets it.

### Collapse guard for the S4.3 rerun and S4.4

**Selection.** A fusion-on arm is **ineligible** for λ selection, whatever its fused score, if its
checkpoint fails either test on the full validation split (k = 0, clean):
- network-only macro-F1 ≤ 0.10, or
- network-only no-fault fraction > 0.95

**S4.4 monitoring.** Network-only metrics are logged every epoch. A fusion-on arm (R5, R6) still
collapsed by these thresholds at the end of epoch 3 is aborted and reported, not completed.
**R8 is exempt**: it is noise-free R5, expected to degenerate by design (an oracle condition).
Arms without fusion (R0ⁿ, R1, R2, R3, R4, R7) cannot collapse this way, because their network
output *is* the model.

### Noise-augmented S4.3 rerun (only if the diagnostic says "fixed")

- **Grid:** λ₁ ∈ {0, 0.01, 0.0333, 0.1}, λ₂ = 0, then one λ₂ = 0.1 arm at the best eligible λ₁.
- **Fixed settings:** seed 42, 15 epochs, noise and dropout augmentation as above.
- **Selection:** validation F1 averaged over k, restricted to guard-eligible arms. The checkpoint
  rule within a run is unchanged (val F1 at k = 0).
- **Cost:** ≈ 5 × 13.9 h ≈ 70 GPU-h at the measured rate. The noise-free cache no longer applies,
  so each epoch costs about +6.7 min.

**Interpretation, stated in advance.** If λ₁ = 0 matches or beats every λ₁ > 0 under noise, the
physics-consistency loss is not contributing. That is reported as a finding, not treated as a
failure, and the S4.4 ablation is designed around it.

## S4.3ᵈ — Auxiliary network-only loss diagnostic `[Added 2026-10-05, written before any code change or run]`

**Why.** S4.3ᶜ was unresolved for both arms. Noise augmentation broke the hard collapse (network-only
"no fault" share ≈ 55 %), but network-only macro-F1 stayed at 0.06–0.08, below the 0.10 bar. The
network does not localize; it leaves localization to the physics term. An auxiliary loss on the
network-only logits must make the network localize without the physics term.

**Loss.**

    L_total = L_current + γ · CE(network-only logits, labels)

- `L_current` is the S4.3ᶜ objective (`models/common.py::pc_loss`), unchanged.
- The network-only logits are the backbone's node logits before the β·ρ fusion, taken from the
  same forward pass as the fused logits (`network_logits` in the collapse monitor).
- The CE uses the same labels and the same class weights as the main loss (unweighted
  `expanded_cross_entropy`).
- Config key `pc_loss.aux_net_loss_weight` (γ), default 0. No new model parameters. The fused
  prediction path and the evaluation code are unchanged.
- **Null test first:** with γ = 0, one fixed batch at seed 42 must give the same total loss and the
  same gradients as the old code path (tolerance 1e-7), with no training. A failure stops the step.

**Run settings.** The S4.3ᶜ setup with only these changes:
- λ₁ = 0, λ₂ = 0, γ = 1
- Noise augmentation on (as S4.3ᶜ), dropout augmentation on, β learned (init 0)
- 5 epochs, seed 42
- Validation split only; the test split is not evaluated

**Logged every epoch, full validation split:**
- Fused macro-F1 at k = 0
- Network-only macro-F1 at k = 0
- Network-only macro-F1 at σ = 1e-2
- Network-only "no fault" share
- Learned β
- Auxiliary loss value

Output: `results/pcstgnn_auxdiag/diagnostic_summary.json`.

**Decision rule, read at epoch 5:**
- **Fixed:** network-only macro-F1 at k = 0 > 0.10, and network-only "no fault" share ≤ 95 %.
  `[Amended 2026-10-05 08:34 IST — replaces "≤ 99 %"]` Amended 2026-10-05 08:34 IST before any
  S4.3ᵈ result was available. Reason: make the diagnostic rule equal to the selection guard.
- **Not fixed:** any other result.
- Fused macro-F1 is for information only. It is compared to 0.879 (S4.3ᶜ, λ₁ = 0, epoch 5).

`[Result 2026-10-05]` **NULL TEST PASSED.** `scripts/run_pcstgnn_aux_null_test.py`, result in
`results/pcstgnn_auxdiag/null_test/null_test_result.json`. One fixed seed-42 training batch was
used, with B = 32, k = 0 and all four σ levels present. With γ = 0 the total loss equals the
old-code loss exactly (3.4546587467, difference 0). All 17 parameter gradients are identical
(max |Δ| = 0). The old path was recorded before the code change and was bit-reproducible.

`[Result 2026-10-05]` **DIAGNOSTIC RUN: verdict FIXED** under the amended rule (≤ 95 %).
- Script: `scripts/run_pcstgnn_aux_diagnostic.py`
- Summary: `results/pcstgnn_auxdiag/diagnostic_summary.json`
- Per-epoch CSV: `results/pcstgnn_auxdiag_gamma1/pcstgnn/seed_00042/metrics_by_epoch.csv`
- Measured on the full validation split (248,000 windows) at k = 0.
- `skip_test_eval: true`; no test-split file was written.

| Epoch | Fused F1 | Net-only F1 | Net-only F1 @ σ=1e-2 | Net "no fault" | β | Aux loss (train) |
|---|---|---|---|---|---|---|
| 1 | 0.855 | 0.157 | 0.153 | 55.8 % | 0.027 | 1.691 |
| 2 | 0.839 | 0.249 | 0.249 | 53.8 % | 0.032 | 1.350 |
| 3 | 0.868 | 0.386 | 0.353 | 54.1 % | 0.031 | 1.176 |
| 4 | 0.856 | 0.524 | 0.478 | 53.4 % | 0.036 | 1.036 |
| **5** | **0.879** | **0.571** | **0.512** | **53.5 %** | **0.025** | **0.939** |
| S4.3ᶜ λ₁ = 0, epoch 5 | 0.879 | 0.078 | 0.064 | 54.5 % | 0.035 | — |

- **Rule check at epoch 5.** Network-only F1 is 0.571 > 0.10 and the "no fault" share is
  53.5 % ≤ 95 %, so the verdict is **fixed**. The old ≤ 99 % rule gives the same verdict. The
  driver script still encodes 0.99; the JSON keeps that output as `verdict_script_rule`.
- **The auxiliary loss makes the network localize.** Network-only F1 is 7.3× the S4.3ᶜ value
  and was still rising at epoch 5 (+0.05 in the last epoch). It stays well above the trivial 0.0256.
  It is still below standalone Paper-1 R0 at epoch 5 (0.641).
- **Fused F1 is unchanged** from S4.3ᶜ (0.8794 vs 0.8788). It was not monotone across epochs.
- **Network-only F1 under σ = 1e-2 falls behind the clean value from epoch 3.** At epoch 5 it
  is 0.512 vs 0.571 clean.
- **β stays small** (0.025–0.036), as in S4.3ᶜ.
- Single seed, 5 epochs. Recorded as a diagnostic, not as a tuning result.

**Compute.** 4.32 GPU-h wall-clock (03:00–07:19 UTC). Of this, 3.55 h is the training loop and
the rest is the per-epoch validation passes.

## S4.3ᵉ — Noise-augmented λ tuning with auxiliary loss `[Pre-registered 2026-10-05, written before any S4.3ᵉ code change or run]`

**Why.** S4.3ᵈ verdict is "fixed": with γ = 1 the network localizes on its own (network-only
F1 0.571 at epoch 5, "no fault" share 53.5 %). This replaces the noise-augmented S4.3 rerun of
S4.3ᶜ. **γ = 1 is now a fixed part of the PC-STGNN design. It is not tuned.**

**Fixed settings (every arm).**
- γ = `pc_loss.aux_net_loss_weight` = 1
- Noise augmentation on, exactly as S4.3ᵇ/S4.3ᶜ (σ per window from {0, 5e-4, 2e-3, 1e-2} p.u.,
  σ₀ = injected σ; k = 0 ρ cache off in training)
- Dropout augmentation on (k ∈ {0, 1, 2, 5})
- 15 epochs, seed 42, β learned (init 0), `use_fusion=True`
- `training.skip_test_eval: true`. Validation split only; the test split is never loaded
- Base config: `config/pcstgnn_auxdiag/gamma1.yaml`, changing only λ₁, λ₂ and epochs

**Grid.** λ₁ ∈ {0, 0.01, 0.0333, 0.1}, λ₂ = 0. Run one at a time in that order.

**Extension rules, applied after the 4 grid runs, in this order.**
1. If λ₁ = 0.1 is the best arm, add one run at λ₁ = 0.3, λ₂ = 0.
2. Then add one run at λ₂ = 0.1 at the best λ₁ (after step 1).

"Best" in both rules means the best **guard-valid** arm under the selection rule. If no arm is
valid at that point, no extension is run (see "No selection").

**Logged every epoch, full validation split (clean inputs):** fused macro-F1 at k = 0,
network-only macro-F1 at k = 0, network-only macro-F1 at σ = 1e-2, network-only "no fault"
share, learned β, auxiliary loss (train CE on network-only logits, averaged over the epoch).

**Early stop.** At the end of epoch 3, if the network-only "no fault" share is > 95 % or the
network-only macro-F1 is ≤ 0.10, the run is stopped and recorded as **collapsed**. A collapsed
run is not valid for selection. The next run is started.

**Checkpoint rule (within a run).** Keep the epoch with the best validation fused macro-F1 at
k = 0 (unchanged Paper-1 rule).

**Selection rule (across arms).** Select the arm with the best validation fused macro-F1
averaged over k ∈ {0, 1, 2, 5, 10}, scored on its checkpoint with the S4.3 method
(`evaluate_dropout_curve`, 3 mask seeds, full validation split). Ties (equal to 4 decimals): the
arm with the smaller λ₁, then the smaller λ₂.

**Collapse guard.** An arm is not valid for selection if, at its selected epoch (full
validation split, k = 0, clean), the network-only macro-F1 is ≤ 0.10 or the network-only
"no fault" share is > 95 %.

**No selection.** If no arm is valid, the result is "no selection". Stop and wait for the
author's decision. S4.4 does not start.

**Scoring outputs.** `results/pcstgnn_tuning_aux/lambda_selection.json` and
`results/pcstgnn_tuning_aux/fusion_decomposition_val_k0.json` (method of
`results/pcstgnn_tuning/fusion_decomposition_val_k0.json`, extended: for fault and no-fault
windows separately, accuracy of fused / network-only / PSM, and % fused = PSM, % fused =
network-only). Run folders: `results/pcstgnn_tuning_aux/<arm>/`.

**Interpretation, stated in advance.** As in S4.3ᶜ: if λ₁ = 0 matches or beats every λ₁ > 0,
the physics-consistency loss is not contributing. That is a finding, not a failure.

**Cost (estimate).** S4.3ᵈ ran ≈ 0.86 GPU-h per epoch including the per-epoch validation
passes, so ≈ 13 GPU-h per 15-epoch arm. 5 arms ≈ 65 GPU-h, 6 arms ≈ 78 GPU-h, plus ≈ 0.3 h
scoring per arm.

## S4.3ᵇ — Noise augmentation is the default training condition `[Added 2026-09-23]`

**Every arm below is trained with measurement-noise augmentation on.** Noise-free training is now
the *ablation*, not the default. The reason is `04_STAGE_C_gonogo.md` S2.0ᵃ consequence 4: on
noise-free data ρ_true = 0 and `softmax(−ρ)` is one-hot, so β learns to trust physics completely,
the network's contribution goes to zero, and PC-STGNN degenerates into PSM with extra steps.
There is nothing for the fusion to learn. Training under noise is what gives the physics prior a
finite, informative dynamic range — and it is the realistic operating point besides.

Three implementation requirements come with it:

- **σ is drawn per window** from Paper 1's existing grid **including 0** — σ ∈ {0, 5e-4, 2e-3,
  1e-2} p.u. — exactly as `DropoutAugmentedLoader` draws k, so the model sees clean and noisy
  windows and the clean case is not abandoned. Reuse `dropout/noise.py` unchanged; the
  augmentation wrapper mirrors `dropout/training.py`.
- **σ₀ in the residual denominator equals the σ actually injected into that window**
  (`PC-STGNN-PLAN.md` §6). The residual must not be evaluated at a σ₀ the observation does not
  have, or the physics prior is mis-scaled in exactly the direction that hides the effect.
- **Dropout augmentation stays on** — this is a dropout-aware backbone and that is not the
  variable under test. Noise augmentation is added *on top of* the Paper-1 dropout-aware recipe.

**This forces a new reference run, and it is not optional.** R0 as it exists on disk
(`results/dropout_aware/rgsage_max/`) was trained without noise augmentation. Comparing a
noise-augmented R5 against it would confound noise augmentation with physics fusion and break the
project's first rule (one variable at a time). **R0ⁿ — the Paper-1 dropout-aware recipe plus noise
augmentation, no physics — is the paired reference for every arm in S4.4, at 3 seeds.** The
existing R0 remains the Paper-1 anchor and the S4.1 null-test target; it is not the ablation
baseline any more.

## S4.4 — Core ablation — 30 epochs, everything else frozen

`[Corrected 2026-09-23]` All arms trained with noise augmentation (S4.3ᵇ); R0ⁿ replaces R0 as the
paired reference; R8 added as the ablation of the new default.

`[Added 2026-10-05, S4.3ᵉ]` All PC-STGNN arms use the auxiliary network-only loss with γ = 1
(S4.3ᵈ).

| # | Variant | Answers the objection | Seeds |
|---|---|---|---|
| R0 | Dropout-aware RGSAGE-Max, no noise aug | Paper-1 anchor, S4.1 null target (already trained) | 3 (done) |
| R0ⁿ | **R0 + noise augmentation, no physics** | **the paired reference — isolates noise aug from physics** | **3 (new)** |
| R1 | + type head only | "the auxiliary task is doing it" | 1 |
| R2 | + learned MLP decoder on D, parameter-matched | "any decoder / masked autoencoding would do it" | 3 |
| R3 | + λ₁ (residual on A) | physics as a training constraint | 1 |
| R4 | + λ₂ (residual on held-out D) | does held-out consistency add over in-sample? | **3** |
| R5 | + β fusion — full PC-STGNN | physics as evidence at test time | 3 |
| R6 | R5 with a shuffled location→signature map | "it's regularisation, not physics" | 3 |
| R7 | Size-matched wider RGSAGE-Max | "it's just a bigger model" | 1 |
| R8 | **R5 trained noise-free** | **the ablation of the new default — expected to degenerate: β saturates, network contribution → 0** | **1** |
| B2ᵃ | learned R̂ head instead of profiling | shows why profiling beats regression on OOD R | 1 |

`[Corrected 2026-09-22]` Seed counts follow the S0.5 rule (one run ≈ 7 h). Promote R1/R3/R7 to
3 seeds only if Gate 4 passes and budget remains.

`[Corrected 2026-09-23]` **Seed swap R4 ↔ R7.** λ₂ is the novel loss term and R4 is the only arm that isolates it, so R4 gets 3 seeds; R7 (size-matched wider backbone) answers a one-line objection with a one-row answer and gets 1. Without error bars on R4 the λ₂ → λ₁ comparison — the one that says whether held-out
physics consistency adds anything over in-sample — cannot be read at all, whereas R7 only has to
land on the right side of a large gap.

R6 and R2 carry the burden of proof. The target pattern is F1_real > F1_none ≈ F1_shuffled.

R7 matters less than it looks — the physics path adds three scalars — but it is a one-line
objection with a one-row answer, so run it.

B2ᵃ is expected to fail on OOD R: a head trained on {0.1, 1, 10} Ω collapses to a three-way
classifier. That failure is a result worth a table row.

R8 is a negative control and a mechanism check: if the fusion degenerates without noise, that is
the sharpest available confirmation of S2.0ᵃ consequence 4. `[Corrected 2026-09-23 — 1 seed, not
3.]` Under the degeneracy ruling, noise-free R5 is an **oracle condition**: its job is to be a
labelled upper bound, and an upper bound does not need three-seed error bars the way a contested
comparison does. One seed states it; the ~14 GPU-h freed goes to a **second retrained
model-mismatch level** (`08_STAGE_G_eval.md` S6.2), because mismatch is now a primary axis and
one retrained point is an anecdote while two make a trend.

**Cost.** `[Corrected 2026-09-23]` R0ⁿ 3 + R1 1 + R2 3 + R3 1 + R4 3 + R5 3 + R6 3 + R7 1 + R8 1
+ B2ᵃ 1 = **20 new runs × 7 h ≈ 140 GPU-h** (R7 wider: somewhat more). R0's 3 runs stay done and
are no longer part of this count. Net ≈ +28 GPU-h over the pre-ruling plan: the paired noise
reference R0ⁿ (+21 h) and R8 at one seed (+7 h). **R0ⁿ never loses seeds** — it is the
comparison's paired reference, and trimming it would put the noise-augmentation confound straight
back into every physics claim.

**Table produced.** T7 — core ablation.
**Figure produced.** F-E2 — ablation bar chart with seed spread (optional if T7 is clear).

## GATE 4

- [x] S4.1 null test exact — PASSED 2026-09-30, byte-identical over 30 epochs (see S4.1)
- [x] S4.2 gradients non-zero on both λ terms; table receives none — PASSED 2026-09-30, 12/12
      applicable checks (see S4.2)
- [ ] **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
      across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ)

**If R5 ≈ R0 on validation.** Do not proceed to Stage F expecting a rescue. Go straight to
Stage G, run the OOD and noise evaluations, and write the "physics buys generalisation, not
in-distribution accuracy" paper.
