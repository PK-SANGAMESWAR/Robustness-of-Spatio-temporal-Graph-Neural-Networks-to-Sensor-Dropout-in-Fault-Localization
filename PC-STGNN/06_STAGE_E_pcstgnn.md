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

`[Result 2026-10-08]` **S4.3ᵉ RUN: SELECTED λ₁ = 0, λ₂ = 0.1.**
- Script: `scripts/run_pcstgnn_tuning_aux.py`. Configs: `config/pcstgnn_tuning_aux/`.
- Outputs: `results/pcstgnn_tuning_aux/lambda_selection.json`,
  `results/pcstgnn_tuning_aux/fusion_decomposition_val_k0.json`, per-arm folders with
  `metrics_by_epoch.csv` and `wallclock.json`. Log: `results/pcstgnn_tuning_aux/run.log`.
- Validation split only. Every run has `test_evaluated: false`; no test-split file was written.
- 5 arms run: the 4 grid arms, then the λ₂ = 0.1 arm at λ₁ = 0. Extension rule 1 did not fire
  (λ₁ = 0.1 was the worst arm, not the best). No arm was early-stopped; every arm passed the
  epoch-3 check and the collapse guard.
- Runner change: `training.collapse_abort` (epoch-3 early stop) added to
  `experiments/runner.py`, off unless a config sets it.

**Arms.** Selected epoch = best validation fused F1 at k = 0. "Avg over k" = selection score
(fused F1, mean over k ∈ {0, 1, 2, 5, 10}, 3 mask seeds). Network-only F1, "no fault" share and β
are at the selected epoch (full validation split, k = 0, clean).

| λ₁ | λ₂ | Best epoch | Fused F1, k=0 | Fused F1, avg over k | Fused F1, k=10 | Net-only F1 | Net "no fault" | β | Guard |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 10 | 0.8932 | 0.8778 | 0.8494 | 0.7375 | 52.8 % | 0.0222 | valid |
| 0.01 | 0 | 12 | 0.8902 | 0.8777 | 0.8526 | 0.7876 | 52.7 % | 0.0286 | valid |
| 0.0333 | 0 | 12 | 0.8581 | 0.8523 | 0.8426 | 0.7174 | 52.9 % | 0.0342 | valid |
| 0.1 | 0 | 14 | 0.8225 | 0.8180 | 0.8146 | 0.7409 | 52.8 % | 0.0554 | valid |
| **0** | **0.1** | **12** | **0.8937** | **0.8797** | **0.8534** | **0.7569** | **52.7 %** | **0.0284** | **valid** |

**Fusion decomposition** (`fusion_decomposition_val_k0.json`). Full validation split, k = 0,
clean, selected checkpoint, ρ from the k = 0 slice of `cache/rtilde_val.npy`. 124,000 fault and
124,000 no-fault windows. The decomposition's fused F1 equals each arm's logged checkpoint F1
exactly.

| λ₁ | λ₂ | Fault: acc fused | net-only | PSM | fused = PSM | fused = net | No-fault: acc fused | net-only | PSM | fused = PSM | fused = net |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 86.8 % | 71.1 % | 85.8 % | 91.5 % | 78.1 % | 100.0 % | 100.0 % | 49.8 % | 49.8 % | 100.0 % |
| 0.01 | 0 | 86.9 % | 76.2 % | 85.8 % | 92.4 % | 82.4 % | 99.4 % | 100.0 % | 49.8 % | 50.4 % | 99.4 % |
| 0.0333 | 0 | 86.5 % | 69.3 % | 85.8 % | 93.6 % | 74.6 % | 86.8 % | 100.0 % | 49.8 % | 61.1 % | 86.8 % |
| 0.1 | 0 | 86.3 % | 71.2 % | 85.8 % | 94.7 % | 75.8 % | 74.2 % | 100.0 % | 49.8 % | 72.1 % | 74.2 % |
| **0** | **0.1** | **86.9 %** | **73.3 %** | **85.8 %** | **92.7 %** | **79.7 %** | **99.9 %** | **100.0 %** | **49.8 %** | **49.9 %** | **99.9 %** |

**Selection.** λ₁ = 0, λ₂ = 0.1 has the best validation fused F1 averaged over k: 0.8797 vs
0.8778 for λ₁ = λ₂ = 0, a margin of +0.0019. It is guard-valid. This is a single seed at
15 epochs; the margin is not a tested difference.

**Observations** (single seed; recorded as observed, not as claims):
- **λ₁ hurts monotonically.** Avg over k: 0.8778 (λ₁ = 0) ≈ 0.8777 (0.01) > 0.8523 (0.0333) >
  0.8180 (0.1). By the interpretation stated in advance, **the physics-consistency loss λ₁ is not
  contributing.** λ₁ = 0 is not beaten by any λ₁ > 0.
- **The λ₁ loss shows up in no-fault windows.** Fused accuracy on no-fault windows falls
  100.0 % → 99.4 % → 86.8 % → 74.2 % as λ₁ rises. Fault-window accuracy stays at 86.3–86.9 %.
  The network alone is right on 100 % of no-fault windows in every arm. Larger λ₁ goes with a
  larger β (0.022 → 0.055), and the physics term then overrides correct "no fault" calls (PSM is
  right on only 49.8 % of no-fault windows).
- **The collapse is fixed by the guard's definition, but on fault windows the fused prediction
  still follows PSM.** It agrees with PSM on 91.5–94.7 % of fault windows (95.9 % in the
  noise-free S4.3 selected arm). Fused fault accuracy (86.3–86.9 %) is about 1 point above PSM
  alone (85.8 %) and well above the network alone (69.3–76.2 %). At clean k = 0 the network
  supplies the "no fault" decision, and localization is still mostly physics. This is clean
  k = 0 only; S2.0ᵃ says the fusion question is read under noise and dropout, not here.
- **λ₂ = 0.1 is neutral to slightly positive** at λ₁ = 0: +0.0019 avg over k, +0.004 at k = 10.
  R4 needs a nonzero λ₂, and this result does not argue against one.
- **The network branch alone was still rising at epoch 15** in every arm: λ₁ = 0 reached 0.772 at
  epoch 15 vs 0.738 at its selected epoch 10. The checkpoint rule selects on the fused F1, so
  the selected network-only figures are below each arm's final-epoch value.

**Compute.** Wall-clock on one T1000, from `wallclock.json` and file times:

| Arm | Training incl. per-epoch validation (h) | Selection scoring (h) | Total (h) |
|---|---|---|---|
| λ₁ = 0, λ₂ = 0 | 13.03 | 1.49 | 14.52 |
| λ₁ = 0.01, λ₂ = 0 | 13.01 | 1.49 | 14.51 |
| λ₁ = 0.0333, λ₂ = 0 | 13.04 | 1.49 | 14.53 |
| λ₁ = 0.1, λ₂ = 0 | 13.01 | 1.49 | 14.51 |
| λ₁ = 0, λ₂ = 0.1 | 13.03 | 1.49 | 14.52 |
| Fusion decomposition (all arms) | — | 0.08 | 0.08 |
| **Total** | **65.12** | **7.54** | **72.67** |

72.67 GPU-h from 2026-10-05 07:32 UTC to 2026-10-08 08:13 UTC. The pure training loop is about
10.7 h per arm; the rest of the 13.0 h is the per-epoch validation passes. Scoring took about
1.5 h per arm, not the estimated 0.3 h, because ρ is computed live for every k > 0 × 3 mask seeds.

~~**S4.4 not started.** It waits for the author's decision.~~ *(Stale as of 2026-10-15 — S4.4's
Gate 4 runs started 2026-10-09, were amended to 30 epochs and restarted 2026-10-12, and are in
progress; see "S4.4 — Frozen settings for Gate 4 runs" and the two amendments below. Kept, not
deleted, as the record of this section's own state when S4.3ᵉ was written.)*

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

### S4.4 — Frozen settings for Gate 4 runs `[Pre-registered 2026-10-09, before any Gate 4 run was started]`

Frozen before training, per `PC-STGNN/S4.4_plan_draft.md` (2026-10-09 revision). Covers only the
two Gate 4 arms, R0ⁿ and R5 — not the rest of the S4.4 table.

**Epoch count: 30.** `[Amended 2026-10-12 14:40 IST / 2026-10-12T09:10:59Z — superseding the
2026-10-09 value below, not deleting it]`

~~**Epoch count: 20**, decided by a fixed rule applied to the S4.3ᵉ per-epoch record (fused F1 at
k = 0, epoch 10 vs epoch 15, `results/pcstgnn_tuning_aux/<arm>/pcstgnn/seed_00042/metrics_by_epoch.csv`):~~

| Arm (λ₁, λ₂) | Fused F1 @ epoch 10 | Fused F1 @ epoch 15 | Δ (15 − 10) |
|---|---|---|---|
| 0, 0 | 0.8932 | 0.8871 | −0.0061 |
| 0.01, 0 | 0.8797 | 0.8851 | +0.0054 |
| 0.0333, 0 | 0.8520 | 0.8338 | −0.0182 |
| 0.1, 0 | 0.8147 | 0.8142 | −0.0005 |
| 0, 0.1 (selected) | 0.8849 | 0.8853 | +0.0004 |

~~Mean signed Δ over the 5 arms = **−0.0038**. Rule: mean Δ < 0.005 ⟹ 20 epochs, else 30. −0.0038
< 0.005, so **20 epochs**. Caveat carried from the plan draft: this signed mean is pulled toward
zero by cancellation (one arm moved −0.0182; the mean of *absolute* differences is 0.0061, which
would flip the rule to 30 under that reading). The rule as specified is on the signed difference;
applied literally here.~~ *(2026-10-09 value, replaced below — kept, not deleted.)*

**Amended 2026-10-12 14:40 IST / 2026-10-12T09:10:59Z. No Gate 4 comparison result existed.
Reason: the deadline was removed, and Phase-1 used 30 epochs.** The 2026-10-09 rule above was a
time-driven rule, picked under deadline pressure that no longer applies; the new choice (30,
matching Phase-1) does not depend on any Gate 4 result — R5 seed 1 had reached only epoch 3 of
20, un-scored, when this was decided. For the record, not as a factor in this decision: R0ⁿ
seed 1's own 20-epoch run (now archived at `results/s4_4_pilot_20ep/`) had best epoch 18
(validation graph-F1 0.8126004083285524) and val_graph_f1 = 0.7293 (epoch 10), 0.7810
(epoch 15), 0.8110 (epoch 20) — still rising through epoch 20, consistent with 30 epochs giving
more room, but this was not used to make the epoch-count decision above.

**R0ⁿ settings** (`config/s4_4/r0n.yaml`): PCSTGNN, `use_fusion=False`, no type head, no decoder,
no table. `pc_loss.enabled=true` purely for the noise-augmentation loader plumbing (noise
augmentation is wired only through the `pc_loss` path, `experiments/runner.py:258-259`) —
`lambda1=lambda2=aux_net_loss_weight=0`, i.e. γ=0, consistent with the design rule "γ applies
only when fusion is ON" (`S4.4_plan_draft.md` Part 2). Noise augmentation on, σ ∈ {0, 5e-4, 2e-3,
1e-2} p.u. Dropout-aware training on, k ∈ {0, 1, 2, 5}. `gnn_layers=4`. AdamW, batch 32, lr
0.001, weight decay 0.0001, grad-clip 1.0, warmup 3 epochs. 95,258 parameters (measured,
byte-identical backbone to RGSAGE-Max).

**R5 settings** (`config/s4_4/r5.yaml`): PCSTGNN, `use_fusion=True` (β learned, init 0), real
physics table, `λ₁=0, λ₂=0.1, γ=1` — the S4.3ᵉ-selected point. Noise augmentation and
dropout-aware training identical to R0ⁿ. 95,259 parameters (measured) — the one-scalar
difference from R0ⁿ is exactly `beta`.

**Evaluation grid** (both arms, every seed, scored after training on that seed's checkpoint):
k ∈ {0, 1, 2, 5, 10} × σ ∈ {0, 5e-4, 2e-3, 1e-2} p.u., 3 mask seeds per k > 0 (k = 0 computed once
per σ, not repeated — matches `dropout/evaluator.py::evaluate_dropout_curve`'s own convention).
**Fusion decomposition at every (k, σ) cell** for R5 (fused / network-only / PSM macro-F1,
`fused_eq_psm`, `fused_eq_network_only`), method of
`results/pcstgnn_tuning_aux/fusion_decomposition_val_k0.json` extended off the k=0-clean-only
cache to the live ρ path at every cell. R0ⁿ has no physics table, so only fused(=network-only)
F1 is reported per cell (`S4.4_plan_draft.md` Part 3: "the decomposition collapses... PSM is
shared/arm-independent").

**Early stop (collapse guard), applied to both arms:** at the end of epoch 3, if the
network-only "no fault" share is > 95% or the network-only macro-F1 is ≤ 0.10, the run is stopped
and recorded as **collapsed**. Wired via `training.collapse_abort` (epoch=3,
max_class0_frac=0.95, min_net_f1=0.10) + `pc_loss.log_network_only=true`
(`experiments/runner.py:383-387`). A collapsed run is not valid and stops the Gate 4 queue (see
below) — it is not retried automatically.

**`training.skip_test_eval: true` on every run.** The test split is never loaded by
`runner.run()` for either arm — Gate 4 and the evaluation grid above both read the validation
split only.

**Gate 4 rule, quoted verbatim from the existing checklist below, unchanged by this
pre-registration:**
> - [ ] **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
>       across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ)

**Gate 4 completion estimate (training + scoring, 20 epochs, 3 seeds each):**

| Component | Per-run | ×3 seeds |
|---|---|---|
| R0ⁿ training (20 epochs, measured rate) | 4.36 h | 13.07 h |
| R0ⁿ scoring (full grid, measured: 2% sample x50) | 0.63 h | 1.88 h |
| R5 training (20 epochs, 52 min/epoch given rate) | 17.33 h | 52.00 h |
| R5 scoring (full grid, measured: 2% sample x50) | 5.93 h | 17.78 h |
| **Gate 4 total** | — | **84.72 h ≈ 3.53 days** |

Scoring costs are measured, not placeholders: 2% of the validation split (4,960 of 248,000
windows) scored across the full (k, σ) grid with 3 mask seeds and (for the physics
configuration) decomposition at every cell, then scaled x50 to the full split
(`results/s4_4_timing/timing.json`, keys `scoring_physics_2pct` / `scoring_nonphysics_2pct`).
Physics scoring costs ≈9.5x the non-physics rate — confirms the ρ computation, not the backbone
forward pass, dominates scoring cost, consistent with the first-draft text's "ρ is computed live
for every k > 0 x 3 mask seeds" explanation of the original S4.3ᵉ scoring cost.

### S4.4 — Tooling incident `[2026-10-12, after R0ⁿ seed 1 training]`

**a. Failure.** After R0ⁿ seed 1 finished training cleanly (20 epochs, best validation
graph-F1 = 0.8126, not collapsed), the scorer (`scripts/score_s4_4_gate4_run.py`) crashed with
`torch.OutOfMemoryError` on the 8GB T1000, inside `apply_measurement_noise_batch`. The Gate 4
queue stopped itself at position 1 per its own exception handler — it was not restarted without
authorisation.

**b. Cause and fix.** The scorer cached the entire validation split (248,000 windows) as GPU
tensors up front, then built a second full-split "noised" copy per σ level inside the loop —
≈3 GB held twice over, on an 8 GB device, plus model/activation memory. This inverted the
pattern `dropout/evaluator.py::evaluate_dropout_curve` already uses and documents the reason
for: loop the (k, offset) — now (σ, k, offset) — combinations *inside* a single streaming pass
over the `DataLoader`, so only the current 32-window batch is ever resident on the GPU. Fixed by
rewriting the scorer to that single-pass structure, and separately made every window's noise
and dropout-mask draw depend only on `(level, mask-seed offset, that window's fixed position in
`dataset.records`)` — never on batch order, batch size, or shared RNG state (each draw gets its
own fresh `Generator`/`Random`). One side effect of the fix: noise now varies by mask-seed
offset even at k = 0 (the frozen seed formula ties noise to the offset unconditionally), so the
old "k = 0 computed once per σ" dedup no longer applies — a deliberate, documented behavior
change, not a regression. The orchestrator (`scripts/run_s4_4_gate4.py`) was also made
resumable (skip training only if `checkpoint_best.pt` + `metrics.json` exist *and* the saved
`config.yaml` matches `config/s4_4/<arm>.yaml` exactly; skip scoring only if
`scoring_full_grid.json` exists) and now releases GPU memory (`gc.collect` +
`torch.cuda.empty_cache`) after every training and scoring call.

**c. Equivalence test result.** R0ⁿ seed 1's checkpoint, first 2% of the validation windows
(4,960 of 248,000), every one of the 60 (σ, k, mask-seed-offset) cells, old cache-then-loop
structure vs. the new single-streaming-pass structure, **both using the identical new per-window
seed formula** (isolating the structural change from any seeding change): **max |Δ F1| = 0.0
across all 60 cells** (tolerance was 1e-6). Separately, the new scorer run on the **full**
validation split at σ = 0, k = 0 gives macro-F1 = 0.8126004083285524, identical to the logged
`best_validation_graph_f1` to full float precision (diff = 0.0). Both checks passed; Part 3 of
the fix task was not blocked.

**d. No pre-registered setting changed.** R0ⁿ and R5's λ₁, λ₂, γ, β, fusion, noise grid, dropout
grid, epoch count (20), collapse guard, and `skip_test_eval` are exactly as frozen above. Only
the scoring *implementation* and the orchestrator's resume/cleanup behavior changed.

~~**Gate 4 completion estimate, recomputed 2026-10-12** (training + scoring, 20 epochs, 3 seeds
each; R0ⁿ's rate is now a direct measurement from its completed run, not an extrapolation):~~

| Component | Per-run (20 ep, superseded) | Basis |
|---|---|---|
| R0ⁿ training | 5.87 h | measured (17.6 min/epoch × 20, R0ⁿ seed 1's actual run) |
| R0ⁿ scoring | 0.95 h | **measured** (R0ⁿ seed 1's actual full-grid scoring, new scorer) |
| R5 training | 17.33 h | given rate (52 min/epoch × 20), not yet re-measured under this queue |
| R5 scoring | 8.95 h | **estimate**, not yet measured — old 2%×50 physics estimate (5.93 h) scaled by the real/old-estimate ratio observed for R0ⁿ's scoring (1.51×), since the new scorer's per-window determinism loop costs more than the 60/52 cell-count ratio alone suggests (that gives 6.84 h; 8.95 h is used as the more honest number pending a real measurement at R5 seed 1) |

~~**Total, all 6 runs:** R0ⁿ seed 1 already complete (6.81 h actual) + 5 remaining runs (R5×3 at
26.28 h each + R0ⁿ×2 at 6.81 h each) = **99.30 GPU-h ≈ 4.14 days total, ≈92.49 GPU-h ≈ 3.85 days
remaining from 2026-10-12** → remaining-work completion **≈ 2026-10-16**, if uninterrupted.~~
*(Superseded below by the epoch-count amendment, before this estimate could be refined with a
real R5 measurement — kept, not deleted.)*

**Gate 4 completion estimate, recomputed 2026-10-12 after the 30-epoch amendment.** Scoring
costs are unchanged by the epoch-count change (scoring runs once per finished checkpoint,
independent of how many epochs it took to get there); only training time scales with epochs.
The queue restarts from zero (the 20-epoch R0ⁿ seed 1 and partial R5 seed 1 runs were moved to
`results/s4_4_pilot_20ep/`, not reused), so there is no "already done" credit this time.

~~| Component | Per-run (30 ep) | Basis |
|---|---|---|
| R0ⁿ training | 8.80 h | 17.6 min/epoch × 30 (measured rate) |
| R0ⁿ scoring | 0.95 h | measured (unchanged by epoch count) |
| R5 training | 26.00 h | 52 min/epoch × 30 (given rate) |
| R5 scoring | 8.95 h | estimate (unchanged by epoch count; still not yet directly measured) |
| **R0ⁿ total** | **9.75 h** | — |
| **R5 total** | **34.95 h** | — |
| **Gate 4 total (6 fresh runs)** | — | **3 × 9.75 + 3 × 34.95 = 134.10 GPU-h ≈ 5.59 days** |~~

*(Superseded below, 2026-10-15, once real measurements existed — kept, not deleted. Note: the
0.95 h R0ⁿ-scoring figure above was itself stale even when superseded — it was carried over from
the 20-epoch pilot's single scoring run, not a measurement of the 30-epoch queue.)*

**Updated with real measurements, 2026-10-15** (3 of 6 Gate 4 runs complete at this point: R0ⁿ
seed 1, R5 seed 1, R0ⁿ seed 2 — all trained and scored; R5 seed 2 in progress):

| Component | Per-run (30 ep) | Basis |
|---|---|---|
| R0ⁿ training | 8.80 h | **measured**, avg of 2 runs (31650.6 s, 31711.1 s) |
| R0ⁿ scoring | 1.35 h | **measured**, avg of 2 runs (4843.9 s, 4848.6 s) — not 0.95 h; see note above |
| R5 training | 26.14 h | **measured**, 1 run (94092.2 s) |
| R5 scoring | 7.45 h | **measured**, 1 run (26817.9 s) — came in under both the 8.95 h estimate and the 12.68 h overrun-adjusted projection |
| **R0ⁿ total** | **10.15 h** | — |
| **R5 total** | **33.59 h** | — |
| **Gate 4 total (6 runs)** | — | **3 × 10.15 + 3 × 33.59 = 131.2 GPU-h ≈ 5.47 days** |

Starting from 2026-10-12 19:14 UTC (R0ⁿ seed 1's actual start) → completion **≈ 2026-10-18,
early hours UTC**, consistent with the live per-run tracking kept in chat status updates.

### S4.4 — Frozen settings for the remaining arms (R1, R2, R3, R4, R6, R8, R5-γ0) `[Pre-registered 2026-10-22 23:32 IST / 2026-10-22T18:02:16Z, decisions D1–D9, full record and GPU-hour recomputation: PC-STGNN/S4.4_remaining_arms_plan.md § 11]`

No run in this subsection has been launched. These settings are frozen before any of them starts.

- **R6 (D1).** `shuffle_seed=6` — one permutation, shared by all 3 training seeds. Verified a
  full derangement of the 25 locations (no location keeps its own signature) with class 0 fixed,
  both by direct computation and against the real `PCSTGNN` class. Record:
  `config/s4_4/r6_shuffle_permutation.json`. Otherwise identical to `r5.yaml` (λ₁=0, λ₂=0.1, γ=1,
  fusion on, noise + dropout augmentation on, 30 epochs). 3 seeds.
- **R5-γ0 (D2).** Exempt from the collapse guard, same mechanism as R8: no
  `training.collapse_abort` key, so the run always completes all 30 epochs even if the
  network-only branch is still collapsed at epoch 3. γ=0 is its own defining feature (unaffected
  by the design rule "γ applies only when fusion is ON" — R5-γ0 has fusion on and γ=0 by choice,
  not by that rule). 1 seed.
- **R3, R4 (D3).** The runner fix: `fault_localization/experiments/runner.py::_model` now loads
  the physics table whenever `use_fusion` is true **or** `pc_loss.enabled` and (λ₁>0 or λ₂>0) —
  previously gated on `use_fusion` alone, so a fusion-off, λ-on config raised
  `KeyError("pc_loss needs out['rho_A'] ...")` at the first training step (the table and
  `forward_pc`'s `rho_A`/`rho_D` outputs were simply never populated). A register_buffer does not
  draw from the RNG stream, so this does not touch S4.1's "lambda = beta = 0 recovers Paper-1
  exactly" byte-identity claim — re-verified by a dedicated unit test. R0ⁿ-style configs
  (`pc_loss.enabled=true`, λ₁=λ₂=0, used purely for the noise-aug loader plumbing) are
  unaffected: still no table loads for them. R4 (λ₂=0.1): 3 seeds. R3 (λ₁=0.01): 1 seed.
- **R1 (D4).** Auxiliary type-head loss weight fixed at **1.0**, not tuned. The cross-entropy
  term on `type_logits` itself is not written yet (`fault_localization/models/common.py::pc_loss`
  has no term for it) — this freezes the weight for when it is. 1 seed.
- **R2 (D5).** The default decoder (`decoder_hidden=64`), not a backbone shrunk to match R5's
  +1-parameter delta. Measured parameter counts: R0ⁿ 95,258; R5 95,259 (+1, `beta`); R0ⁿ +
  default decoder 99,613 (**+4,355** over R0ⁿ, not parameter-matched to R5). The reconstruction
  loss term itself is not written yet, same as R1. 3 seeds.

  **R2 reconstruction-loss weight, decided `[2026-10-24 16:48 IST / 2026-10-24T11:18:09Z]`:
  `pc_loss.recon_loss_weight = 1.0`.** The loss code (`models/common.py::pc_loss`,
  `recon_loss_weight`/`recon_target`/`recon_mask`) was written on 2026-10-24 (no training run).
  Balance check before freezing the weight: on one batch from the seed-42 training stream (the
  first batch in that stream with a synthetically dropped channel — the seed-42 stream's own
  first batch draws k=0, which would give a vacuous 0/0 reconstruction loss), at model
  initialization (no training steps), main loss (CE) = 3.458318, reconstruction loss
  (unweighted MSE on the 192 dropped channels of that batch) = 0.726126, ratio
  recon/main = 0.210 (main/recon = 4.76) — inside the pre-registered [0.1, 10] acceptance
  band (`scripts/check_r2_recon_weight.py`), so weight 1.0 needs no rescaling. `config/s4_4/r2.yaml`
  updated accordingly; R2 is not part of the approved training queue yet.
- **R7: dropped, not run (D6).** R5 has exactly one parameter more than R0ⁿ (`beta`). No width
  change can match a +1-parameter target, so a wider backbone does not test what A3 ("capacity
  does not reproduce it", `14_PREREGISTRATION.md` §2) asks. Removed from the ablation queue; the
  table row in `S4.4_remaining_arms_plan.md` § 3 is kept struck through, not deleted.
- **B2ᵃ: deferred to Stage G, not run in this round (D7).** Its implementation plan
  (`S4.4_remaining_arms_plan.md` § 6.2) stands unwritten and unexecuted; tracked under Stage G's
  own budget, not this ablation's near-term total.
- **Seed counts: not promoted (D9).** R1 and R3 stay at 1 seed each (R7 is dropped, so moot for it).
- **Recomputed near-term total: 318.81 GPU-h ≈ 13.28 days** for the 7 decided arms (R6 ×3, R8 ×1,
  R5-γ0 ×1, R4 ×3, R3 ×1, R2 ×3, R1 ×1 = 13 run-seeds), down from the pre-decision 364.41 GPU-h
  (9 arms, 15 run-seeds: −12.01 GPU-h from dropping R7, −33.60 GPU-h from deferring B2ᵃ).
  R8's cost (33.60 GPU-h) is unaffected and is the only arm in this set approved to launch
  immediately after S5.1 completes and H1 is checked (D8) — see GATE 4 below.
- **Code this round, no training run:** the runner fix above; the scorer generalized to accept
  any arm and to report `f1_by_resistance` per cell (needed for the H1 check below);
  `scripts/make_r6_shuffle_permutation.py` + its output file; `config/s4_4/r8.yaml`; 18 new unit
  tests in `tests/test_s4_4_remaining_arms.py` (passing), plus the existing 16
  `tests/test_pc_stgnn.py` tests re-run clean as a regression check. R1/R2/B2ᵃ loss/head code was
  explicitly not written this round.

## GATE 4

- [x] S4.1 null test exact — PASSED 2026-09-30, byte-identical over 30 epochs (see S4.1)
- [x] S4.2 gradients non-zero on both λ terms; table receives none — PASSED 2026-09-30, 12/12
      applicable checks (see S4.2)
> **[Replaced 2026-10-22 23:11 IST / 2026-10-22T17:41:10Z — the item below is superseded by
> "GATE 4 — corrected record" after it. Kept unchanged, not deleted. Three errors: its Holm values
> omit the step-down maximum; it says "every noise operating point (k ∈ {1,2,5,10} × σ ∈ {…})"
> (16 cells) but its table has 7; and its 7-cell Holm family was not listed before the runs.]**

- [x] **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
      across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ) —
      **PASSED 2026-10-22**, all 6 Gate 4 runs trained and scored (seeds 42/43/44, 30 epochs
      each, `results/s4_4/{r0n,r5}/pcstgnn/seed_{00042,00043,00044}/`). R5 > R0ⁿ at **every**
      noise operating point (k ∈ {1,2,5,10} × σ ∈ {0,5e-4,2e-3,1e-2}), **all 3 seeds**, sign
      never flips:

      | Cell | Δ (R5 − R0ⁿ), seed42/43/44 | mean Δ | Holm-adj. p |
      |---|---|---|---|
      | k=1, σ=0 | +0.0575 / +0.0796 / +0.0669 | +0.0680 | 0.0175 |
      | k=2, σ=0 | +0.0667 / +0.0886 / +0.0779 | +0.0777 | 0.0196 |
      | k=5, σ=0 | +0.0993 / +0.1200 / +0.1170 | +0.1121 | 0.0166 |
      | k=10, σ=0 | +0.1874 / +0.2059 / +0.2109 | +0.2014 | 0.0088 |
      | k=10, σ=5e-4 | +0.1845 / +0.2043 / +0.2074 | +0.1987 | 0.0078 |
      | k=10, σ=2e-3 | +0.1440 / +0.1726 / +0.1755 | +0.1641 | 0.0149 |
      | k=10, σ=1e-2 | +0.0781 / +0.1175 / +0.1063 | +0.1006 | 0.0133 |

      Paired t-test over the 3 seeds, Holm-corrected within this 7-cell family (per the n=3
      discipline in `CLAUDE.md`) — every cell survives at α=0.05. At clean k=0 (not a gate
      condition, reported for context only): R5 also leads, +0.049/+0.071/+0.057. No run
      collapsed (`collapse_abort.triggered` false in all 6 `metrics.json`). **Gate 4 passes.**

### GATE 4 — corrected record `[Added 2026-10-22 23:11 IST / 2026-10-22T17:41:10Z]`

**The rule as written before the runs.** The first Gate 4 run started 2026-10-12T09:14:32Z
(`results/s4_4/gate4_progress.json`, R0ⁿ seed 42). The last commit of this file before that is
`9d18184` (2026-10-12 14:44:20 IST = 09:14:20Z). In `9d18184` the rule is at lines 657–658 (quoted
in the frozen settings) and lines 758–759 (this checklist); in the current file at lines 660–661
and the checklist item above. Verbatim:

> **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
> across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ)

The rule asks for sign consistency over the 3 seed pairs. It does not ask for a significance test.

**"Pre-registered conditions", as written before the runs** (`git show 9d18184:<file>`):
- `14_PREREGISTRATION.md` lines 51–54: H1 10 Ω, k = 0 · H2 k = 10 · H3 R ∈ {3,5,20,30} Ω, k = 0
  (sealed OOD) · H4 σ₀ ∈ {5e-4, 2e-3, 1e-2}, k = 0.
- `13_BUDGET_TIMELINE.md` line 64: "10 Ω, k = 10, OOD R, injected noise, silent failures".

**What the Gate 4 data can evaluate.** `results/s4_4/*/scoring_full_grid.json` holds validation F1
pooled over all resistances on the grid k ∈ {0,1,2,5,10} × σ ∈ {0, 5e-4, 2e-3, 1e-2}.
- H2 (k = 10) and H4 (σ₀ > 0, k = 0): yes.
- H1 (10 Ω slice): **no** — the scored files have no per-resistance breakdown. It needs a scoring
  run, which was not done.
- H3 (sealed OOD R) and silent failures: Stage G and Stage F conditions, outside Gate 4.

**Rule applied** (Δ = R5 − R0ⁿ, validation macro-F1, pairs by training seed):

| Condition | Cell | Δ seed 42 / 43 / 44 | mean Δ | Sign-consistent |
|---|---|---|---|---|
| H2 | k=10, σ=0 | +0.1874 / +0.2059 / +0.2109 | +0.2014 | yes |
| H2 | k=10, σ=5e-4 | +0.1845 / +0.2043 / +0.2074 | +0.1987 | yes |
| H2 | k=10, σ=2e-3 | +0.1440 / +0.1726 / +0.1755 | +0.1641 | yes |
| H2 | k=10, σ=1e-2 | +0.0781 / +0.1175 / +0.1063 | +0.1006 | yes |
| H4 | k=0, σ=5e-4 | +0.0509 / +0.0687 / +0.0576 | +0.0590 | yes |
| H4 | k=0, σ=2e-3 | +0.0384 / +0.0642 / +0.0528 | +0.0518 | yes |
| H4 | k=0, σ=1e-2 | +0.0083 / +0.0437 / +0.0218 | +0.0246 | yes |

21 of 21 seed pairs are positive. If "read at the noise operating points" excludes σ = 0, the H2
row at σ = 0 drops out; the result does not change.

**Result: passed** on every pre-registered condition that the Gate 4 data can evaluate (H2, H4).
**H1 (10 Ω) is not evaluated** — open item.

**Holm.** No Holm family for the Gate 4 rule was written before the runs. The only family written
before the runs is `14_PREREGISTRATION.md` §3 (line 78 in `9d18184`): "families = {H1–H4}, {H5},
{A1–A3}", for the hypotheses (test split, with an event bootstrap). It cannot be formed from the
Gate 4 data, because H1 and H3 are not available. So no Holm result is part of the Gate 4 verdict.
Raw paired-t p (n = 3, df = 2) for the H4 cells, descriptive only: 0.0077, 0.0200, 0.1398.

**Correction of the old table — Holm values.** The old table's 7-cell family (k ∈ {1,2,5} at σ = 0;
k = 10 at all four σ) was chosen on 2026-10-22, after the results existed. It was **not listed
before the runs**; treat it as post hoc. Its old values are the Holm step products without the
step-down maximum. Corrected (step-down maximum applied):

| Cell | raw p | Old value | Corrected Holm p |
|---|---|---|---|
| k=1, σ=0 | 0.0087 | 0.0175 | **0.0196** |
| k=2, σ=0 | 0.0065 | 0.0196 | 0.0196 |
| k=5, σ=0 | 0.0033 | 0.0166 | 0.0166 |
| k=10, σ=0 | 0.0013 | 0.0088 | 0.0088 |
| k=10, σ=5e-4 | 0.0013 | 0.0078 | **0.0088** |
| k=10, σ=2e-3 | 0.0037 | 0.0149 | **0.0166** |
| k=10, σ=1e-2 | 0.0133 | 0.0133 | **0.0196** |

All seven stay below 0.05.

**Correction of the old table — cell count.** The old text says "every noise operating point
(k ∈ {1,2,5,10} × σ ∈ {0,5e-4,2e-3,1e-2})", which is **16** cells, but its table has **7**.
Correct statements: Δ > 0 in all 16 cells for all 3 seeds (48 of 48 pairs); the table shows the
post-hoc 7-cell family. In a 16-cell Holm family (also post hoc), 5 of 16 cells are below 0.05:
k=5 σ=0 (0.0431), k=5 σ=5e-4 (0.0411), k=10 σ=0 (0.0201), k=10 σ=5e-4 (0.0201), k=10 σ=2e-3
(0.0447).

**If R5 ≈ R0 on validation.** Do not proceed to Stage F expecting a rescue. Go straight to
Stage G, run the OOD and noise evaluations, and write the "physics buys generalisation, not
in-distribution accuracy" paper. *(Not applicable — gate passed, see above.)*
