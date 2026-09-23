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

**Branch on failure.** You have an unintended change in the data path, seeding, or graph
construction. Find it now. Every attribution claim in the paper rests on this single property —
it is what lets you say "λ = 0 recovers the Paper-1 model exactly."

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

## S4.3 — Tuning

**Do.** Single seed, 15-epoch budget (as in the Paper-1 depth study and the PIGNN screen). Sweep
λ₁ ∈ {0.1, 0.3, 1}, then λ₂ ∈ {0.1, 1} at the best λ₁. β is learned. ≈ 5–6 half-cost runs.

**Check.** Select on validation F1 averaged over k ∈ {0,1,2,5,10}. Never test. Never OOD.
(Checkpoint selection *within* a run stays validation F1 at k = 0, as in Paper 1 — two different
rules, both stated in the paper.)

**Branch.** Best λ₁ at a grid boundary → extend one step in that direction.

**Cost.** 6 × 3.5 h ≈ 21 GPU-h.

**Figure produced.** F-E1 — validation F1 vs λ₁ (appendix).

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

- [ ] S4.1 null test exact
- [ ] S4.2 gradients non-zero on both λ terms; table receives none
- [ ] **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
      across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ)

**If R5 ≈ R0 on validation.** Do not proceed to Stage F expecting a rescue. Go straight to
Stage G, run the OOD and noise evaluations, and write the "physics buys generalisation, not
in-distribution accuracy" paper.
