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

**Check.** Loss curve and per-epoch validation F1 against the existing R0 run
(`results/dropout_aware/rgsage_max/seed_00042/metrics_by_epoch.csv`).

**Pass.** Exact reproduction. (`tests/test_pignn.py` already asserts byte-identity of the PIGNN
model with all switches off against RGSAGE-Max — copy that pattern.)

**Branch on failure.** You have an unintended change in the data path, seeding, or graph
construction. Find it now. Every attribution claim in the paper rests on this single property —
it is what lets you say "λ = 0 recovers the Paper-1 model exactly."

## S4.2 — Gradient audit — guards the bug that was in the first draft

**Do.** On a single batch, finite-difference check.

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

## S4.4 — Core ablation — 30 epochs, everything else frozen

| # | Variant | Answers the objection | Seeds |
|---|---|---|---|
| R0 | Dropout-aware RGSAGE-Max | reference (already trained) | 3 (done) |
| R1 | + type head only | "the auxiliary task is doing it" | 1 |
| R2 | + learned MLP decoder on D, parameter-matched | "any decoder / masked autoencoding would do it" | 3 |
| R3 | + λ₁ (residual on A) | physics as a training constraint | 1 |
| R4 | + λ₂ (residual on held-out D) | does held-out consistency add over in-sample? | 1 |
| R5 | + β fusion — full PC-STGNN | physics as evidence at test time | 3 |
| R6 | R5 with a shuffled location→signature map | "it's regularisation, not physics" | 3 |
| R7 | Size-matched wider RGSAGE-Max | "it's just a bigger model" | 1 |
| B2ᵃ | learned R̂ head instead of profiling | shows why profiling beats regression on OOD R | 1 |

`[Corrected 2026-09-22]` Seed counts follow the S0.5 rule (one run ≈ 7 h). Promote R1/R3/R4/R7 to
3 seeds only if Gate 4 passes and budget remains.

R6 and R2 carry the burden of proof. The target pattern is F1_real > F1_none ≈ F1_shuffled.

R7 matters less than it looks — the physics path adds three scalars — but it is a one-line
objection with a one-row answer, so run it.

B2ᵃ is expected to fail on OOD R: a head trained on {0.1, 1, 10} Ω collapses to a three-way
classifier. That failure is a result worth a table row.

**Cost.** 3+1+3+1+1+3+3+1+1 = 17 runs × 7 h ≈ 120 GPU-h (R7 wider: somewhat more).

**Table produced.** T7 — core ablation.
**Figure produced.** F-E2 — ablation bar chart with seed spread (optional if T7 is clear).

## GATE 4

- [ ] S4.1 null test exact
- [ ] S4.2 gradients non-zero on both λ terms; table receives none
- [ ] R5 > R0 on validation, in the pre-registered conditions, sign-consistent across 3 seeds

**If R5 ≈ R0 on validation.** Do not proceed to Stage F expecting a rescue. Go straight to
Stage G, run the OOD and noise evaluations, and write the "physics buys generalisation, not
in-distribution accuracy" paper.
