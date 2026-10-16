# Frozen Specification

If any line here changes, restart from Gate 1.

`[Corrected 2026-09-22]` §2–§4 were rewritten after checking the simulator: the corpus load is a
**single global scalar** per event (`simulation/runner.py:116`, `monitors.py:196`), not a
per-bus vector. The load-ensemble operator of the first draft is kept at the end of §2 as the
rejected alternative.

`[Corrected 2026-09-23]` §2, §4, §6 and §7 amended per `../PC-STGNN-PLAN.md` §2: the L-grid step
is decided by S1.2 rather than defaulted (§2); the residual is the **shifted, halved** quantity ρ
in Gaussian log-likelihood units, replacing the per-channel average r̃ (§4); fusion is stated as
the **per-node** broadcast (a product of experts) it actually is (§6); and the LOO score is
defined as **subtract-then-re-minimise** (§7).

`[Corrected 2026-09-25]` §2's R axis amended: the original 15-point log-spaced grid
(0.05–30 Ω) kept the corpus's own training resistances {0.1, 1, 10} Ω off-grid, on the theory
that this is what keeps S1.7's argmin_R interpolation check non-vacuous. S1.5 (`03_STAGE_B_
operator.md`) found this corrupts the load-axis check instead: a nearby-but-wrong off-grid R
combines with a compensating L shift to fit better than the true (off-grid) R + true L, so
profiled L* recovery failed (38.4 % within one grid step, against the >90 % criterion) even
though R* itself always landed sensibly. Replaced with an 18-point **hybrid** grid —
`sorted(unique(15pt log grid, {0.1, 1, 10}))` — which puts the corpus's own R values exactly
on-grid; L* recovery jumped to 97.3 %. Densifying to 25 log-spaced points (S1.7's own
branch-on-failure remedy) does **not** fix it — only 58.6 % — because it shrinks ordinary
discretisation error, which was never the diagnosed mechanism. Side effect on σ_ε (§4): the
corrected table's true-hypothesis error dropped along with the R-L compensation, so `sigma_eps`
was rescaled from 5.145e-3 to **7.2555e-4** (S1.6's own branch instruction) to keep median
r̄_{f_true} at 1.0000. Unexpected side effect beyond S1.5: PSM top-1 accuracy at each L step's
own discretisation σ moved from **~0.555–0.562** (the same R-L compensation degrading location
attribution, not just load recovery) to **0.865–0.942** on the corrected table. Full derivation,
the three-way grid comparison, and the re-checked S1.2/S1.6/S1.7 numbers:
`../results/pcstgnn/operator/S1.5_RGRID_CORRECTION.md`. The operator description in §2 below is
otherwise unchanged; only the R axis's grid points and `NUM_R` (15 → 18) moved.

## 1. Inherited constants (Paper 1)

| Item | Value |
|---|---|
| Feeder | IEEE 123-bus, 4.16 kV, 60 Hz, OpenDSS Daily mode, 1 ms step |
| µPMUs | 25, three-phase RMS voltage magnitudes; only 14 of 25 nodes are three-phase, absent phases carry V = 0 and m = 0 |
| Graph | measured-only: 25 nodes, 24 topology edges (tree), self-loops added |
| Candidate fault locations | 25 (only 6 coincide with instrumented buses) |
| Fault types | 11 (3 LG, 3 LL, 3 LLG, 2 three-phase) |
| Fault resistances (training) | 0.1, 1, 10 Ω (`config/constants.py:59`) |
| Load | **one scalar** multiplier L ~ U(0.5, 1.3) per event, applied to every load; a fixed 0.3 % sinusoid over the 40 pre-fault steps; load frozen at the step-39 value (≈ 0.99953 L) during the fault |
| Scenarios | 25 × 11 × 3 = 825, × 50 runs = 41,250 events |
| Record / window | 60 ms record, inception at step 40, 40 windows of 20 steps, stride 1 ms; window w covers steps [w, w+20) |
| Window labels | windows 0–19 no-fault, 20–39 fault (`processing/labels.py:79`); fault window w has w−20 post-inception samples |
| Node features | F = 6: [V₁ V₂ V₃ \| m₁ m₂ m₃]; voltage channels are z-scored with training-split stats in the loader (`data/dataset.py:61`) |
| Classes | 26 = 25 locations + no-fault; exactly 50 % of windows are class 0 |
| Split | 70/15/15 at the event level; 1.65 M windows; 6,175 test events. Every one of the 825 scenarios appears in all three splits (the split partitions runs, not scenarios) |
| Backbone | GRU(128) → 4 × RGSAGE-Max(64), ReLU → BN/ReLU/Dropout 0.35/Linear(26) → soft vote |
| Training | AdamW, batch 32, 30 epochs, grad-clip 1.0, warm-up 3, seeds 42/43/44, checkpoint by validation graph-F1 at k = 0 |
| Dropout sweep | k ∈ {0, 1, 2, 5, 10}, 3 mask seeds, per-batch seed `mask_seed·10000 + batch_idx` |
| Runtime | one RGSAGE-Max run = 6.7 h (standard) / 6.9 h (dropout-aware) on the T1000 (`results/runtime/runtime_training.csv`) |
| Known weak points | 10 Ω: F1 0.676 · dropout-aware at k = 10: F1 0.674 · SDR-STGNN AUDC 0.892 < dropout-aware 0.913 · noise: standard 0.846 → 0.345 at σ = 0.01 p.u. and dropout-aware does not help |

## 2. Physics operator

Fixed tensor. Zero parameters. Never receives a gradient. Built with the existing fault-injection
code in OpenDSS snapshot mode.

| Axis | Values |
|---|---|
| f — candidate location | 25, plus the no-fault hypothesis f = 0 |
| τ — fault type | 11 |
| R — fault resistance | 15, log-spaced 0.05 → 30 Ω |
| L — load multiplier | linear 0.5 → 1.3 — **an explicit axis, because the corpus load is a scalar**. `[Corrected 2026-09-23]` The step is **decided by S1.2**, not defaulted: 0.05 (17 points) and 0.025 (33 points, ≈ 140 k solves) are both affordable on 24 cores, and with σ_ε ≈ 1e-3 p.u. against ≈ 2–5e-3 p.u. of voltage change per 0.05 of load, discretisation — not fidelity — is the likely dominant error. The chosen step is recorded in `operator/grid.json` |
| i, p — sensor, phase | 25 × 3 = 75 channels, of which only the instrumented ones are ever used |

For each (f, τ, R, L) run one snapshot solve at load L × 0.99953 (the frozen pre-fault
modulation) and store

```
μ[f, τ, R, L, i, p]   post-fault magnitude, p.u., per µPMU phase
μ₀[L, i, p]           no-fault magnitude at load L (the f = 0 hypothesis)
```

Cost at step 0.05: 25 × 11 × 15 × 17 = 70,125 solves + 17 for μ₀, ≈ 21 MB in float32; at step
0.025, 136,125 + 33 solves and ≈ 41 MB. Wall-clock from the S0.5 solve rate.

**Why L is profiled, not marginalised.** The only within-scenario randomness in the corpus is
one scalar L; the 50 runs of a scenario lie on a 1-D curve μ(L). A diagonal "load std" is the
wrong model of that (channel deviations are perfectly correlated along the curve) and it would
have made the residual over-confident on the very axis it was meant to absorb. Profiling L
matches the generative process exactly, and because `meta["load_multiplier"]` is stored per
event, argmin_L is directly testable (S1.5).

**Rejected alternative (first draft):** N_L = 12 independent per-bus draws from U(0.5, 1.3)
per (f, τ, R), storing μ and σ_load. Valid only for a corpus with per-bus load variation. If the
corpus is ever regenerated with per-bus loads, this becomes the right operator again and the
first-draft S1.5/S1.6 tests apply.

## 3. Observation vector

```
s_ip = x[i, S−1, p]          the LAST timestep of the window, de-normalised to p.u.
A    = { (i,p) : m_ip = 1 }  available sensor-phases (excludes uninstrumented phases and dropped nodes)
D    = synthetically dropped set (training only; true values known)
```

`[Corrected 2026-09-22]` The first draft used "mean of the post-inception timesteps", which
needs the window's offset inside the event — information that is not in the `[25, 20, 6]`
tensor and would violate §5. The fault period is quasi-static with loads frozen, so every
post-inception sample is the same level to numerical precision (verify: S0.3) and the last sample
is that level whenever the window has ≥ 1 post-inception sample. For window 20 (0 samples) the
last sample is the pre-fault level and the residual naturally prefers f = 0; that is the
"physics disabled" policy, obtained without any metadata. Windows with 0 post-inception samples
are still excluded from L₁ and L₂ (they are identifiable from the window id at training time).

## 4. Residual — inputs only

`[Corrected 2026-09-23]` The per-channel average r̃ = r/|A| of the first draft is replaced by the
**shifted, halved** residual ρ. Reason: with L profiled and σ_ε ≈ 1e-3 p.u., a wrong hypothesis
contributes per-channel terms of order (0.1/1e-3)² = 1e4, so summed over ≤ 75 channels r_f reaches
1e5–1e6 for wrong locations while the true one is ≈ |A|. Dividing by |A| does not remove that
scale — it only rescales it — so λ₁ would have needed to be ≈ 1e-5 and any usable β would have
saturated the softmax. Subtracting the minimum and halving puts the quantity in Gaussian
log-likelihood units, where a difference of 1 is one nat and λ, β of order 1 are meaningful.

```
r_f(A)  = min over (τ, R, L) of  Σ_{(i,p)∈A}  (s_ip − μ[f,τ,R,L,i,p])² / (σ_ε² + σ₀²)      f = 1…25
r₀(A)   = min over L         of  Σ_{(i,p)∈A}  (s_ip − μ₀[L,i,p])²        / (σ_ε² + σ₀²)      f = 0

ρ_f     = ( r_f − min_g r_g ) / 2            ≥ 0, one exact zero, scale-free
P_phys  = softmax(−ρ)
```

`residual.py` returns ρ together with the argmin hypothesis (τ*, R*, L*) per f. PSM = argmin_f ρ_f
= argmax_f P_phys. Fusion, L₁, L₂, the LOO score, the identifiability score of §5 in
`11_METRICS.md` and every reported diagnostic (ρ_true, margin ρ_(2) − ρ_(1)) use ρ. β = 0 still
recovers the baseline exactly, so the S4.1 null test is unaffected.

- σ_ε — a **fixed** per-channel floor. It is not fitted on data. It exists so the residual is
  finite on channels the fault barely moves.
  `[Corrected 2026-09-23]` It can **not** be set from the snapshot-vs-Daily equivalence error as
  this section originally said: S0.4 measured that error at **1.02e-12 p.u.**, because the
  operator is not an approximation of the corpus — it is the corpus's own generator re-run. A
  floor of that size would make the residual absurdly over-confident. Two consequences, both
  load-bearing:
  - **On the nominal table σ_ε is a numerical quantity, not a physical one.** Its true source is
    **grid discretisation** — the corpus's L is continuous while the table's L is on a grid — so
    it is measured in S1.2 and calibrated in S1.6, and it changes if the grid changes. It does
    not describe any physical measurement error, and the paper must not imply that it does.
  - **It carries a stated physical floor.** σ_ε is never set below the µPMU accuracy class
    (IEEE C37.118 1 % TVE; the noise grid is `05_STAGE_D_infra.md` §"injectors"), so the residual
    scale still means something in the perturbed-table and noisy settings — which, per the
    degeneracy ruling in `04_STAGE_C_gonogo.md` S2.0ᵃ, is where the evidence now lives.
    Outside the oracle setting the effective denominator is σ_ε² + σ₀² and is dominated by σ₀.
- σ₀ — the injected measurement-noise level under the noise experiments (0 on the clean corpus).
- ρ_f(D) reuses the (τ*, R*, L*) profiled at f from A alone, then scores at the sensors in D, and
  is shifted by its own minimum over f in the same way.

Properties this buys:
- (τ, R, L) profiled, not regressed → defined at resistances never seen in training.
- Sum over A only → dropout handled by construction.
- Depends on inputs alone → no differentiable table interpolation, no solver in the training
  loop, and ρ can be precomputed from the raw `.npz` (in p.u., no normalisation issue) and cached
  for the whole corpus.
- Shift-invariant to any per-window constant → the scale of ρ is set by hypothesis *contrast*,
  not by |A|, so it stays comparable across dropout levels.

## 5. Architectural rule (hard)

The physics operator is a function of the inputs only. No network output is ever an argument to
μ or μ₀.

- The network predicts location only, z ∈ ℝ²⁶.
- A type head, if kept, is auxiliary supervision (ablation R1) and is never read by the physics.
- A learned R̂ head is not part of the model. It is ablation B2ᵃ, run once to show what a
  regressed resistance costs on OOD R.

## 6. PC-STGNN

`[Corrected 2026-09-23]` Fusion is applied **per node**, before the soft vote — not to a
graph-level logit vector. The per-node cross-entropy of `models/common.py:52`, the soft vote of
`models/voting.py`, `dropout/evaluator.py` and every loader therefore stay untouched.

```
z'_{n,f} = z_{n,f} − β · ρ_f(A)        same ρ broadcast to every node n     (β learned, init 0)

Σ_n softmax(z'_n)_f  ∝  Σ_n exp(z_{n,f}) · exp(−β ρ_f) / Z_n
```

Each node's distribution is reweighted multiplicatively by the same physics prior before the
votes are summed: a **product of experts** per node, not an additive graph-level bias. The paper
must describe it that way.

```
L = CE(z', y) + λ₁ · Σ_f P(f|X)·ρ_f(A) + λ₂ · Σ_f P(f|X)·ρ_f(D)

P(f|X) = normalised soft-vote scores        λ₁ = λ₂ = β = 0 ⟹ exactly dropout-aware RGSAGE-Max
```

λ₁ = expected residual on the available sensors. λ₂ = the same operator on the held-out sensors —
a leave-k-out physics-consistency term. Both are differentiable through P(f|X).

w = 20 windows (0 post-inception samples, §3) are excluded from the λ terms by a loss mask
derived from `window_id`. That is the **only** use of the window index anywhere, it applies at
training time only, and it never enters the residual itself.

Do not use a reconstruction loss of the form ‖μ[f̂, …] − s_true‖². f̂ is an arg-min index and
constant in the weights, so that term has zero gradient and R4 would be silently identical to R3.
Verified by S4.2.

**Prior evidence (already on disk):** `results/pignn_p3_full/` — a physics-in-the-loss term
built on the first-order rank-1 sag template of `graph/sensitivity.py` — is a null at 3 seeds
(0.847 vs 0.846 standard; 0.837 vs 0.839 dropout-aware), and the same template as a training-free
matched filter scores 0.162 (`results/physics_baseline/`). The operator above differs in kind
(full (τ, R, L) signatures with the driving-point term, from the solver), and the gap B1 → B2 in
Stage C is the measurement of that difference. If B2 is not far above 0.162, λ₁ has no reason
to do better than that pilot.

## 7. PCRG-STGNN

- Pass 1 gives f̂ = argmax z'.
- For each available sensor i: **subtract, then re-minimise**. Build the per-channel error tensor
  E[hypothesis, channel] at f̂ once; zero sensor i's channels; take the min over the hypothesis
  axis *again* → the refit (τ*, R*, L*) on A \ {i}; read sensor i's normalised error **at that
  refit** → ρᵢ^LOO. `[Corrected 2026-09-23]` Reading sensor i's term at the *global* argmin
  (τ*, R*, L*) fitted on all of A is the in-sample score, not a LOO score: it lets a corrupted
  sensor pull the fit toward itself and hide its own error, and it would make the S5.2 comparison
  a variant against itself. `residual.py` implements both — `rho_loo` and `rho_insample` — and the
  difference between them is the S5.2 result.
- Gate: gᵢ = mᵢ · σ(−α (ρᵢ^LOO − c)), with α, c learned.
- Pass 2: scale sensor i's input embedding and soft-vote weight by gᵢ, then re-run the 4-layer
  RGSAGE-Max and head. **Not** hop-level silencing — Paper 1 showed that cuts the multi-hop paths
  that route around a lost node (SDR-STGNN AUDC 0.892 vs 0.913).

Total new learnable parameters: three scalars (β, α, c).

## 8. Gradient audit table

| Quantity | Depends on network? | Gradient flows to |
|---|---|---|
| ρ_f(A), ρ_f(D), ρᵢ^LOO | No — inputs only | nothing |
| z'_{n,f} = z_{n,f} − β·ρ_f | Yes | network via z, and β |
| Σ_f P(f\|X)·ρ_f | Yes, through P | network via P |
| gᵢ | only through α, c | α, c |
| Pass-2 output | Yes, gᵢ·hᵢ | network via hᵢ |
| μ, μ₀, σ_ε | — | never |
