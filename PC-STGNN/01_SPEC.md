# Frozen Specification

If any line here changes, restart from Gate 1.

`[Corrected 2026-09-22]` §2–§4 were rewritten after checking the simulator: the corpus load is a
**single global scalar** per event (`simulation/runner.py:116`, `monitors.py:196`), not a
per-bus vector. The load-ensemble operator of the first draft is kept at the end of §2 as the
rejected alternative.

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
| L — load multiplier | 17, linear 0.5 → 1.3 (step 0.05) — **an explicit axis, because the corpus load is a scalar** |
| i, p — sensor, phase | 25 × 3 = 75 channels, of which only the instrumented ones are ever used |

For each (f, τ, R, L) run one snapshot solve at load L × 0.99953 (the frozen pre-fault
modulation) and store

```
μ[f, τ, R, L, i, p]   post-fault magnitude, p.u., per µPMU phase
μ₀[L, i, p]           no-fault magnitude at load L (the f = 0 hypothesis)
```

Cost: 25 × 11 × 15 × 17 = 70,125 solves + 17 for μ₀ ≈ 25–30 min at ~45 solves/s (measure in
S0.5), ≈ 21 MB in float32.

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

```
r_f(A)  = min over (τ, R, L) of  Σ_{(i,p)∈A}  (s_ip − μ[f,τ,R,L,i,p])² / (σ_ε² + σ₀²)      f = 1…25
r₀(A)   = min over L         of  Σ_{(i,p)∈A}  (s_ip − μ₀[L,i,p])²        / (σ_ε² + σ₀²)      f = 0
r̃_f(A) = r_f(A) / |A|
```

- σ_ε — a **fixed** per-channel floor, set a priori from the snapshot-vs-daily equivalence error
  measured in S0.4 (order 1e-3 p.u.). It is not fitted on data. It exists so the residual is
  finite on channels the fault barely moves.
- σ₀ — the injected measurement-noise level under the noise experiments (0 on the clean corpus).
- r̃_f(D) reuses the (τ*, R*, L*) profiled at f from A alone, then scores at the sensors in D.

Properties this buys:
- (τ, R, L) profiled, not regressed → defined at resistances never seen in training.
- Sum over A only → dropout handled by construction.
- Depends on inputs alone → no differentiable table interpolation, no solver in the training
  loop, and r̃ can be precomputed from the raw `.npz` (in p.u., no normalisation issue) and cached
  for the whole corpus.

## 5. Architectural rule (hard)

The physics operator is a function of the inputs only. No network output is ever an argument to
μ or μ₀.

- The network predicts location only, z ∈ ℝ²⁶.
- A type head, if kept, is auxiliary supervision (ablation R1) and is never read by the physics.
- A learned R̂ head is not part of the model. It is ablation B2ᵃ, run once to show what a
  regressed resistance costs on OOD R.

## 6. PC-STGNN

```
z'_f = z_f − β · r̃_f(A)                                   f = 0 … 25

L = CE(z', y) + λ₁ · Σ_f P(f|X)·r̃_f(A) + λ₂ · Σ_f P(f|X)·r̃_f(D)

λ₁ = λ₂ = β = 0   ⟹   exactly dropout-aware RGSAGE-Max
```

λ₁ = expected residual on the available sensors. λ₂ = the same operator on the held-out sensors —
a leave-k-out physics-consistency term. Both are differentiable through P(f|X).

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
- For each available sensor i: refit (τ, R, L) at f̂ on A \ {i}; read the normalised error at
  i → r̃ᵢ^LOO. Leaving the sensor out stops a corrupted sensor pulling the fit toward itself and
  hiding its own error.
- Gate: gᵢ = mᵢ · σ(−α (r̃ᵢ^LOO − c)), with α, c learned.
- Pass 2: scale sensor i's input embedding and soft-vote weight by gᵢ, then re-run the 4-layer
  RGSAGE-Max and head. **Not** hop-level silencing — Paper 1 showed that cuts the multi-hop paths
  that route around a lost node (SDR-STGNN AUDC 0.892 vs 0.913).

Total new learnable parameters: three scalars (β, α, c).

## 8. Gradient audit table

| Quantity | Depends on network? | Gradient flows to |
|---|---|---|
| r̃_f(A), r̃_f(D), r̃ᵢ^LOO | No — inputs only | nothing |
| z'_f = z_f − β·r̃_f | Yes | network via z, and β |
| Σ_f P(f\|X)·r̃_f | Yes, through P | network via P |
| gᵢ | only through α, c | α, c |
| Pass-2 output | Yes, gᵢ·hᵢ | network via hᵢ |
| μ, μ₀, σ_ε | — | never |
