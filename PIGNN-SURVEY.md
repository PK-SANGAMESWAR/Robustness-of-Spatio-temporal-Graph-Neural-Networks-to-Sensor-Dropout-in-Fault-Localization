# Architectures & Physics-Informed GNNs — literature survey and design decision

**Purpose:** decide, on evidence, whether the next paper should (a) design a new architecture
from scratch, (b) graft a PIGNN mechanism onto the existing RGSAGE backbone, or (c) do both as
a controlled comparison. Also: which *kind* of PIGNN, given that "PIGNN" is not one thing.

**Status:** survey complete. Local audit complete. Recommendation in §7. Experiment plan in §8.
No new experiments run. Every number quoted is either from `results/` on disk or from a cited
paper.

---

## 1. The single most important local finding

`fault_localization/simulation/monitors.py:78-80`:

```python
re = pu_voltages[2 * idx]
im = pu_voltages[2 * idx + 1]
voltages[node - 1] = (re**2 + im**2) ** 0.5   # <-- angle computed, then discarded
```

**You already have voltage phase angles. You are throwing them away at extraction time.**

This matters more than any architectural choice below, because the entire "hard physics" branch
of the PIGNN literature — Kirchhoff residuals, `Y·ΔU = ΔC + Δ_ij`, power-flow-constrained
losses — requires *complex* voltages. With magnitudes only, those losses are not
approximable; they are **undefined**. With angles, they become directly computable.

Concretely: PPGN (§4.1), the closest published competitor to what you are proposing, takes
`X ∈ R^{n×6} = [V_a, θ_a, V_b, θ_b, V_c, θ_c]`. Your feature vector is
`[V_a, V_b, V_c, m_a, m_b, m_c]`. You have the same width and half the physics.

**Cost to fix:** a two-line change in `extract_bus_voltages_with_mask` plus a corpus
regeneration (825 scenarios x 50 runs x 60 ms). The *methodology* does not change; the frozen
corpus does. Needs a timing estimate before committing — this is the one decision that gates
everything else, so make it first.

**Caveat to state in the paper either way:** real µPMUs report synchrophasors, so angles are a
*realistic* measurement, not a cheat. But angle measurement is also where TVE error and clock
drift live, so an angle-using model must be noise-tested at least as hard as the current one.

---

## 2. Where your work currently sits

The base paper is Karabulut, Manna & Develder, *Robustness of Spatio-temporal Graph Neural
Networks for Fault Location in Partially Observable Distribution Grids* (arXiv 2604.20403) —
RGSAGE/RGATv2 + the measured-only graph on IEEE-123.

**That group has already published a follow-up you should know about:** *Generalization of
Graph Neural Network Models for Distribution Grid Fault Detection* (arXiv 2510.03571). It tests
exactly the topology-shift axis you cut as the "green feeder" scenario, and finds RGATv2
degrades ~12% F1 on unseen topologies vs ~25% for other RGNNs and ~60% for pure RNNs.

Two consequences:

1. The green-feeder idea is **taken**. Do not resurrect it as a contribution.
2. Their result — attention generalizes best under structural shift — is *evidence for* an
   attention-based decoder, which is what `FaultQuery-STGNN` already is.

Your unique territory, still unclaimed: **sensor dropout as a first-class stressor**, with
degradation curves, dropout-aware training, targeted/persistent/correlated policies, and a
measured noise axis.

---

## 3. Architecture landscape — what has been tried for this task

| Family | Representative | Used for grid fault loc.? | Verdict for you |
|---|---|---|---|
| RNN only | GRU, LSTM | yes (base paper baseline) | dominated; you have it at 3 seeds |
| Spectral GCN | RGCN | yes, the prior SOTA | you have it; most dropout-robust of your five |
| Sampling/aggregation | GraphSAGE (RGSAGE) | yes (base paper, first) | your strongest at k=0 (0.846) |
| Attention | GAT / GATv2 (RGATv2) | yes (base paper, first) | best topology generalization per 2510.03571 |
| Graph transformer | Graphormer, GraphGPS | **not** in this task | global receptive field — plausible, untested |
| Object-query decoder | DETR-style | **not** in this task | your `FaultQuery-STGNN` |
| SSM / Mamba | Graph Mamba, STG-Mamba | **not** in this task | fashionable; no reason to expect a win at S=20 |
| Neural ODE / jump ODE | PI Graph Neural Jump ODE | cascading failure, not fault loc. | wrong time constant for a 20 ms window |
| NAS | Evolutionary GNAS + mask predictor | yes, distribution fault detection | exists; a search is not a contribution here |
| Classical ML | logreg, RF | yes (your own baseline) | **0.713 / 0.633** — see §3.1 |

### 3.1 The headroom argument — read this before designing anything

From `results/classical_baseline/`: plain **logistic regression on 375 summary features
reaches 0.713** at k=0, against RGSAGE-Max's 0.846, **on 8% of the training data**.

The entire deep-graph margin over a linear model is **0.133**. Whatever a new architecture
wins at k=0, it wins inside that 0.133 band, and your seed-to-seed spread is already ~0.007.

**Therefore: do not target clean accuracy.** It is nearly saturated relative to what the data
supports, and any gain will be statistically indefensible at n=3. This is the strongest single
argument against building an architecture from scratch to "beat 0.846".

---

## 4. Physics-Informed GNNs — the taxonomy, and which types you can actually use

"PIGNN" spans at least six mechanisms with completely different requirements. This is the
section that answers "they are of different types".

| Type | Mechanism | Needs | Feasible for you *now* (V-magnitude only) | Feasible *with angles* |
|---|---|---|---|---|
| **A. Physics-guided loss** (soft) | add residual penalty `L = L_CE + λ·L_phys` | differentiable residual from model outputs | ✗ Kirchhoff undefined; ✓ only via linearized surrogate | ✓ full `Y·ΔU` residual |
| **B. Physics-encoded architecture** (hard) | constraint holds by construction — projection layer, equivariance, conservation | closed-form structure | ✓ **`zone_refine.py` already does this** | ✓ stronger |
| **C. Physics-informed graph construction** | physics sets the adjacency / edge weights | topology + impedances | ✓ you have `get_line_impedances()`, `edge_attributes.py` | ✓ |
| **D. Physics priors / features / init** | hop distance, sensitivity templates as bias or initialization | topology | ✓ `fault_hop_distance.npy`, `fault_sensitivity.npz` | ✓ |
| **E. Physics-based augmentation / differentiable simulator** | simulate in the loop | OpenDSS in training loop | ✓ but slow | ✓ |
| **F. Physics-based similarity / label propagation** | physically-similar samples share labels | embeddings | ✓ (PPGN Stage II) | ✓ |

**Read the table this way:** Type A — the thing most people *mean* by "physics-informed" — is
the one type your current data forbids. Types B/C/D are open to you today, and you have already
built parts of all three without labelling them PIGNN work.

### 4.1 PPGN — the paper you must position against

Li, Deka et al., *PPGN: Physics-Preserved Graph Networks for Real-Time Fault Location in
Distribution Systems with Limited Observation and Labels* (arXiv 2107.02275, HICSS). Same
feeder (IEEE 123 + 37), same observability framing, OpenDSS.

What "physics-preserved" actually means there — worth being precise, because it is **not** a
loss term:

- **Physics analysis (§2 of their paper):** from Kirchhoff, `Y·ΔU = ΔC + Δ_ij` where `Δ_ij` is
  sparse with nonzeros only at the two terminals of the faulted line, and `ΔC ≈ 0` because
  loads rarely change during a fault. So *admittance-weighted voltage deviation is large near
  the fault*. That is the physical claim.
- **Type C — Stage I:** adjacency `a_ij = exp(-d_ij²/δ_i²)` for `j` in the `k_I` nearest
  neighbours by **Dijkstra shortest path**, with `k_I` chosen so every *unobserved* node has at
  least one *observed* neighbour. This is the mechanism that handles sparse observability.
- **Type F — Stage II:** a second GNN over a **sample graph** (each vertex = one data sample),
  with adjacency from subspace-angle similarity of Stage-I embeddings, masked to nodes near the
  Stage-I prediction. Label propagation to unlabeled samples.
- Theory via **random-walk equivalence** of the two adjacency matrices.
- **Results:** LAR 92.9% at 15% label rate, 95.7% at 3%; robust to load variation and topology
  change (first 6 switches opened).

**Overlap with you:** their Stage-I adjacency and your measured-only graph solve the same
problem differently — they connect unobserved nodes by shortest path, you delete them. Their
`exp(-d²/δ²)` distance kernel and your FaultQuery `-softplus(α)·D` attention bias are the same
physical prior in different slots.

**Gaps PPGN leaves you:**
- It assumes voltage **angles**. (See §1.)
- It never studies **sensor dropout** — observability is sparse but *fixed*. Your entire
  contribution axis is untouched by it.
- It has no **noise** study.
- Its physics is static-snapshot; there is no temporal encoder. Your GRU window is orthogonal.

### 4.2 Other directly relevant PIGNN work

- **Physics-Informed Graph Learning for Robust Fault Location** (Li & Deka) — the two-stage
  framework PPGN extends; >97% accuracy at 15% labels on IEEE-123.
- **Physics-Informed Inductive Biases for Voltage Prediction in Distribution Grids**
  (arXiv 2509.25158) — usefully *negative*: of three biases, **complex-valued networks** win
  big on angles, **physics loss** gives the most stable magnitudes but is "highly reliant on
  hyperparameters", and **residual learning gives negligible gains**. Combinations are
  "non-additive and grid-specific". Cite this to justify testing PIGNN variants one at a time.
- **Physics-Informed Graph Learning: A Survey** (arXiv 2202.10679) — the taxonomy anchor.
- **Engineering Hybrid PINNs for Next-Generation Electricity Systems** (arXiv 2605.21903) —
  2026 state-of-the-art review; names PIGNNs as one hybrid family among DeepONet/FNO/XPINN.
- **Robust Fault Location Under Noisy and Incomplete Measurements via Physics-Aware Decoupled
  Inference** (Energies 2026, 10.3390/en19112583) — closest to your noise+dropout framing.
  MDPI blocked automated fetch; **retrieve manually, it may be a direct competitor.**
- **KCLNet** (arXiv 2506.12902) — hard Kirchhoff-current-law constraint via projection; the
  cleanest Type-B exemplar if you get angles.
- **Feature Propagation** (arXiv 2111.12128) — Dirichlet-energy diffusion for missing node
  features. This is the *learned imputation baseline your own CLAUDE.md lists as a named gap.*
  Cheap to add, closes a stated hole.

---

## 5. What you have already built (local audit)

Two undocumented-in-CLAUDE.md architectures exist on disk, both uncommitted-to-paper:

### `fault_localization/models/fault_query.py` — FaultQuery-STGNN
26 learned fault-bus queries, cross-attention over the 25 sensors, attention biased by
hop-distance (Type D), learned reliability, and hard availability masking **in the decoder
only**. 118,279 params (+24.2%). 15 unit tests pass.

**Pilot status — diverged, not a valid result.** `results/fault_query/.../metrics_by_epoch.csv`:
val F1 0.567 (ep 1) → 0.549 → 0.145 → 0.056, train loss 0.65 → 0.85 → 1.73. The 0.5691 in
`metrics.json` is the **epoch-1 checkpoint of a diverged run** and must not be quoted as this
model's score. Needs an LR/stability fix before any conclusion.

### `fault_localization/models/zone_refine.py` — ZoneRefine-STGNN
**This is already a Type-B physics-encoded PIGNN, and you did not label it as one.** It uses
the first-order radial-feeder factorization `Δv_i ≈ a · s_{b,i}` from `graph/sensitivity.py`,
partitions the 26 classes into 16 **electrical zones** by signature collinearity, and factors
the prediction as `p(class) = p(zone | physics residual) · p(class | zone, learned)`.

Its dropout property is the elegant part: Stage 1 fits templates by **masked least squares**,
so a dropped sensor *removes an equation* rather than injecting an out-of-distribution zero.
That is a principled alternative to gating, augmentation, and imputation alike.

**Pilot status — valid and promising.** Full 30 epochs, seed 42: **test graph macro-F1 0.803**,
best val 0.811. Still climbing at the budget's end (val 0.798 → 0.811 → 0.811 over the last
three epochs), so 0.803 is a **floor**, not a converged score. Against RGSAGE-Max's 0.846 it is
−0.043 at n=1 — below the 0.83 gate, above the 0.75 abort line.

**This is your most valuable existing asset for a PIGNN paper**, and it is one seed and one
under-trained run away from being reportable.

---

## 6. The strategic argument — what a new architecture should actually target

Your own results say where the room is:

| Axis | Best current answer | Room left |
|---|---|---|
| Clean accuracy (k=0) | RGSAGE-Max 0.846 | **~0.13 over logreg — nearly none** |
| Random dropout | dropout-aware training, Ret@10 80.3%, AUDC 0.913 | little; SDR-STGNN matched and did not beat |
| **Measurement noise** | **nothing works** | **wide open** |
| **Targeted (worst-case) dropout** | DA training penalized: AUDC 0.913 → 0.747 | **open** |
| **Scenario generalization** | untested (retrains queued) | **open** |
| Label efficiency | untested | **open — and PPGN's whole claim** |

The noise row is the important one. From `results/dropout_study_noise/`, standard RGSAGE-Max at
k=0 goes **0.846 → 0.786 → 0.579 → 0.345** at σ = 0, 5e-4, 2e-3, 1e-2 p.u. Dropout-aware
training is **not better at any σ** (−0.008 to −0.034, all CIs span zero, p ≥ 0.107) — a clean
null, because noise leaves the availability mask at 1 and the learned "(0,0,0) = dark" response
has nothing to fire on.

**A binary availability gate cannot fix noise, by construction. A physics residual can** —
because a noisy-but-present sensor produces a measurable inconsistency with its neighbours under
`Δv ≈ a·s_b`, whereas it produces no mask signal at all.

That is the thesis of the next paper, and it is falsifiable.

---

## 7. Recommendation

**Combine and extend — do not build from scratch.** Three reasons, all evidential:

1. **§3.1** — clean-accuracy headroom over a *linear model* is 0.133. An architecture designed
   to win there will produce differences smaller than your seed noise.
2. **Your own precedent** — SDR-STGNN was a from-scratch architectural idea that matched but did
   not beat a *training-time* fix, at +18% latency. Repeating that pattern is the predictable
   outcome of another clean-slate design aimed at the same stressor.
3. **Controlled comparison requires a shared backbone.** Keeping GRU-128 + RGSAGE-Max means
   every PIGNN variant is paired against the existing 3-seed baselines under identical splits
   and mask seeds. A from-scratch model forfeits that and turns every result into a confound.

**The paper to write:** *a controlled study of physics-informed mechanisms for fault
localization under sensor dropout and measurement noise* — holding the backbone fixed and
varying the physics-injection **type** (§4 table). That directly answers "PIGNNs are of
different types, we need to experiment" and turns the type-multiplicity from a problem into
the contribution.

Framing: **the axis nobody has tested is noise**, and you have the null result that motivates it.

---

## 8. Proposed experiment plan

**Decision gate 0 (do this first):** time one OpenDSS event, multiply by 41,250, and decide on
angle recovery (§1). Everything in Tier 2 depends on it.

### Tier 1 — no new data, uses what is on disk

| # | Experiment | Type | Why |
|---|---|---|---|
| 1.1 | **ZoneRefine to 3 seeds + longer budget** | B | validated pilot at 0.803 and still climbing; cheapest path to a real result |
| 1.2 | ZoneRefine full dropout / targeted / **noise** curves | B | tests the §6 thesis; masked-LS should degrade gracefully |
| 1.3 | Ablations Z2 (frozen templates) / Z3 (no physics zones) | B | isolates the *physics* from the two-stage structure — already coded |
| 1.4 | Fix FaultQuery divergence, then 3 seeds | D | it is written and tested; a diverged run is not a result |
| 1.5 | **Feature Propagation imputation baseline** | — | closes the "learned imputation" gap CLAUDE.md already names |
| 1.6 | PPGN-style Type-C adjacency: `exp(-d²/δ²)` edge weights on the measured-only graph | C | one-line-ish change to `edge_attributes.py`; direct comparison to PPGN |

### Tier 2 — requires angle recovery

| # | Experiment | Type | Why |
|---|---|---|---|
| 2.1 | Kirchhoff residual `‖Y·ΔU − ΔC − Δ_ij‖` as auxiliary loss | A | the real physics loss, finally defined |
| 2.2 | Hard KCL projection layer (KCLNet-style) | B | soft vs hard constraint, the field's central question |
| 2.3 | Complex-valued temporal encoder | B | the one bias arXiv 2509.25158 found decisively effective |
| 2.4 | Physics residual as a **noise detector** feeding FaultQuery's reliability head | A+D | the mechanism §6 predicts; currently untrainable because nothing supervises `r_i` |

### Tier 3 — optional, positions against PPGN

| # | Experiment | Type |
|---|---|---|
| 3.1 | Label-rate sweep (75/50/25/15/3%) | — |
| 3.2 | PPGN Stage-II sample-graph label propagation | F |

**Protocol, non-negotiable:** one variable at a time; identical frozen corpus, splits and mask
seeds; seeds 42/43/44; run everything on `dataset_fullscale_50_scenario` too, since a new
architecture is exactly what could exploit the 825/825 scenario overlap.

**Parameter-matched control:** any PIGNN that wins on accuracy needs a parameter-matched plain
RGSAGE-Max control before the claim stands. FaultQuery is +24.2% params; that confound is real.

---

## 9. Honest risks

- **PPGN may already be the answer.** If a Type-C adjacency alone recovers most of the gain,
  your contribution shrinks to "PPGN's physics also helps under dropout" — still publishable,
  but as an extension, not a new architecture.
- **The Energies 2026 paper is unread** (MDPI blocked the fetch). Retrieve it before committing;
  "noisy and incomplete measurements" + "physics-aware" is uncomfortably close to your plan.
- **Physics losses are hyperparameter-fragile** — arXiv 2509.25158 says so explicitly. Budget
  for λ tuning, and report the sensitivity rather than the best λ.
- **The ceiling may be data, not architecture.** 25 sensors, 20-timestep windows, and logreg at
  0.713. If Tier 1 shows every variant landing inside ±0.02, that is itself the finding — and a
  more useful one than a fourth architecture that ties.
- **n=3 cannot resolve small effects.** Anything under ~0.05 will not survive Holm. Design the
  experiments to produce large effects (noise, low label rates) rather than small ones (clean F1).

---

## 10. EXPERIMENTAL RESULTS — the selection screen (run 2026-09-03/07)

Everything below is on disk. Reproduce with `scripts/run_pignn_screen.py`,
`scripts/run_physics_matched_filter.py`, `scripts/build_pignn_screen_summary.py`.

### 10.1 What was built

Three *minimal grafts* on one fixed backbone (GRU-128 + 4-hop RGSAGE-Max), each
changing exactly one thing. `models/pignn.py` holds P1/P2 as independent
switches; `models/physics_loss.py` holds P3.

| Arm | Type | Mechanism | Extra params |
|---|---|---|---|
| P1 | C — graph | messages scaled by `exp(-softplus(gamma)*|Z|/median|Z|)` | **+1** |
| P2 | D — prior | logits biased by `beta * S[c,i]` (shared-path sensitivity) | **+1** |
| P3 | A — loss | `CE + lambda*L_phys`, sag-pattern cosine residual, lambda=5.0 | **0** |

With both switches off the model is **byte-for-byte RGSAGE-Max** (asserted,
max|diff| = 0.0). 95,258 -> 95,259 params. **There is no capacity confound
anywhere in the screen** — unlike FaultQuery's +24.2%.

### 10.2 Clean accuracy — a null across all three arms

Seed 42, 15 epochs, compared against RGSAGE-Max's own epoch-15 validation F1
(0.8162), read from its `metrics_by_epoch.csv`. No baseline re-run.

| Arm | best val | delta | test |
|---|---|---|---|
| baseline RGSAGE-Max | **0.8162** | — | — |
| P3 loss | 0.8111 | −0.0051 | 0.8061 |
| P2 prior | 0.8110 | −0.0052 | 0.8091 |
| P1 graph | 0.8069 | −0.0093 | 0.8005 |

P2 and P3 differ by 0.0001 — far inside the ~0.007 seed spread. **No arm beats
the baseline.** This is exactly what §3.1 predicted: with logreg at 0.713 the
whole deep-model margin is 0.133, and nothing decidable lives inside it at n=1.

### 10.3 The learned physics scalars — the actual finding

Both priors were made learnable and softplus/sign-free precisely so the model
could reject them. It did, in two different ways:

```
P1  gamma:  +1.0000 -> -17.3770   (softplus 1.3133 -> 0.0000)   REJECTED
P2  beta:   +0.1000 ->  -1.8982   (sign flipped, 19x magnitude) INVERTED
```

**P1 zeroed its prior**, recovering uniform weighting — and paid for it: it was
−0.176 behind at epoch 1 and never fully caught up in 15 epochs. **Diagnosis:
P1 encodes the wrong physical quantity.** It weights by the impedance of the
edge *between two adjacent sensors*, but `graph/sensitivity.py` says sag
correlation is governed by the impedance of the path two buses **share with the
substation** — a mutual quantity, not a pairwise-adjacent one. The rejection is
the physically correct outcome. A physics prior must match the governing
relation, not merely be physics-flavoured.

**P2 kept its prior and inverted it.** The model uses the sensitivity matrix
hard, in the opposite direction to the naive "a sensor should vote for what it
sees strongly". Likely reading: the signature rows are highly collinear (median
pairwise cosine 0.944), so a negative bias suppresses the common mode and acts
as contrast enhancement — the discriminative information is which sensors
*fail* to sag. Worth stating as an observation, not a mechanism, at n=1.

**P3's `train_phys_loss` saturated**: 0.0796 -> 0.0745 by epoch 4, then flat for
11 epochs. The constraint is satisfied early and stops exerting pressure.

### 10.4 Convergence — real, but do not overclaim

P3 led at 11 of 15 epochs (+0.099 at e7, +0.063 at e9) and reached 0.8111 at
**epoch 9** where the baseline needed **epoch 12**. But P3 then oscillated around
0.80 for six epochs while the baseline climbed to 0.8162. Combined with the
saturating physics term, the honest read is that the physics arms **accelerate
early training but cap slightly lower** — not that they converge faster to the
same point. A 30-epoch run is needed before any convergence claim.

### 10.5 The physics matched filter — a new training-free baseline

`scripts/run_physics_matched_filter.py`. Zero learned parameters; one threshold
`tau` fitted on **validation** and held fixed across every k and sigma.
Decision rule: `||d||_a < tau` -> no-fault, else `argmax_c cos_a(d, s_c)`.

**Dropout — completely flat:**

| k | standard RGSAGE-Max | matched filter | ratio |
|---|---|---|---|
| 0 | 0.8463 | 0.1618 | 0.19x |
| 5 | 0.2360 | 0.1676 | 0.71x |
| 10 | **0.0481** | **0.1652** | **3.43x** |

Retention at k=10 is **102%** (std over mask seeds <= 0.0007), against 5.7% for
standard and 80.3% for dropout-aware. A masked cosine *drops terms from a sum*;
a network gets fed zeros it never saw in training. Report as **flat**, not as
improvement — tau was fitted at k=0.

**Noise — and the decomposition that matters:**

| sigma (p.u.) | macro-F1 | **localization only** |
|---|---|---|
| 0.0005 | −2.0% | **0.0%** |
| 0.002 | −30.6% | **−2.7%** |
| 0.01 | −39.9% | **−11.7%** |

The filter's *localization* is nearly noise-immune; its macro-F1 loss is almost
entirely the fixed detection threshold, since noise inflates deviation energy
and pushes no-fault windows over a tau fitted at sigma=0. The trained model
loses **−59.3%** over the same range. At sigma=0.01 with k=10 the trained model
scores **0.0177 — below the 0.0256 trivial floor** — while the physics scores
**0.1024, 5.79x higher**.

Figures: `paper/figures/physics_baseline_degradation.png`,
`paper/figures/physics_noise_dropout.png`, `paper/figures/pignn_screen.png`.

### 10.6 A bug worth recording, because it nearly produced a wrong conclusion

The first `observed_deviation` measured sag against the **corpus channel means**
(0.83/0.60/0.76 p.u.), which sit far below nominal because absent phases are
stored as 0.0. The relu then zeroed **97.6%** of node entries, leaving **0.60 of
25** sensors with any signal. With `d` all-zero the cosine is 0 for every class,
the residual is a constant 1.0, and **P3's loss did nothing but push p_0 up** —
it was actively training the model toward "no fault".

All 19 synthetic unit tests passed while this was true. It was caught only by
evaluating the matched filter on the **full** test split, where it scored 0.058
against an earlier subset probe that had suggested 77.5% top-1 — that probe was
doubly artifactual (13 of 26 labels present, and a rank formula that counted
ties as rank 1). **Never size this from a subset**, exactly as `CLAUDE.md` warns.

Fix: a reference-free within-window sag depth, `max_p [max_t v - min_t v]` over
available phases. Justified by the windowing — the fault starts at step 40 and
windows are 20 steps with starts 0-39, so **every fault-labelled window contains
at least one pre-fault sample**. Chosen on measurement over three alternatives
(14.8% top-1, 0% degenerate windows; a `max-last` variant scored 16.3% but was
degenerate on 25.3% of windows, which would poison a loss term).

Two regression tests now run on the **real corpus** (`tests/test_pignn.py`,
marked `slow`): the deviation must be informative on >80% of live sensors, and
the matched filter must beat chance with <5% all-tie windows.

### 10.7 Where this leaves the paper

Two components with **exactly complementary failure modes**, both measured:

| | clean F1 | dropout Ret@10 | noise (localization) |
|---|---|---|---|
| learned RGSAGE-Max | **0.846** | 5.7% | −59.3% |
| physics matched filter | 0.162 | **102%** | **−11.7%** |

That is the argument for hybridization stated in measurements rather than
assertion. The screen shows the three obvious ways of combining them do **not**
buy clean accuracy; whether any buys *robustness* is settled by the noise x
dropout grid over the three trained checkpoints (`results/pignn_noise/`), which
is inference-only and therefore cheap — and is the selection criterion that
actually matters, rather than a 0.0001 clean-accuracy tie.

---

## 10A. FINAL RESULTS AT n=3 — the screen closes as a negative result

Supersedes the n=1 / 15-epoch numbers in §10.2-10.4 wherever they disagree.

### 10A.1 Clean accuracy — null, and P3 matches the baseline

P3 at the frozen 30-epoch recipe, seeds 42/43/44, against RGSAGE-Max's own 3 seeds:

| | clean test macro-F1 (n=3) |
|---|---|
| standard RGSAGE-Max | 0.8463 +/- 0.0070 |
| P3 (physics loss) | **0.8469 +/- 0.0183** |
| dropout-aware RGSAGE-Max | 0.8388 +/- 0.0111 |
| P3 + dropout-aware | 0.8374 +/- 0.0111 |

- P3 vs standard: **+0.0006**, 95% CI [-0.040, +0.041], p = 0.958
- P3+DA vs DA: **-0.0014**, p = 0.452
- P3+DA vs P3: -0.0095, p = 0.185

**Correction to §10.4.** The 15-epoch screen showed P3 at -0.005 and I read that as
a slightly lower ceiling. It was a *budget artifact*: at 30 epochs P3 closes the
gap exactly. There is no ceiling claim; there is no gain either.

**P3's seed variance is 2.6x the baseline's** (+/-0.0183 vs +/-0.0070) and is
larger at nearly every noise/dropout cell too. The physics loss makes training
*less* stable.

### 10A.2 The noise signal REVERSED at n=3 — do not cite the n=1 number

The single most important correction in this document.

| sigma=0.01, k=0 | P3 | baseline | diff |
|---|---|---|---|
| n=1, 15 epochs (§10 earlier draft) | 0.3781 | 0.3446 | **+0.0335** |
| **n=3, 30 epochs** | 0.3131 +/- 0.0344 | 0.3446 +/- 0.0416 | **-0.0315** |

The sign flipped. The apparent "first mitigation that helps against noise" was a
single-seed artifact of an under-trained model. **Nothing in the 20-cell
(sigma x k) grid reaches p_Holm < 0.10.** The only nominally significant cell is
sigma=0.002 / k=0 at p=0.062, and it is P3 *worse*.

P3 is nominally ahead at k=10 for all four sigmas (+0.016 to +0.017, oddly
uniform), but those cells sit at F1 0.02-0.06 — at or barely above the 0.0256
trivial floor. That is variance around a dead model, not robustness. The same
caveat retires P1's apparent k=10 "wins".

### 10A.3 What the screen established

Four arms, every one parameter-matched or parameter-free, every one null:

| Arm | Injection | Learned scalar | Verdict at n=3 |
|---|---|---|---|
| P1 | graph (Type C) | `gamma` -> **0.0000** | prior **rejected** by the model |
| P2 | prior (Type D) | `beta` +0.1 -> **-1.90** | prior **inverted**, never wins |
| P3 | loss (Type A) | — (`phys_loss` saturates by ep 4) | **null** on clean, noise, dropout |
| P3+DA | loss + augmentation | — | **null** vs DA alone (p = 0.452) |

**This is a complete negative result, and a stronger one than the paper's
existing single data point.** SDR-STGNN already showed one architectural idea
matching but not beating a training-time fix. This adds three structurally
different physics injections — in the graph, in the prior, and in the loss —
each with a mechanism explaining the failure, none beating the same training-time
fix, and none costing more than one scalar parameter.

**Diagnosis, stated once:** P1 encoded the wrong physical quantity (pairwise edge
impedance, where the governing relation is substation-shared path impedance). P2
found the sensitivity matrix informative but only with inverted sign, consistent
with common-mode suppression rather than the intended "vote for what you see".
P3's constraint is satisfied within four epochs and then exerts no further
pressure. Underlying all three: the physics available from **voltage magnitudes
alone** is weak — the matched filter reaches 14.8% top-1 against 4% chance — so
there is little for a prior to contribute that the data does not already supply.

### 10A.4 What survived as a positive contribution

Only the training-free baseline (§10.5), and it survived well:

- **dropout-invariant**: 102% retention at k=10 (std <= 0.0007) where standard
  RGSAGE-Max retains 5.7%; at k=10 it scores 0.165 vs the trained model's 0.048.
- at sigma=0.01 **and** k=10 the trained model scores 0.0177, **below** the
  0.0256 trivial floor, while the matched filter scores 0.1024 — **5.8x higher**.
- its noise loss is **detection, not localization**: -39.9% macro-F1 but only
  -11.7% fault-window accuracy, because tau was fitted at sigma=0 and held fixed.

### 10A.5 The honest conclusion for the next paper

The magnitude-only physics does not help a model that already has the
measurements. The open question is not *where* to inject this physics — that is
now answered three ways — but *whether stronger physics changes the answer*, and
that requires the phase angles the simulator computes and discards (§1).

Until angles are recovered, further magnitude-only PIGNN variants are predicted
to be null, and this survey recommends against spending GPU on them.

---

## 10. Sources

Base + follow-up: arXiv [2604.20403](https://arxiv.org/abs/2604.20403),
[2510.03571](https://arxiv.org/abs/2510.03571).
PIGNN in this task: [PPGN 2107.02275](https://arxiv.org/abs/2107.02275),
[Physics-Informed Graph Learning for Robust Fault Location](https://www.semanticscholar.org/paper/744b95344083893c24bc8f78e21f7293de643d0d),
[Energies 2026 10.3390/en19112583](https://www.mdpi.com/1996-1073/19/11/2583).
Taxonomy/reviews: [2202.10679](https://arxiv.org/abs/2202.10679),
[2605.21903](https://arxiv.org/abs/2605.21903),
[PINNs for power systems: systematic review](https://www.sciencedirect.com/science/article/abs/pii/S2352467726001438).
Mechanisms: [2509.25158](https://arxiv.org/abs/2509.25158),
[KCLNet 2506.12902](https://www.arxiv.org/pdf/2506.12902),
[Feature Propagation 2111.12128](https://arxiv.org/abs/2111.12128).
Other architectures: [Graph Mamba 2402.08678](https://arxiv.org/abs/2402.08678),
[GNAS for distribution fault detection](https://spj.science.org/doi/10.34133/icomputing.0229),
[GNNs in power systems review 2101.10025](https://arxiv.org/abs/2101.10025).
