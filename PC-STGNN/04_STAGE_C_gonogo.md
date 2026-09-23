# Stage C — Training-Free Science (Weeks 4–6). The Main GO/NO-GO

**Purpose.** Decide whether the project proceeds. Everything here runs on existing checkpoints and
CPU.
**Compute.** No GPU training. ~4–5 h of OpenDSS for the OOD sets (see S2.0).

## Context you need

`01_SPEC.md` §4 for the residual. The operator passed Gate 1. Existing dropout-aware RGSAGE-Max
checkpoints at seeds 42/43/44 (`results/dropout_aware/rgsage_max/`). Training R ∈ {0.1, 1, 10} Ω.

**Priors already on disk** `[Corrected 2026-09-22]`:
- A rank-1 sag-template matched filter (the classical direction-only method, no τ/R/L profiling)
  scores **macro-F1 0.162 at every k** (`results/physics_baseline/matched_filter.json`). This is
  the B1 floor.
- Logistic regression on 375 summary features scores 0.713 at k = 0
  (`results/classical_baseline/`); RGSAGE-Max 0.846.
- The corpus is noise-free and its only within-scenario randomness is the scalar L, so a table
  from the same simulator with L profiled is close to an exact generative model. **Expect the
  "PSM ≈ or > STGNN at k = 0" row of Gate 2 to be the base case**, and plan the paper's centre of
  gravity around noise, dropout, silent failures and model mismatch from the start.

## S2.0ᵃ — The operator is the generator `[Added 2026-09-23 — read before any Stage C result]`

S0.4 measured the table against the corpus at **1.02e-12 p.u.** The operator is not an
approximation of the corpus; it is the corpus's own generator re-run. That is a calibration
problem for σ_ε (`01_SPEC.md` §4) and, more importantly, it makes **every clean, in-distribution
comparison in this project degenerate**. Four consequences, stated here once so that no stage
re-derives them:

1. **S1.4's bias check is vacuous** — it returns ~1e-12 by construction. Gate 1's
   `median |e| / sag < 0.3` criterion no longer tests anything; fidelity is true by construction,
   not by measurement. Good engineering, bad science: the falsifiable test Gate 1 existed to run
   is gone, and Gate 1's content moves to S1.5 / S1.7 / the ρ-margin band (`03_STAGE_B` Gate 1).
2. **PSM at k = 0 on the clean corpus is an oracle, not a baseline.** ρ at the true hypothesis is
   exactly zero and every rival is bounded away, so it scores ≈ 100 %. Presented as a baseline
   win, it earns exactly one reviewer sentence — *"the authors built a lookup table from the
   simulator that generated the data, then showed the lookup table wins"* — and that sentence
   ends the paper.
3. **The OOD-resistance experiment is degenerate for PSM too.** The OOD sets come from the same
   generator and the R grid spans 0.05–30 Ω, so PSM is an oracle there as well. The
   *network's* collapse under unseen R is still true and still interesting, but the contrast can
   no longer be sold as "physics generalises": physics was handed the answer key.
4. **The fusion cannot learn anything on clean data.** With ρ_true = 0 and ρ_wrong in the
   thousands, `softmax(−ρ)` is one-hot; β learns to trust physics completely, the network's
   contribution goes to zero, and PC-STGNN becomes PSM with extra steps. **No choice of σ_ε fixes
   this** — compressing the scale shrinks the true and wrong residuals together.

### What survives — and it is the better paper

In each of these the operator is *not* the generator of what the model sees:

| Family | Why it survives |
|---|---|
| **Model mismatch** (S6.2) | Perturbed table ≠ generator. **Promoted from robustness appendix to primary axis** (`08_STAGE_G_eval.md`) |
| **Measurement noise** | Observation ≠ what the generator produced |
| **Silent corruption** (Stage F) | Same |
| **Identifiability** (S2.1) | *Strengthened.* With an exact operator, D(f₁,f₂;A) measures the pure information content of the sensing infrastructure, with zero model error contaminating it — arguably the project's cleanest contribution |
| **Dropout** | Partially. PSM degrades only through lost identifiability, not model error — a legitimate and interesting reading, but **frame it as an identifiability result** |

### Three changes this forces, made now rather than at Gate 4

1. **Noise augmentation is the default training condition for PC-STGNN**, not an ablation. On
   noise-free data the fusion has nothing to learn (consequence 4). Noise-free training becomes
   the ablation. This reallocates seeds — see `06_STAGE_E_pcstgnn.md`.
2. **The nominal table is an explicitly labelled oracle column.** Headline tables report the
   realistic operating point (noise + perturbed table); the exact-operator / noise-free column
   sits beside it labelled **"oracle upper bound: perfect model knowledge, perfect
   measurements."** This turns the problem into an asset — a free oracle/realistic pair that
   almost no paper in this area can produce.
3. **Identifiability is promoted from supporting figure to contribution.** *"How much can 25
   sparse, magnitude-only µPMUs reveal about fault location, measured with an exact physics
   operator"* is defensible, novel, and reviewer-proof because it does not depend on any model
   gain (`09_STAGE_H_writing.md`).

## S2.0 — Generate the OOD set, and a throwaway pilot

**Do.** Two separate generations with different RNG seeds:

| Set | Contents | Fate |
|---|---|---|
| OOD-pilot | R ∈ {5, 20} Ω × 25 locations × 11 types × ~2 runs ≈ 1,100 events | used at Gate 2, then discarded, never reported |
| OOD-test | R ∈ {3, 5, 20, 30} Ω × 25 × 11 × 10 runs ≈ 11,000 events | sealed until Stage G |

Same generator (`scripts/generate_trend_dataset.py` path), same load distribution.
`[Corrected 2026-09-22]` Each event is 60 Daily-mode steps, so this is ≈ 730 k solves ≈ 4–5 h,
not 0.3–1.5 h. Measure the rate in S0.5.

**Why two sets.** You need OOD evidence to make the Gate-2 decision, but touching the real OOD set
this early burns it — any decision made after seeing it contaminates it. The pilot buys the
signal; the sealed set carries the claim.

**Check.** Pilot and test sets share no seeds. The sealed set is write-protected and its hash
recorded.

**Output.** `data/ood_pilot/`, `data/ood_test/` (sealed), `data/ood_test.sha256`
**Table produced.** T2 — dataset + OOD set configuration.

## S2.1 — Identifiability matrix

**Do.** For every location pair (including f = 0) compute the separability score:

```
D(f₁,f₂;A) = min over (τ,R,L) pairs of  Σ_{(i,p)∈A}  (μ_f₁ − μ_f₂)² / (2 σ_ε²)
```

The min over hypotheses makes it the hardest case — the most confusable (τ, R, L) combination.
Average over random sensor subsets at k ∈ {0, 2, 5, 10}, reported separately per R.

**Check.** 26 × 26 heatmap per k. Count pairs with D < 1 (indistinguishable at the floor scale).
Report min_f₂ D(f₁,f₂) per location.

**Interpretation rule.** Judge D against σ_ε, and — under the noise experiments — against
σ_ε² + σ₀². On a noise-free corpus everything looks separable if you compare to zero; the
noise-level identifiability map is the one that matters for the paper.

**Figures produced.** F-C1 — identifiability heatmaps at k = 0, 2, 5, 10. F-C2 — fraction of
indistinguishable pairs vs k, per R and per σ₀.

## S2.2 — Empirical confusability

**Do.** Independently of the table: nearest-neighbour confusion between training events at
different locations, using raw observation vectors.

**Check.** Does the empirical confusion structure match F-C1?

| Table says | Corpus says | Diagnosis |
|---|---|---|
| separable | confusable | your table is wrong → back to Stage B |
| confusable | confusable | genuine information limit → publishable in its own right |
| separable | separable | physics is informative → proceed |

**Figure produced.** F-C3 — table D vs empirical confusability scatter.

## S2.3 — B1: ΔV nearest-signature baseline

**Do.** Δs = s − v_nominal, with v_nominal the per-channel no-fault mean from the training split.
Nearest table signature at nominal load; no profiling over τ, R, or L. Also report the existing
matched filter (`scripts/run_physics_matched_filter.py`) as B1ᵃ — it is the same idea with a
rank-1 template and is already evaluated at every k.

**Check.** Full Paper-1 evaluation: F1 at every k, F1 by R and type, AUDC (relative and absolute).

**Purpose.** This is the classical method. B1 → B2 measures exactly what profiling and the load
axis buy you — the empirical defence of the operator design.

## S2.4 — B2: PSM (Physics Signature Matching)

**Do.** f̂ = argmin_f r̃_f(A) over f = 0…25, no network at all. Full Paper-1 evaluation. Also run
on the OOD-pilot.

**Check.** Against B1 and against all Paper-1 models (standard and dropout-aware).

**Why mandatory.** Without it a reviewer says "your physics module is doing all the work; the GNN
is unnecessary." The scientifically interesting outcome is PSM < STGNN < PC-STGNN — but see the
priors above: PSM ≥ STGNN at k = 0 on the clean corpus is likely, and then the question becomes
where learning still helps.

**Metrics.** Additionally report PSM top-3 accuracy and the rank of the true location under r̃.

## S2.5 — The extrapolation contrast — the most important experiment in Stage C

**Do.** Evaluate the existing dropout-aware RGSAGE-Max checkpoints on the OOD-pilot.

**Check.** Compare the neural model's F1 drop (in-distribution → OOD) against PSM's.

**Why.** The network was trained on {0.1, 1, 10} Ω. PSM has a dense continuous representation of
R. If the network collapses on unseen resistance and PSM does not, the paper's central claim is
demonstrated before a line of model code is written.

**Figure produced.** F-C4 (pilot version; the paper version is F-G3 from the sealed set).

## S2.6 — Inference-only fusion

**Do.** Apply the per-node fusion z'_{n,f} = z_{n,f} − β·ρ_f(A) of `01_SPEC.md` §6 to the
existing R0 checkpoints. Grid-search β on the
validation split only. Run it clean **and** at σ₀ ∈ {5e-4, 2e-3, 1e-2} p.u. (the existing noise
grid, `results/dropout_study_noise/`), setting σ₀ in the residual accordingly.

**Check.** ΔF1 at k = 0, at k = 10, at 10 Ω, and at each σ₀ — all on validation.

**Pass.** Any consistent positive Δ at 10 Ω, k = 10 or under noise, sign-consistent across the
three seeds.

**Time.** Minutes. The cheapest possible test of *does physics know something the network
doesn't*. The noise arm is the one Paper 1 could not answer: dropout-aware training does not help
against noise at all (−0.008 to −0.034, all CIs span zero).

## S2.4ᵇ — How PSM's clean number is reported `[Added 2026-09-23]`

PSM scoring near-perfectly at k = 0 on the clean corpus is a **tautology, not a result**: the
table comes from the same simulator that generated the corpus, the corpus is noise-free, the only
within-scenario randomness is the scalar L that the operator profiles, and σ_ε ≈ 1e-3 p.u. A
generative model evaluated on its own generator is expected to win.

So, from this stage onward and in every table and every sentence of the paper:

- PSM's clean k = 0 number is reported as an **operator-fidelity sanity check**, labelled as such
  in T6's caption, and is never described as the physics "beating" the network;
- **noise (σ₀) and model mismatch (S6.2) are primary axes, not appendix sub-tables.** Physics
  degrades under noise and under a wrong table; the network degrades under OOD resistance and
  severe dropout. The hybrid's claim is that it owns the union;
- the Gate-2 question is therefore not "is physics better?" but **"physics and learning fail under
  different perturbations — does fusing them cover both?"**

Noise levels stay on Paper 1's existing grid σ ∈ {5e-4, 2e-3, 1e-2} p.u. so that the standard and
dropout-aware curves are reused rather than re-run. State their realism against µPMU accuracy
classes: IEEE C37.118 caps total vector error at 1 %, so 1e-2 p.u. is the *spec limit*, not an
extreme — and at 10 Ω the sag is only ≈ 0.01–0.02 p.u., making σ = 1e-2 a ~1–2 : 1 SNR regime and
the most interesting operating point in the study. Check instrument datasheets before that
sentence is written.

## S2.7 — Pre-registration

**Do.** Before opening any test-split results, fill in `14_PREREGISTRATION.md` — dated,
committed to the repo.

**Why.** An hour of work, and it is the difference between "we found an effect" and "we searched
until we found one." It also makes the results section write itself.

## GATE 2 — The Decision

Using validation split and OOD-pilot only. `[Corrected 2026-09-23]` **The decision is read off
the noise and model-mismatch arms, never off clean k = 0** — clean, in-distribution numbers are
oracle references by construction (S2.0ᵃ) and cannot discriminate between these rows:

| Observation | Reading | Action |
|---|---|---|
| S2.6 improves F1 at 10 Ω, k = 10 or under noise | Physics carries information the network lacks | **GO** — full plan |
| S2.6 flat, but PSM holds on the OOD pilot while the network collapses (S2.5) | The network already learned in-distribution physics; physics buys extrapolation | **GO** — reframe around OOD resistance + silent failures. Stages D–H unchanged |
| PSM ≈ or > STGNN everywhere on the clean corpus | **The expected base case, and a tautology** — same simulator, scalar load, no noise (S2.0ᵃ, S2.4ᵇ). This row is an *oracle reference*; it is not evidence either way | **GO** — the question becomes "where does learning still help?": noise, dropout, silent failures, model mismatch (S6.2). Honest, still a good paper |
| PSM weak and many pairs with D < 1 at k = 10 | At the information limit of 25 magnitude-only µPMUs | **PIVOT** to the identifiability + silent-failure paper. Stage F still runs; Stage E is dropped |

**Deliverables.** F-C1…F-C4 · T6 baseline ladder (B1, B1ᵃ, B2 rows) · the S2.6 numbers · the
pre-registration note.
**Time:** 2–3 weeks. GPU used: none.
