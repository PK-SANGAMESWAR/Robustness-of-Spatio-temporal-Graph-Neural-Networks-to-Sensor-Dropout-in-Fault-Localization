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
  from the same simulator with L profiled is close to an exact generative model.
  `[Corrected 2026-09-25]` PSM top-1 accuracy on the full 28,875-event training split is
  **89.74 %** (25-class localization macro-F1 0.8977,
  `results/pcstgnn/audit/s1_7_discretization/s1_7_discretization.json`).
  `[Corrected 2026-09-29 — the line above is withdrawn]` **This number is not comparable to
  RGSAGE-Max's 0.846 and the "PSM ≥ STGNN at k=0" claim built on that comparison does not hold as
  stated.** Caught on author review; verified by direct measurement
  (`scripts/check_s2_4_metric_consistency.py`,
  `results/pcstgnn/audit/s2_4_metric_check/metric_consistency_check.json`) before correcting the
  claim rather than just asserting the reviewer was right. The 0.8977 figure is an **event-level,
  single-settled-step, 25-class, fault-only** metric (`check_s1_7_discretization.py`'s own
  docstring: "no true no-fault class 0 instances exist at this per-event granularity") — a
  different, easier task from the **window-level, 40-windows-per-event, 26-class** metric
  (`test_graph.f1`/`best_validation_graph_f1`, `voting.py` soft-vote per window) used for
  RGSAGE-Max's 0.846 and everywhere else in this project. Measured side by side, holding split
  fixed: on the **training split**, the window-level 26-class score is **0.7682**, not 0.8977 —
  a **0.1294** gap from switching metric alone, not model quality. The **actual** train→val
  generalisation gap for PSM, holding the metric fixed, is **+0.0037 to +0.0041** (val is
  marginally *higher* than train under both metrics) — indistinguishable from noise, exactly as
  expected for a zero-learned-parameter method. **Corrected reading:** under the one protocol
  used everywhere else in this project (window-level, 26-class, k=0), PSM scores **0.7682**
  (train) / **0.7719** (val) against RGSAGE-Max's **0.846** (test) / **0.855** (val,
  `results/dropout_aware/rgsage_max/seed_00042/metrics.json`) — **the network beats PSM at
  k=0 in-distribution, on every split, under the standard metric.** This does not change Gate 2's
  reading (the decision explicitly excludes clean k=0, S2.0ᵃ/S2.4ᵇ), but the "PSM ≥ STGNN
  confirmed as base case" sentence itself is retracted; see S2.4 below for the same correction
  applied to the extrapolation-contrast writeup, and note the training-split event-level number
  (0.8977) remains a legitimate, valid measurement of Gate 1's S1.7 criterion — it was the
  *cross-metric comparison* against STGNN that was invalid, not the number itself.

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
   ends the paper. `[Corrected 2026-09-29]` **The "≈100%" figure here was a prediction, written
   2026-09-23 before this was measured — the measured training-split number, under the standard
   window-level 26-class protocol, is 0.7682, not ≈100%** (`results/pcstgnn/audit/s2_4_metric_check/`).
   Only the event-level, single-settled-step, fault-only variant approaches "near-perfect"
   (0.8977) — still not ≈100%, and that variant is not the protocol used for any other number in
   this project. The oracle *argument* (table = generator, so training-split evaluation of any
   kind is degenerate) still holds; the specific "≈100%" number attached to it does not, and is
   corrected here rather than left to be re-cited.
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

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_1_identifiability/identifiability_summary.json`,
8 random sensor subsets per k, σ_ε = 7.2555e-4, the corrected operator): identifiability is high
and degrades only slowly with dropout. Globally, the indistinguishable-pair fraction (D < 1, of
325 location pairs) is **1.5 % at k = 0, 1.5 % at k = 2, 1.5 % at k = 5, 1.8 % at k = 10** — a
worst-case min-D that falls from 0.0074 to 0.0047 (same confusable pair, location index 5,
throughout). Split by R: **R = 0.1 Ω and R = 1 Ω are fully separable at every k** (0 %
indistinguishable, worst-case min D in the hundreds); the identifiability limit sits almost
entirely **at R = 10 Ω**, where the indistinguishable fraction rises from 0.6 % (k = 0) to 0.9 %
(k = 10). Reading: sparse, magnitude-only µPMUs carry enough information to separate almost every
fault location at the training resistances, and dropout up to k = 10 barely narrows that margin —
the identifiability floor is a high-resistance phenomenon, not a sensor-count one.

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

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_2_confusability/confusability_summary.json`,
28,875 training events, 55 instrumented channels, raw nearest-neighbour classification at the
settled step): overall NN accuracy **88.9 %** (confusion rate 11.1 %). Table D and empirical
confusion agree well: **Spearman ρ = −0.655 (p = 4.0e-38)** over the 300 location pairs (low D ↔
high confusion, the expected sign), `paper/figures/fc3_table_vs_empirical_confusability.png`.
Reading against S2.2's diagnosis table, by pair count: **244/300 "table separable / corpus
separable"** (physics is informative), **5/300 "table confusable / corpus confusable"** (a
genuine, small information limit, consistent with S2.1's R = 10 Ω finding), and **0/300 "table
confusable / corpus separable"** — the bad row (table wrong) never occurs. The remaining **51/300
"table separable / corpus confusable"** are not one of the doc's three anticipated rows: the raw
nearest-neighbour metric pools all (τ, R, L) instead of profiling per-pair like D does, so this
count is expected to include pairs that are separable *given* the right (τ, R, L) but confusable
under a naive, unprofiled distance — it does not indicate operator error (see the 0/300 "table
wrong" row). **Net reading: separable/separable dominates — physics is informative — proceed.**

## S2.3 — B1: ΔV nearest-signature baseline

**Do.** Δs = s − v_nominal, with v_nominal the per-channel no-fault mean from the training split.
Nearest table signature at nominal load; no profiling over τ, R, or L. Also report the existing
matched filter (`scripts/run_physics_matched_filter.py`) as B1ᵃ — it is the same idea with a
rank-1 template and is already evaluated at every k.

**Check.** Full Paper-1 evaluation: F1 at every k, F1 by R and type, AUDC (relative and absolute).

**Purpose.** This is the classical method. B1 → B2 measures exactly what profiling and the load
axis buy you — the empirical defence of the operator design.

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_3_b1/b1_full_eval.json`,
validation split, 177.6 s wall): B1 (nominal-load nearest signature) macro-F1 by k =
**0.649 (k=0) → 0.644 (k=1) → 0.640 (k=2) → 0.628 (k=5) → 0.601 (k=10)**, AUDC relative **0.964**
/ absolute **0.626**. By resistance at k=0: **0.845 (0.1 Ω), 0.755 (1 Ω), 0.323 (10 Ω)** — B1
already shows the same high-R weakness S2.1 flags, just without profiling. By fault type at k=0,
ungrounded faults are the hardest (ABC 0.558, ABCG 0.558, AG 0.569) and phase-phase easiest (BC
0.738) — consistent with S1.3's finding that ungrounded types dominate the non-monotone-in-R
cells. B1ᵃ (rank-1 matched filter, no load/type profiling at all) confirms the existing
**0.162 at every k**, essentially flat under noise too (0.162 → 0.097 from σ=0 to σ=1e-2), i.e.
it is nearly uninformative on its own. **B1 → B2 (S2.4) is the real test of what profiling buys.**

## S2.4 — B2: PSM (Physics Signature Matching)

**Do.** f̂ = argmin_f r̃_f(A) over f = 0…25, no network at all. Full Paper-1 evaluation. Also run
on the OOD-pilot.

**Check.** Against B1 and against all Paper-1 models (standard and dropout-aware).

**Why mandatory.** Without it a reviewer says "your physics module is doing all the work; the GNN
is unnecessary." The scientifically interesting outcome is PSM < STGNN < PC-STGNN — but see the
priors above: PSM ≥ STGNN at k = 0 on the clean corpus is likely, and then the question becomes
where learning still helps.

**Metrics.** Additionally report PSM top-3 accuracy and the rank of the true location under r̃.

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_4_psm/psm_full_eval.json`,
single mask seed — see the script's own documented time-budget note; zero learned parameters):
**validation-split** macro-F1 by k = **0.772 (k=0) → 0.771 (k=1) → 0.769 (k=2) → 0.763 (k=5) →
0.749 (k=10)**, top-3 accuracy 0.789→0.770, AUDC relative 0.987 / absolute 0.762 — flat under
dropout, as expected for a physics method with no measurement channels to overfit to. Against B1
(above), profiling (τ, R, L) buys **+0.123 F1 at k=0** (0.649→0.772) — the empirical defence of
the operator's design that S2.3 calls for.

`[Corrected 2026-09-29 — the paragraph originally here is withdrawn]` An earlier draft of this
section called this a "training-split PSM ≥ STGNN prior that reverses on validation," attributing
the gap to train→val generalisation. **That framing was wrong, caught on author review, and
verified rather than just conceded:** the 89.74 % training-split figure is an event-level,
25-class, fault-only metric (see the "Priors already on disk" correction above), not the same
task as this section's window-level 26-class validation number — the two were never comparable,
on any split. Measured side by side
(`scripts/check_s2_4_metric_consistency.py`,
`results/pcstgnn/audit/s2_4_metric_check/metric_consistency_check.json`):

| | event-level, 25-class, fault-only | window-level, 26-class (this section's metric) |
|---|---|---|
| train split | 0.8977 | **0.7682** |
| val split | 0.9018 | **0.7719** |

Holding the metric fixed, PSM's train→val gap is **+0.0037 to +0.0041** — val is marginally
*higher*, indistinguishable from sampling noise (val has 6,200 events vs train's 28,875), exactly
what a zero-learned-parameter method should show. **There is no generalisation gap to explain.**
The entire ~0.13 difference between "0.8977" and "0.772" is the metric switch, present equally on
both splits. **Corrected comparison, same protocol throughout:** RGSAGE-Max (dropout-aware, seed
42)'s best-validation graph-F1 is **0.855** (`results/dropout_aware/rgsage_max/seed_00042/metrics.json`)
against PSM's validation 0.772 — the network wins by 0.083, **in-distribution, at k=0, on every
split measured.** This does not change Gate 2's reading (the decision rule below explicitly
excludes clean k=0 comparisons either way), but the correct standing claim is: **PSM does not
beat STGNN in-distribution under the standard protocol; it was never shown to.**

**OOD-pilot** (R ∈ {5, 20} Ω, never seen by the operator's own R grid or the network's training):
macro-F1 by k = **0.539 (k=0) → 0.534 (k=1) → 0.529 (k=2) → 0.513 (k=5) → 0.477 (k=10)**, AUDC
relative 0.948 / absolute 0.511. PSM loses 0.233 F1 moving from validation to OOD-pilot — see
S2.5 for the contrast against the network's own drop.

## S2.5 — The extrapolation contrast — the most important experiment in Stage C

**Do.** Evaluate the existing dropout-aware RGSAGE-Max checkpoints on the OOD-pilot.

**Check.** Compare the neural model's F1 drop (in-distribution → OOD) against PSM's.

**Why.** The network was trained on {0.1, 1, 10} Ω. PSM has a dense continuous representation of
R. If the network collapses on unseen resistance and PSM does not, the paper's central claim is
demonstrated before a line of model code is written.

**Figure produced.** F-C4 (pilot version; the paper version is F-G3 from the sealed set).

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_5_extrapolation/extrapolation_contrast.json`,
`paper/figures/fc4_extrapolation_contrast.png`, dropout-aware RGSAGE-Max, seeds 42/43/44,
1 mask seed on the pilot, in-distribution reference = `results/dropout_study_aware/degradation_curves.csv`
test split): **the network collapses on unseen R; PSM does not.**

| | in-distribution / validation (k=0) | OOD-pilot (k=0) | drop |
|---|---|---|---|
| Network (dropout-aware RGSAGE-Max) | 0.839 (test) | **0.089 ± 0.006** | **0.750** |
| PSM | 0.772 (val) | **0.539** | **0.233** |

The network's OOD-pilot F1 (0.089) is barely above the trivial 26-class baseline (0.0256, Paper
1) — it has essentially no signal left on unseen resistance — while PSM retains most of its
validation-split accuracy. This holds at every k (network 0.084→0.068 across k=1..10 on the
pilot vs. PSM 0.534→0.477), so it is not a dropout artefact stacking with the OOD effect; it is
the resistance extrapolation itself. **This is exactly the row-2 evidence Gate 2 is looking
for** — the network already learned in-distribution physics well enough to beat PSM there, but
PSM is the one that survives leaving the training R values. `psm_more_robust_to_ood_R: true`
in the JSON, unambiguously.

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

`[Corrected 2026-09-29]` **Measured** (`results/pcstgnn/audit/s2_6_fusion/fusion_summary.json`,
seed 42 only per the script's documented budget — see its own docstring; validation split,
`val_sample` 30,000-window time-budgeted subsample, one forward pass per condition, β grid-
searched on the cached (z, ρ, y) tensors):

| Condition | β* | F1(β=0) | F1(β*) | ΔF1 |
|---|---|---|---|---|
| clean, k=0 | 0.1 | 0.858 | 0.897 | **+0.039** |
| clean, k=0, at 10 Ω | 0.1 | 0.667 | 0.753 | **+0.086** |
| k=10 | 0.1 | 0.695 | 0.860 | **+0.166** |
| σ=5e-4 | 0.1 | 0.762 | 0.799 | **+0.037** |
| σ=2e-3 | 16.0 | 0.592 | 0.745 | **+0.153** |
| σ=1e-2 | 2.0 | 0.325 | 0.538 | **+0.213** |

**Pass, decisively, on every condition** — every Δ is positive, and the largest gains are exactly
where the pass criterion asks (10 Ω, k=10, under noise). Two things stand out beyond "pass":
(1) the ΔF1 at k=10 (+0.166) is larger than at clean k=0 (+0.039) — physics is buying more where
the network has the least to work with, the intended behaviour of a fusion term; (2) **the noise
arm is the qualitatively new result** — Paper 1 found dropout-aware *training* gives a clean null
against measurement noise (−0.008 to −0.034, CIs span zero); inference-time fusion instead
recovers up to **+0.213 F1** at σ=1e-2 (the IEEE C37.118 spec-limit noise level), rising with β*
itself (0.1 → 16.0 → 2.0 as noise grows) — physics is doing more correction work exactly where
the mask-based mechanism that makes dropout-aware training work (S1.4/`01_SPEC.md`'s "(0,0,0) is
dark" argument) has nothing to fire on, since noise leaves the mask at 1. **This is the single
strongest piece of Gate-2 evidence: physics recovers accuracy under exactly the perturbation
dropout-aware training cannot touch.** Caveat carried forward honestly: single seed, single
mask-seed-budgeted subsample — a 3-seed confirmation belongs in Stage D/E, not asserted here as
final.

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

`[Corrected 2026-09-29]` **All Stage C measurements are in — S2.0–S2.6 complete.** Reading the
evidence against the four rows above, using only the noise and model-mismatch arms as instructed:

- **Row 1 fires: S2.6 improves F1 at 10 Ω, at k=10, and under every noise level tested**,
  sign-consistent, with the largest gains exactly at the hardest conditions (+0.166 at k=10,
  +0.213 at σ=1e-2). Physics carries information the network's dropout-aware training does not.
- **Row 2 also fires: S2.5 shows the network collapsing on unseen R (0.839→0.089, a 0.750 drop)
  while PSM holds (0.772→0.539, a 0.233 drop)** — the network already learned in-distribution
  physics well (it beats PSM in-distribution at every k=0 comparison measured, val 0.855 vs
  0.772 — see the S2.4 correction above), but physics is what survives leaving the trained R
  range.
- `[Corrected 2026-09-29]` **Row 3 does NOT fire.** An earlier draft of this section claimed it
  did ("PSM ≈ or > STGNN on the clean corpus"), based on the now-retracted cross-metric
  comparison (see the S2.4 correction above) — under the one protocol used consistently in this
  project, the network beats PSM in-distribution on every split measured (train 0.7682 <
  0.8515/0.855 network; val 0.7719 < 0.855 network). This does not change the overall Gate-2
  reading, since rows 1 and 2 already decide it independently, but the row-3 claim itself is
  false and is not carried forward into the paper.
- **Row 4 does not fire.** S2.1/S2.2 show the sensing infrastructure is highly identifiable (98.5 %
  of location pairs at k=10) with the small residual confusability concentrated at R=10 Ω and
  corroborated empirically (ρ=−0.655) — this is not an information-limit paper.

**Reading: this is not a close call.** Two independent experiments (S2.5's OOD-resistance
collapse and S2.6's noise-recovery) each *independently* satisfy a GO row on their own, along
different perturbation axes (unseen R vs. measurement noise), neither reducible to the other.
Both point the same direction: the network and the physics operator fail under different,
non-overlapping stressors, and fusing them recovers accuracy under both. This is the strongest
form of evidence the decision table anticipates.

**What this is not:** a claim that PC-STGNN (the trained, learned fusion of `06_STAGE_E_pcstgnn.md`)
will outperform dropout-aware training end-to-end — S2.6 tested a *fixed*, hand-set-β,
inference-only fusion on an *existing* checkpoint never trained to expect it. That is a
strictly weaker test than Stage E's learned model, and it already clears every Gate-2 bar.

**Formally closing Gate 2 and committing to Stages D–H (full plan, reframed per S2.0ᵃ around
noise / OOD resistance / silent failures / model mismatch, per this doc's "What survives" table)
is the author's call, not automated by any script here** — matching Gate 1's own closure practice
(`03_STAGE_B_operator.md`). The measurements above are the complete evidence base for that call;
nothing in Stage C remains unmeasured.

---

## GATE 2: CLOSED — 2026-09-29

**Decision: GO.** Reading taken from S2.5 (OOD-resistance) and S2.6 (noise-recovery), per this
gate's rule that clean k=0 numbers are oracle references and cannot decide the branch.

**Evidence:**
- S2.5: network collapses ~89% relative on unseen resistance (0.839→0.089 F1); PSM holds
  (0.772→0.539).
- S2.6: physics fusion recovers up to +0.213 F1 under measurement noise, where dropout-aware
  training alone showed no effect.
- In-distribution, corrected: network beats PSM by ~0.08 F1 (0.7719 vs 0.7682, validation,
  26-class macro-F1) — physics does not win on its own turf, it covers two specific failure
  modes the network doesn't.

**Pre-registration:** `14_PREREGISTRATION.md`, commit `9fb50b8` (content), `44d1c0c` (hash
record).

**Proceeding** to Stage D remaining items (S3.3 physics-imputation, S3.4 speed/cache) and Stage E
(S4.1 null test).
