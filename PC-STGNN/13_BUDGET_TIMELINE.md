# Budget, Timeline, Fallbacks

All GPU figures use the **measured** per-run cost on the T1000
(`results/runtime/runtime_training.csv`): RGSAGE-Max 30 epochs = 6.7 h standard / 6.9 h
dropout-aware; a 15-epoch tuning run ≈ 3.5 h; RGATv2 12.3 h (not needed here). The first draft
assumed 3–4 h per run and was ~2× low.

## GPU budget

| Block | Runs | GPU-h | Notes |
|---|---|---|---|
| Dropout-aware baselines (4 × 3 seeds) | 0 | 0 | **done** — `results/dropout_aware/` |
| S4.1 null test | 1 | 7 | may be stopped early once curves match |
| λ tuning (15-epoch) | 6 | ≈ 21 | |
| Core ablation R1–R7 (seed plan of `06`) | 16 | ≈ 115 | R7 wider: +10–20 % |
| B2ᵃ learned-R̂ head | 1 | 7 | |
| Gate study G0 ×3, G1 ×1, G2 ×3, G3 ×1 | 8 | ≈ 60–70 | G2/G3 two-pass: ≈ 1.3× |
| Model-mismatch retrain at ±5 % | 1 | 7 | |
| **Total** | **≈ 33** | **≈ 220–235 h ≈ 9–10 days** continuous | |
| Optional: promote R1/R3/R4/R7/G1/G3 to 3 seeds | +12 | +85 | only after Gate 4 |

## CPU-only work

| Item | Cost |
|---|---|
| Signature table (70 k snapshot solves) | ≈ 25–30 min at ~45 solves/s — measure in S0.5 |
| L-grid / R-grid studies | < 1 h |
| Perturbed tables (3–5) | ≈ 2–2.5 h |
| OOD-pilot (1,100 events × 60 Daily-mode steps) | ≈ 25 min |
| OOD-test (11,000 events × 60 steps) | ≈ 4–5 h |
| All evaluation | inference — minutes per model per condition; the Paper-1 conditions are reused for the existing models |

## Schedule

| Weeks | Stage | GPU in background |
|---|---|---|
| 1 | A — audit | — |
| 2–4 | B — operator + fidelity → Gate 1 | — |
| 4–6 | C — identifiability, PSM, fusion → Gate 2 | — (nothing to pre-train: the baselines exist) |
| 5–9 | D — silent injector, OOD sets, physics-impute baseline, residual layer | — |
| 8–13 | E — null test, tuning, R1–R7 → Gate 4 | ablation runs |
| 12–16 | F — detection AUROC, G0–G3 → Gate 5 | gate runs |
| 15–19 | G — full sweep, OOD, model mismatch | mismatch retrain |
| 19–26 | H — analysis, figures, writing, release | — |

About 6.5 months. The decisive commitment point is week 6, after roughly five weeks of CPU-only
work. If the physics does not carry information, you learn it before spending a single
significant GPU hour, and you still have four publishable fallbacks.

## Optional axis: scenario-held-out split

`dataset_fullscale_50_scenario/` (575/125/125 scenarios, 0 overlap, all 26 classes per split)
already exists. The frozen split shares all 825 scenarios across train/test, so in-distribution
accuracy is generalisation to unseen *load* of a known configuration. The OOD-R set covers the
resistance axis; this split would cover unseen (f, τ, R) combinations. Not in the base plan
(≈ 7 h per retrain); worth one R0-vs-R5 pair if a reviewer asks.

## Success criteria — the pattern, not a threshold

A fixed "≥ 0.03–0.05 macro-F1" bar is the wrong instrument. With this test-set size, significance
is cheap; what makes the result physical is *where* the gain appears. Judge against all six:

1. The gain concentrates in the pre-registered conditions: 10 Ω, k = 10, OOD R, injected noise,
   silent failures.
2. Clean k = 0 accuracy does not drop by more than one seed standard deviation.
3. Sign-consistent across all three seeds.
4. Paired bootstrap p < 0.05 on the pre-registered comparisons only.
5. R6 (shuffled physics) removes the gain and R2 (learned decoder) does not reproduce it.
6. It survives ±5 % model mismatch.

+0.01 in-distribution with +0.10 on unseen resistance is a stronger paper than +0.05 everywhere —
only the first pattern is evidence that the mechanism is physical rather than regularisation.

## Fallback map — every branch still ends in a paper

| Where it lands | The paper you write |
|---|---|
| Gains on OOD R and noise, flat in-distribution | "Physics buys generalisation, not in-distribution accuracy." Honest, and the most likely outcome |
| PSM ≥ STGNN on the clean corpus | Physics dominates in-distribution on a noise-free simulator corpus with a scalar load; show where learning still helps (noise, dropout, silent failures, model mismatch). Likely on this corpus — see Stage C priors |
| Gains only under silent failures | Refocus on silent-failure robustness; PC is the enabling component, not the headline |
| Gate AUROC < 0.75 | PC-STGNN paper + a bounded negative result on magnitude-only silent-failure detection |
| No model gains, clear identifiability limit | Identifiability + benchmark paper: how much 25 sparse magnitude-only µPMUs can reveal. IEEE Access or SEGAN |
| Gate 1 fidelity fails | Fork to the voltage-angle route at week 4 (seeds stored; angles already computed in `monitors.py:78–80`) |
