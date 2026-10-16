# Budget, Timeline, Fallbacks

All GPU figures use the **measured** per-run cost on the T1000
(`results/runtime/runtime_training.csv`): RGSAGE-Max 30 epochs = 6.7 h standard / 6.9 h
dropout-aware; a 15-epoch tuning run ≈ 3.5 h; RGATv2 12.3 h (not needed here). The first draft
assumed 3–4 h per run and was ~2× low.

`[Corrected 2026-09-30]` These physics-free rates still apply to every PC-STGNN arm that does
**not** load the signature table — R0ⁿ, R1, R2, R7, and S4.1's null test (`use_fusion=False`,
table never loaded). Any arm with `use_fusion=True` or λ₁/λ₂ > 0 additionally pays the
**measured live ρ(A) cost of 44.7 ms/batch-of-32** (`results/pcstgnn/audit/s3_4_residual_speed/
speed_benchmark.json`, Gate 3, confirmed compute-bound and chunk_f-invariant), applied to the
train split's 36,093 batches/epoch: **+6.7 h on a 15-epoch run, +13.4 h on a 30-epoch run**, on
top of the backbone rate above. That gives two rates, not one:

- Physics-free: 3.5 h (15 epochs) / 6.9 h (30 epochs) — R0ⁿ, R1, R2, R7, S4.1
- **Physics-loaded: ≈10.2 h (15 epochs) / ≈20.3 h (30 epochs)** — every other arm below

## GPU budget

| Block | Runs | GPU-h | Notes |
|---|---|---|---|
| Dropout-aware baselines (4 × 3 seeds) | 0 | 0 | **done** — `results/dropout_aware/` |
| S4.1 null test | 1 | 7 | may be stopped early once curves match. `use_fusion=False`, table never loaded — unaffected by the live-ρ correction below. **PASSED, done** |
| λ tuning (15-epoch) | 5 | ≈ 51 | `[Corrected 2026-09-30]` was "6 runs, ≈21 GPU-h". Run count corrected 6→5: the sweep is 3 λ₁ arms + 2 λ₂ arms (`06_STAGE_E_pcstgnn.md` S4.3's own "Do" bullet); there is no 6th arm — the λ₂ sweep's λ₂=0 baseline **is** the winning λ₁ run, not a separate retrain. Rate corrected from the physics-free 3.5 h/run to the measured physics rate **≈10.2 h/run** (every arm here has `use_fusion=True`, so every arm loads the table): 5 × 10.2 ≈ 51 |
| **R0ⁿ — dropout-aware + noise aug, no physics** | **3** | **≈ 21** | `[Added 2026-09-23]` the paired reference for every ablation arm, forced by the noise-augmentation default (`06` S4.3ᵇ). Not optional: without it, R5 vs R0 confounds noise augmentation with physics. No physics table loaded — unaffected by the live-ρ correction |
| Core ablation R1–R8 (seed plan of `06`) | 16 | ≈ 258 | `[Corrected 2026-09-23]` seed swap R4 ↔ R7 (R4 3, R7 1), plus **R8** (R5 trained noise-free) at **1 seed** — it is an oracle condition, i.e. a labelled upper bound, not a contested comparison. R7 wider: +10–20 %. `[Corrected 2026-09-30]` run count corrected 17→16 to match `06`'s current per-arm table (R1 1 + R2 3 + R3 1 + R4 3 + R5 3 + R6 3 + R7 1 + R8 1 = 16). Split by table use: **R1, R2, R7 (5 runs, no table)** at 6.9 h/run = 34.5; **R3, R4, R5, R6, R8 (11 runs, table loaded)** at ≈20.3 h/run = 223.8; row total 34.5 + 223.8 ≈ 258 |
| B2ᵃ learned-R̂ head | 1 | ≈ 20 | `[Corrected 2026-09-30]` was 7 (physics-free rate assumed). B2ᵃ loads the table — physics rate applies: ≈20.3 |
| **Stage E subtotal (S4.1 → B2ᵃ)** | **26** | **≈ 357** | `[Added 2026-09-30]` new row, for visibility only (sum of the five rows above: 7 + 51 + 21 + 258 + 20 ≈ 357). Not present before this correction because the rows it sums were themselves wrong |
| Gate study G0 ×3, G1 ×1, G2 ×3, G3 ×1 | 8 | ≈ 60–70 | G2/G3 two-pass: ≈ 1.3×. `[Note 2026-09-30]` **Not recomputed by this correction.** Stage F's PCRG-STGNN (`07_STAGE_F_gate.md`) is not built yet, and its gate reuses the same ρ(A) fusion pathway plus a per-sensor LOO refit, so this row is very likely under-costed for the identical reason (rho_LOO alone measured fine at 2.8 ms/batch — Gate 3 — but that is on top of, not instead of, the ρ(A) pass). Revisit when Stage F exists; do not treat this row as validated |
| Model-mismatch retrains at **two** levels | 2 | 14 | `[Corrected 2026-09-23]` ±5 % plus one of ±2 %/±10 %, funded by R8's freed seeds. Mismatch is a primary axis now; one retrained point is an anecdote, two make a trend. Out of scope for the 2026-09-30 correction (Stage G, unaffected either way — model-mismatch retrains reuse a fixed table, not the live path) |
| **Total** | **≈ 36** | **≈ 431–441 h ≈ 18–18.4 days** continuous | `[Corrected 2026-09-30]` was ≈255–270 h ≈11 days. The +~176 h increase is almost entirely the Core-ablation row (+~139 h) and λ tuning (+~30 h); see the dated note below for why this was missed at Gate 3 close. **Gate study is still at its pre-correction, likely-optimistic rate** (see its row note) — the true total is probably higher than 441 h once Stage F is built and measured |
| Optional: promote R1/R3/R7/G1/G3 to 3 seeds | +10 | ≈ +96 | `[Corrected 2026-09-30]` was "+70" (flat 7 h/run assumed for all five). Recomputed per arm: R1 +2 runs, no table (+13.8) + R3 +2 runs, table (+40.7) + R7 +2 runs, no table (+13.8) + G1 +2 / G3 +2, Stage-F rate unchanged (+28) = ≈96.3. Only after Gate 4 |

`[Corrected 2026-09-30]` **Why this wasn't caught at Gate 3 close.** The 44.7 ms/batch live-ρ(A)
cost was measured, and the "the layer stays online for masked inputs" design decision was made,
on 2026-09-29 (`05_STAGE_D_infra.md` S3.4) — both the measurement and the decision were correct.
But no PC-STGNN training loop existed yet to run that decision through: `forward_pc`/`pc_loss`
had only been exercised by the S4.2 gradient audit, a CPU-only script checking three isolated
batches, never by `experiments/runner.py`'s actual training loop. The GPU-hour *consequence* of
"stays online" was therefore invisible at the moment the decision was recorded, and it was never
propagated into this file — every rate above was carried over unchanged from
`results/runtime/runtime_training.csv`, which is the physics-free RGSAGE-Max rate, because that
was the only rate anyone had measured. This is a process gap (a decision's cost not reaching the
budget doc that owns it), not a measurement error or a wrong decision, and it is being recorded
here plainly rather than folded into a parenthetical so the same gap doesn't recur when Stage F's
PCRG-STGNN is built (see the Gate-study row above, which is in exactly that position now). Found
2026-09-30 while building and smoke-testing S4.3's tuning script — the first time `forward_pc`
was ever run on a GPU-resident model with a loaded table.

## CPU-only work

| Item | Cost |
|---|---|
| Signature table (70 k solves at L-step 0.05; 136 k at 0.025) | `[Measured 2026-09-23, S0.5]` **2.9 min** and **5.6 min** *serial* — 402 solves/s in `reuse` mode, ≈ 9× the planned rate, because the estimate was paying the OpenDSS compile on every solve. Cost is no longer an input to the S1.2 grid choice (`03_STAGE_B` S1.2 selects on ρ-margin usability) |
| L-grid / R-grid studies | < 1 h |
| Perturbed tables (3–5) | ≈ 2–2.5 h |
| OOD-pilot (1,100 events × 60 Daily-mode steps) | `[Measured 2026-09-23]` ≈ **20 s** (17 ms/event) |
| OOD-test (11,000 events × 60 steps) | `[Measured 2026-09-23]` ≈ **3 min**, not 4–5 h |
| All evaluation | inference — minutes per model per condition; the Paper-1 conditions are reused for the existing models |

## Schedule

| Weeks | Stage | GPU in background |
|---|---|---|
| 1 | A — audit | — |
| 2–4 | B — operator + fidelity → Gate 1 | — |
| 4–6 | C — identifiability, PSM, fusion → Gate 2 | — (nothing to pre-train: the baselines exist) |
| 5–9 | D — silent injector, OOD sets, physics-impute baseline, residual layer. `[Corrected 2026-09-29]` **Gate 3 closed, all five items done** — see `05_STAGE_D_infra.md`. Physics-impute baseline evaluated on the full sweep (AUDC 0.601 rel / 0.508 abs vs neighbour-mean's 0.476 / 0.403); residual speed check measured 44.7 ms/batch (9x over budget, compute-bound) so `cache/rtilde_{test,val,train}.npy` was built instead (correctness verified to ~1e-5 vs the direct-method reference) | — |
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
| PSM ≥ STGNN on the clean corpus | Physics dominates in-distribution on a noise-free simulator corpus with a scalar load; show where learning still helps (noise, dropout, silent failures, model mismatch). `[Corrected 2026-09-29]` **Retracted as "likely"** — measured (`04_STAGE_C_gonogo.md` S2.4, `results/pcstgnn/audit/s2_4_metric_check/`), the network beats PSM in-distribution at k=0 on every split (val 0.855 vs 0.772). This row is not the observed branch; Gate 2 fired on rows 1 and 2 instead (fusion gain under noise/dropout, and PSM's OOD-R resistance) |
| Gains only under silent failures | Refocus on silent-failure robustness; PC is the enabling component, not the headline |
| Gate AUROC < 0.75 | PC-STGNN paper + a bounded negative result on magnitude-only silent-failure detection |
| No model gains, clear identifiability limit | Identifiability + benchmark paper: how much 25 sparse magnitude-only µPMUs can reveal. IEEE Access or SEGAN |
| Gate 1 fidelity fails | Fork to the voltage-angle route at week 4 (seeds stored; angles already computed in `monitors.py:78–80`) |
