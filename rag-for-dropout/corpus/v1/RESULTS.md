# RESULTS.md — current headline numbers

**Updated 2026-08-16; §8 added 2026-10-16.** Every number is post the mask-diversity bugfix (`798406e`) and is copied from the CSV/JSON named alongside it. Precedence: CSV wins over `paper/overleaf.tex`; the `.tex` wins over this file.

Corpus: `dataset_fullscale_50` — 825 scenarios × 50 runs × 40 windows = **1,650,000 windows**, event-level 70/15/15 split (0 leakage), 6,175 test events. Frozen config: GNN depth 4, 30 epochs, unweighted CE, grad-clip 1.0, 3 warm-up epochs. Seeds 42/43/44 throughout.

## 1. Clean-condition benchmark (k = 0)
Soft-voted graph macro-F1, mean ± std over 3 training seeds. Source: `fullscale_final/benchmark_summary.csv`.

| Model | macro-F1 | per-window F1 | 0.1 Ω | 1 Ω | 10 Ω |
|---|---|---|---|---|---|
| GRU (no graph) | 0.286 ± 0.009 | 0.150 | 0.410 | 0.292 | 0.105 |
| RGATv2 | 0.522 ± 0.087 | 0.460 | 0.630 | 0.574 | 0.312 |
| RGCN | 0.712 ± 0.017 | 0.601 | 0.874 | 0.792 | 0.428 |
| RGSAGE-Mean | 0.786 ± 0.023 | 0.660 | 0.906 | 0.843 | 0.587 |
| **RGSAGE-Max** | **0.846 ± 0.007** | **0.718** | **0.949** | **0.893** | **0.676** |

No-fault precision ≥ 0.97, recall ≈ 1.00 for every model. Best-validation epoch was 27.7–28.7 of 30 for RGCN/RGSAGE-*, so those accuracies are a **floor**, not a converged ceiling.

## 2. Random sensor dropout — standard training
Macro-F1 vs k μPMUs dropped at inference (of 25); 3 training seeds × 3 mask seeds. Source: `dropout_study/degradation_curves.csv`.

| Model | k=0 | k=1 | k=2 | k=5 | k=10 | AUDC | Ret.@10 |
|---|---|---|---|---|---|---|---|
| **RGSAGE-Max** | 0.846 | 0.736 | 0.590 | 0.236 | **0.048** | **0.402** | **5.7 %** |
| RGSAGE-Mean | 0.786 | 0.720 | 0.647 | 0.393 | 0.108 | 0.540 | 13.8 % |
| RGCN | 0.712 | 0.664 | 0.610 | 0.418 | 0.164 | 0.607 | 23.1 % |
| RGATv2 | 0.522 | 0.490 | 0.444 | 0.302 | 0.157 | 0.621 | 30.1 % |
| GRU | 0.286 | 0.273 | 0.252 | 0.111 | 0.051 | 0.521 | 17.9 % |

**Accuracy/robustness inversion:** the most accurate model is the most fragile. Same sign in all 3 seeds, but **not** significant at n=3 (Δ=+0.116 for RGCN over RGSAGE-Max at k=10, p=0.076) — an observed trend, not a proven law.

## 3. Random sensor dropout — dropout-aware training
Source: `dropout_study_aware/degradation_curves.csv`.

| Model | k=0 | k=1 | k=2 | k=5 | k=10 | AUDC (std → aware) |
|---|---|---|---|---|---|---|
| **RGSAGE-Max** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | 0.402 → **0.913** |
| RGSAGE-Mean | 0.673 | 0.666 | 0.657 | 0.628 | 0.550 | 0.540 → 0.921 |
| RGCN | 0.613 | 0.607 | 0.600 | 0.579 | 0.533 | 0.607 → 0.940 |
| RGATv2 | 0.553 | 0.550 | 0.546 | 0.533 | 0.505 | 0.621 → 0.960 |
| GRU | 0.300 | 0.300 | 0.299 | 0.278 | 0.078 | 0.521 → 0.784 |

The AUDC gain is significant for **all five** architectures (p_Holm ≤ 0.027). RGCN and RGSAGE-Mean pay ~0.1 macro-F1 in *clean* accuracy for it; RGSAGE-Max does not (−0.008, p=0.181).

## 4. Mitigation comparison (RGSAGE-Max backbone)
Source: `../paper/figures/sdr_robustness_summary.csv`, `dropout_study_impute/imputation_summary.json`.

| Variant | AUDC | k=0 | k=1 | k=2 | k=5 | k=10 | Ret.@10 |
|---|---|---|---|---|---|---|---|
| Standard (zeroing) | 0.402 | 0.846 | 0.736 | 0.590 | 0.236 | 0.048 | 5.7 % |
| Neighbour-mean imputation | 0.476 | 0.846 | 0.739 | 0.637 | 0.336 | 0.099 | 11.8 % |
| **Dropout-aware training** | **0.913** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | **80.3 %** |
| SDR-STGNN (gated) | 0.892 | 0.839 | 0.827 | 0.813 | 0.758 | 0.632 | 75.2 % |

**SDR-STGNN matches but does not beat dropout-aware training** (ΔAUDC = 0.021 in favour of dropout-aware, p=0.098 — a statistical tie) and costs **+18 % inference latency**. Report this faithfully; do not claim the gate wins.

## 5. Targeted (worst-case) dropout
Drops the k μPMUs nearest the fault bus. Deterministic given the label → no mask seeds; error bars over training seeds only. RGSAGE-Max only. Source: `dropout_study_targeted/targeted_curves_*.csv`.

| Variant | k=0 | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| Standard, **targeted** | 0.846 | **0.448 ± 0.039** | 0.361 | 0.206 | 0.048 |
| Standard, random (ref) | 0.846 | 0.736 ± 0.013 | 0.590 | 0.236 | 0.048 |
| Dropout-aware, **targeted** | 0.839 | 0.755 | 0.695 | 0.600 | **0.521** |

Targeting is far more damaging where the model is still usable: at k=1 the standard model drops **3.6× further** than under random loss (p=0.010). Dropout-aware training is penalized too (AUDC 0.913 → **0.747**, p=0.003), so worst-case loss is a **distinct stressor, not one the mitigation absorbs** — but from k=2 upward its *worst-case* score still exceeds the standard model's *average-case* score (0.695 vs 0.590 at k=2, p=0.042). At k=1 the two are indistinguishable (+0.018, p=0.33). *Caveat:* no-fault windows have no fault bus and fall back to random selection, so these numbers blend worst- and average-case windows — a **conservative bound** on the true worst case.

## 6. Persistent (per-event) dropout — a null result
Draws the dropped set once per event and holds it across the event's 40 windows. Source: `dropout_study_persistent/persistent_curves_*.csv`.

**Nothing moves.** No point on either curve shifts by more than **0.003** macro-F1 (standard AUDC 0.402 → 0.401; dropout-aware 0.913 → 0.913; Ret@10 5.7 % → 5.6 % and 80.3 % → 80.1 %); no difference survives Holm across the 12 paired comparisons (min p_Holm = 0.46). **Why:** the decision rule is per window — soft voting pools the 25 nodes of a *single* window, never the 40 windows of an event, and a window's mask has the same marginal distribution under both schemes. **This equivalence would not hold for an event-level rule**, which per-window resampling would flatter.

## 7. Significance and cost
`statistics/significance_tests.json` — paired two-sided t-tests over seeds 42/43/44 (n=3, df=2), 95 % CIs, Holm-Bonferroni per family, plus a paired cluster bootstrap over the 6,175 test events.

| Claim | Δ | p_Holm | Verdict |
|---|---|---|---|
| Dropout-aware > standard @ k=10 | +0.626 | 0.003 | significant |
| Retention @10 | +74.6 pp | 0.003 | significant |
| AUDC | +0.511 | 0.003 | significant |
| Dropout-aware > imputation @ k=10 | +0.574 | 0.008 | significant |
| Clean-F1 cost of dropout-aware | −0.008 | 0.196 | null, as claimed |
| Dropout-aware over SDR-STGNN (AUDC) | +0.021 | 0.196 | statistical tie |
| Accuracy/robustness inversion | +0.116 | p=0.076 | **trend only** |
| Imputation over zeroing @ k=10 | +0.051 | p=0.102 | **not significant** |

`runtime/runtime_summary.json` — dropout-aware has **identical** params (95,258) and latency (0.639 vs 0.640 ms @ B=1; 6,038 windows/s both); its only cost is **+2.7 %** training sec/epoch. SDR-STGNN's gate adds 0 params but **+18 %** latency for no robustness gain. Activation memory is 145 MiB for *all* models (the shared GRU-128 encoder dominates). Training: 4.7–12.3 GPU-hours per run on a T1000 8 GB.

## 8. Reviewer-hardening findings (2026-08-17 pass; moved from CLAUDE.md 2026-10-16)
Moved verbatim in substance from the root `CLAUDE.md` section "2026-08-17 reviewer-hardening pass"; all traceable to `results/`. Modules: `dropout/noise.py` (Gaussian noise in per-unit voltage, applied as `sigma_pu / std_c`, gated by the availability mask — verified 0.00502 p.u. realized at σ=0.005, exactly 0 elsewhere, mask channels untouched); `dropout/correlated.py` (dropped set grown as a **connected** subgraph of the 25-node tree from a uniform root).

- **The frozen split shares 825 of 825 scenarios between train and test.** It partitions the 50 runs of each scenario, not the scenarios, so accuracy measures generalization to unseen *load conditions* of a known configuration. `dataset_fullscale_50_scenario` (575/125/125 scenarios, 0 overlap, all 26 classes per split; `processed/` is a symlink to the frozen corpus) is the held-out counterpart (`scripts/build_scenario_heldout_split.py`).
- **A classical baseline nearly matches the GNNs.** Logreg **0.713** / RF **0.633** at k=0 on **8% of the training data**, vs RGSAGE-Max 0.846 and RGCN 0.712. The GNN margin over a linear model is 0.133, not the 0.560 the GRU (0.286) implies. `classical_baseline/` (`scripts/run_classical_baseline.py`: logreg + random forest on 375 summary features, identical splits and mask seeds).
- **RF is the most dropout-robust untreated model in the study**: Ret@10 **35.9%**, AUDC_abs 0.470, above RGCN's 23.2%/0.432. "No baseline retains more than a third" is now scoped to the five networks. Dropout-aware training still dominates everything (80.3%, 0.766).
- **Contiguous outages are MILDER than scattered ones**, not harsher (standard RGSAGE-Max: 0.452 vs 0.236 at k=5). **But no contiguous-vs-scattered difference survives Holm at n=3** (min p_Holm 0.054) — report as an observation. **Confound:** per-node drop marginals are far from uniform (0.18–0.87 vs 0.40 at k=10), so the policy does not isolate spatial correlation; an incidence-matched control was NOT run. At k=1 the policies provably coincide and both the marginal gap (2%) and the F1 gap (+0.001) vanish — the harness check. For the **dropout-aware** model the policy is irrelevant (|Δ| ≤ 0.012). `dropout_study_correlated/` (`scripts/run_correlated_dropout_study.py`, 3 seeds, both arms).
- **Measurement noise costs real accuracy, and dropout-aware training does NOT help against it.** Standard RGSAGE-Max at k=0: 0.846 → 0.786 (σ=5e-4 p.u.) → 0.579 (σ=2e-3) → 0.345 (σ=1e-2, the IEEE C37.118 1% TVE cap), i.e. ≈ **0.6 / 2.1 / 4.1 sensors' worth** of loss. Dropout-aware is **not better at any σ** at k=0 (−0.008 to −0.034, all CIs span zero, p ≥ 0.107) — a clean null and the sharpest confirmation of the mask mechanism: **noise leaves the mask at 1**, so the learned "(0,0,0) = dark" response has nothing to fire on. DA's *dropout* advantage survives, attenuated: at k=10, +0.627 (clean) → +0.579 → +0.428 → +0.209; only the first two survive Holm over the 50-test noise family (p_Holm 0.029 / 0.023), so claim "continues to help", NOT "undiminished". **Noise and dropout compound below the floor**: σ=0.01 with k=10 gives 0.018, under the 0.026 trivial baseline. `dropout_study_noise/` (`scripts/run_noise_robustness_study.py`, 3 seeds, both arms). **CAUTION:** an early truncated-subset run suggested a far worse collapse (~77% loss at σ=0.002), an artifact of few classes in a 400-batch subset; the full-split number is a 33% loss. **Never size this from a subset.**
- **Absolute-scale AUDC** (`statistics/absolute_audc.csv`, `scripts/build_absolute_audc.py`) recomputes Eq. 8 on raw macro-F1. The inversion survives (RGCN 0.432 > RGSAGE-Max 0.340), but **RGATv2's top relative-AUDC position is a normalization artifact** (0.626 relative → 0.324 absolute, 4th of 5) and is no longer claimed as robustness.
- **Trivial reference:** an always-"no fault" classifier scores 26-class macro-F1 **0.0256**. Standard RGSAGE-Max at k=10 (0.048) is only **1.9×** that, i.e. localization is gone while detection survives.

**Deliberately NOT done:** per-architecture depth selection (the full RGATv2 fix — its claim was neutralized in prose instead); a learned imputation baseline (Feature Propagation / VGAE), now cited as a named gap; the incidence-matched contiguous control.

**Runs listed as QUEUED on 2026-08-17 — outputs now exist (checked 2026-10-16; numbers not summarized here):** `scripts/run_localization_metrics.py` → `dropout_study_localization/` (26-class / 25-class localization / class-0 detection split of macro-F1, RGSAGE-Max standard + dropout-aware, seeds 42/43/44); `scripts/run_reviewer_retrains.py` → `scenario_heldout_{std,da}/` (RGSAGE-Max, 3 seeds each), `dropout_aware_k10/` (3 seeds), `fullscale_extra_seeds/` (RGCN + RGSAGE-Max, seeds 45–47). Folders with matching names also hold `degradation_curves.csv`: `dropout_study_{scenario_std,scenario_da,k10,extra_seeds}/` (no script names them; provenance not recorded).

## Appendix — historical diagnostics (SUPERSEDED, do not quote)
From the **`trend_medium_A` era**: a much smaller corpus, depth-1/2/3 probes, before the frozen depth-4 config. Kept only to record *why* depth 4 was chosen and why edges were kept. **Not comparable to anything above.** Left: edge ablation (GO signal for keeping the graph). Right: depth probe 1 → 2 (197.7 min).

| Metric | edges ON | edges OFF | ratio | depth 1 | depth 2 | delta |
|---|---|---|---|---|---|---|
| graph macro-F1 (soft-voted) | 0.343 | 0.124 | 2.8× | 0.439 | 0.571 | +0.132 |
| per-window macro-F1 | 0.227 | 0.082 | 2.8× | — | — | — |
| F1 @ 0.1 Ω | 0.471 | 0.152 | 3.1× | 0.584 | 0.726 | +0.142 |
| F1 @ 1 Ω | 0.352 | 0.132 | 2.7× | 0.457 | 0.578 | +0.121 |
| F1 @ 10 Ω | 0.153 | 0.049 | 3.1× | 0.199 | 0.371 | +0.172 |
| no-fault precision | 0.994 | 0.922 | — | — | — | — |

Conclusion at the time: 1-hop depth was a genuine architectural ceiling, not underfitting — which led to the depth-4 freeze. Full depth-3/4 probe outputs: `diag_depth3/`, `diag_depth4/`, `diag_combined_depth4_epochs30/`.
