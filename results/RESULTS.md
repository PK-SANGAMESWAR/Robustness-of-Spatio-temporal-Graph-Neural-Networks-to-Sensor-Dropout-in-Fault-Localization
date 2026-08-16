# RESULTS.md — current headline numbers

**Updated 2026-08-16.** Every number here is post the mask-diversity bugfix (`798406e`) and is
copied from the CSV/JSON files named alongside it. Where this file and `paper/overleaf.tex`
disagree, the `.tex` wins; where the `.tex` and a CSV disagree, the CSV wins.

Corpus: `dataset_fullscale_50` — 825 scenarios × 50 runs × 40 windows = **1,650,000 windows**,
event-level 70/15/15 split (0 leakage), 6,175 test events. Frozen config: GNN depth 4, 30 epochs,
unweighted CE, grad-clip 1.0, 3 warm-up epochs. Seeds 42/43/44 throughout.

---

## 1. Clean-condition benchmark (k = 0)

Soft-voted graph macro-F1, mean ± std over 3 training seeds.
Source: `fullscale_final/benchmark_summary.csv`.

| Model | macro-F1 | per-window F1 | 0.1 Ω | 1 Ω | 10 Ω |
|---|---|---|---|---|---|
| GRU (no graph) | 0.286 ± 0.009 | 0.150 | 0.410 | 0.292 | 0.105 |
| RGATv2 | 0.522 ± 0.087 | 0.460 | 0.630 | 0.574 | 0.312 |
| RGCN | 0.712 ± 0.017 | 0.601 | 0.874 | 0.792 | 0.428 |
| RGSAGE-Mean | 0.786 ± 0.023 | 0.660 | 0.906 | 0.843 | 0.587 |
| **RGSAGE-Max** | **0.846 ± 0.007** | **0.718** | **0.949** | **0.893** | **0.676** |

No-fault precision ≥ 0.97, recall ≈ 1.00 for every model. Best-validation epoch was 27.7–28.7 of
30 for RGCN/RGSAGE-*, so those accuracies are a **floor**, not a converged ceiling.

## 2. Random sensor dropout — standard training

Macro-F1 vs k μPMUs dropped at inference (of 25). 3 training seeds × 3 mask seeds.
Source: `dropout_study/degradation_curves.csv`.

| Model | k=0 | k=1 | k=2 | k=5 | k=10 | AUDC | Ret.@10 |
|---|---|---|---|---|---|---|---|
| **RGSAGE-Max** | 0.846 | 0.736 | 0.590 | 0.236 | **0.048** | **0.402** | **5.7 %** |
| RGSAGE-Mean | 0.786 | 0.720 | 0.647 | 0.393 | 0.108 | 0.540 | 13.8 % |
| RGCN | 0.712 | 0.664 | 0.610 | 0.418 | 0.164 | 0.607 | 23.1 % |
| RGATv2 | 0.522 | 0.490 | 0.444 | 0.302 | 0.157 | 0.621 | 30.1 % |
| GRU | 0.286 | 0.273 | 0.252 | 0.111 | 0.051 | 0.521 | 17.9 % |

**The accuracy/robustness inversion:** the most accurate model is the most fragile. Same sign in
all 3 seeds, but **not** statistically significant at n=3 (Δ=+0.116 for RGCN over RGSAGE-Max at
k=10, p=0.076) — report as an observed trend, not a proven law.

## 3. Random sensor dropout — dropout-aware training

Source: `dropout_study_aware/degradation_curves.csv`.

| Model | k=0 | k=1 | k=2 | k=5 | k=10 | AUDC (std → aware) |
|---|---|---|---|---|---|---|
| **RGSAGE-Max** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | 0.402 → **0.913** |
| RGSAGE-Mean | 0.673 | 0.666 | 0.657 | 0.628 | 0.550 | 0.540 → 0.921 |
| RGCN | 0.613 | 0.607 | 0.600 | 0.579 | 0.533 | 0.607 → 0.940 |
| RGATv2 | 0.553 | 0.550 | 0.546 | 0.533 | 0.505 | 0.621 → 0.960 |
| GRU | 0.300 | 0.300 | 0.299 | 0.278 | 0.078 | 0.521 → 0.784 |

The AUDC gain is significant for **all five** architectures (p_Holm ≤ 0.027). Note RGCN and
RGSAGE-Mean pay ~0.1 macro-F1 in *clean* accuracy for it; RGSAGE-Max does not (−0.008, p=0.181).

## 4. Mitigation comparison (RGSAGE-Max backbone)

Source: `../paper/figures/sdr_robustness_summary.csv`, `dropout_study_impute/imputation_summary.json`.

| Variant | AUDC | k=0 | k=1 | k=2 | k=5 | k=10 | Ret.@10 |
|---|---|---|---|---|---|---|---|
| Standard (zeroing) | 0.402 | 0.846 | 0.736 | 0.590 | 0.236 | 0.048 | 5.7 % |
| Neighbour-mean imputation | 0.476 | 0.846 | 0.739 | 0.637 | 0.336 | 0.099 | 11.8 % |
| **Dropout-aware training** | **0.913** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | **80.3 %** |
| SDR-STGNN (gated) | 0.892 | 0.839 | 0.827 | 0.813 | 0.758 | 0.632 | 75.2 % |

**SDR-STGNN matches but does not beat dropout-aware training** (ΔAUDC = 0.021 in favour of
dropout-aware, p=0.098 — a statistical tie) and costs **+18 % inference latency**. Report this
faithfully; do not claim the gate wins.

## 5. Targeted (worst-case) dropout

Drops the k μPMUs nearest the fault bus instead of random ones. Deterministic given the label →
no mask seeds; error bars over training seeds only. RGSAGE-Max only.
Source: `dropout_study_targeted/targeted_curves_*.csv`.

| Variant | k=0 | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| Standard, **targeted** | 0.846 | **0.448 ± 0.039** | 0.361 | 0.206 | 0.048 |
| Standard, random (ref) | 0.846 | 0.736 ± 0.013 | 0.590 | 0.236 | 0.048 |
| Dropout-aware, **targeted** | 0.839 | 0.755 | 0.695 | 0.600 | **0.521** |

Targeting is far more damaging in the regime where the model is still usable — at k=1 the
standard model drops **3.6× further** than under random loss (p=0.010). Dropout-aware training is
penalized too (AUDC 0.913 → **0.747**, p=0.003), so worst-case loss is a **distinct stressor, not
one the mitigation absorbs** — but from k=2 upward its *worst-case* score still exceeds the
standard model's *average-case* score (0.695 vs 0.590 at k=2, p=0.042). At k=1 the two are
indistinguishable (+0.018, p=0.33).

*Caveat:* no-fault windows have no fault bus and fall back to random selection, so these numbers
blend worst-case and average-case windows — a **conservative bound** on the true worst case.

## 6. Persistent (per-event) dropout — a null result

Draws the dropped set once per event and holds it across that event's 40 windows.
Source: `dropout_study_persistent/persistent_curves_*.csv`.

**Nothing moves.** No point on either curve shifts by more than **0.003** macro-F1 (standard AUDC
0.402 → 0.401; dropout-aware 0.913 → 0.913; Ret@10 5.7 % → 5.6 % and 80.3 % → 80.1 %), and no
difference survives Holm correction across the 12 paired comparisons (min p_Holm = 0.46).

**Why:** the decision rule is per window — soft voting pools the 25 nodes of a *single* window,
never the 40 windows of an event. A window's prediction depends only on its own mask, whose
marginal distribution is identical under both schemes. **This equivalence would not hold for an
event-level rule**, which per-window resampling would flatter.

## 7. Significance and cost

`statistics/significance_tests.json` — paired two-sided t-tests over seeds 42/43/44 (n=3, df=2),
95 % CIs, Holm-Bonferroni per family, plus a paired cluster bootstrap over the 6,175 test events.

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

`runtime/runtime_summary.json` — dropout-aware has **identical** params (95,258) and latency
(0.639 vs 0.640 ms @ B=1; 6,038 windows/s both); its only cost is **+2.7 %** training sec/epoch.
SDR-STGNN's gate adds 0 params but **+18 %** latency for no robustness gain. Activation memory is
145 MiB for *all* models (the shared GRU-128 encoder dominates). Training: 4.7–12.3 GPU-hours per
run on a T1000 8 GB.

---

## Appendix — historical diagnostics (SUPERSEDED, do not quote)

The numbers below are from the **`trend_medium_A` era**: a much smaller corpus, depth-1/2/3
probes, before the frozen depth-4 config. They are kept only to record *why* depth 4 was chosen
and why edges were kept. **They are not comparable to anything above.**

**Edge ablation (GO signal for keeping the graph):**

| | edges ON | edges OFF | ratio |
|---|---|---|---|
| graph macro-F1 (soft-voted) | 0.343 | 0.124 | 2.8× |
| per-window macro-F1 | 0.227 | 0.082 | 2.8× |
| F1 @ 0.1 Ω | 0.471 | 0.152 | 3.1× |
| F1 @ 1 Ω | 0.352 | 0.132 | 2.7× |
| F1 @ 10 Ω | 0.153 | 0.049 | 3.1× |
| no-fault precision | 0.994 | 0.922 | — |

**Depth probe (depth 1 → 2), 197.7 min:**

| | depth 1 | depth 2 | delta |
|---|---|---|---|
| graph macro-F1 | 0.439 | 0.571 | +0.132 |
| F1 @ 0.1 Ω | 0.584 | 0.726 | +0.142 |
| F1 @ 1 Ω | 0.457 | 0.578 | +0.121 |
| F1 @ 10 Ω | 0.199 | 0.371 | +0.172 |

Conclusion at the time: 1-hop depth was a genuine architectural ceiling, not underfitting — which
is what led to the depth-4 freeze. Full depth-3/4 probe outputs are in `diag_depth3/`, `diag_depth4/`,
and `diag_combined_depth4_epochs30/`.
