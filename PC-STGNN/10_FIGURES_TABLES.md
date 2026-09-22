# Figure and Table Registry

Every asset, where it comes from, and whether it is main text or appendix.

## Figures

| ID | Figure | Source step | Placement |
|---|---|---|---|
| F1 | Measured-only graph: 25 µPMU nodes, 24 edges, 6 instrumented fault locations ringed | reuse Paper 1 Fig. 1 (`paper/figures/measured_only_graph.png`) | §3 |
| F2 | Fault signature vs resistance at one bus | reuse Paper 1 Fig. 2 (`paper/figures/fault_signature.png`) | §1 or §3 |
| F3 | PCRG-STGNN architecture diagram (operator, input, pass 1, residual, fusion, LOO gate, pass 2, loss, ablation switch map) | — | §4 |
| F-B1 | Transfer-sensitivity map \|Z_pf\|/\|Z_ff\| over the feeder | S1.1 | §3 |
| F-B2 | L-grid interpolation error vs step | S1.2 | appendix |
| F-B3 | Fidelity: \|e\| distribution by R and fault type | S1.4 | §3 |
| F-B4 | Profiled L* vs stored load multiplier, 1:1 line — the load-model validation | S1.5 | §3 |
| F-B5 | r̃ at the true hypothesis, by R | S1.6 | appendix |
| F-B6 | PSM F1 vs R-grid density; argmin_R accuracy | S1.7 | appendix |
| F-C1 | Identifiability heatmaps D(f₁,f₂) at k = 0, 2, 5, 10 | S2.1 | §3 |
| F-C2 | Fraction of indistinguishable pairs vs k, per R and per σ₀ | S2.1 | §3 |
| F-C3 | Table D vs empirical confusability scatter | S2.2 | appendix |
| F-C4 | OOD extrapolation, pilot version | S2.5 | internal only — not published |
| F-E1 | Validation F1 vs λ₁ | S4.3 | appendix |
| F-E2 | Ablation bar chart with seed spread | S4.4 | §7 (optional) |
| F-F1 | Detection ROC curves per corruption type | S5.1 | §7 |
| F-F2 | Detection AUROC vs corruption severity | S5.1 | §7 |
| F-F3 | LOO vs in-sample detection AUROC | S5.2 | §5 |
| F-F4 | Gate value gᵢ distribution, clean vs corrupted | S5.3 | §7 |
| F-G1 | Macro-F1 vs k, all models | S6.1 | §7 |
| F-G2 | F1 by fault resistance, in-distribution | S6.1 | §7 |
| F-G3 | F1 vs R across {0.1, 1, 3, 5, 10, 20, 30} Ω for PSM / R0 / R5 / B2ᵃ — the headline extrapolation figure | S6.1 | §7 |
| F-G4 | F1 vs noise level | S6.1 | §7 |
| F-G5 | Structured vs random vs persistent dropout | S6.1 | §7 |
| F-G6 | ΔF1 vs feeder-model perturbation level | S6.2 | §7 |
| F-H1 | Confusion matrix overlaid on identifiability | S7.1 | §8 |
| F-H2 | Residual profile r̃_f case studies (3 events) | S7.2 | §8 |
| F-A1 | Window census histogram | S0.3 | appendix |

## Tables

| ID | Table | Source |
|---|---|---|
| T1 | Related-work positioning (Paper 1 `tab:related` + physics source + silent faults columns) | §2 |
| T2 | Dataset and OOD set configuration | S2.0 |
| T3 | Signature table configuration: axes, grid, size, build cost | S1.3 |
| T4 | Fidelity summary: bias by R and type; L* recovery | S1.4, S1.5 |
| T5 | Main results: all models × k (macro-F1, retention, AUDC relative and absolute) | S6.1 |
| T6 | Baseline ladder B1 → B5 (see below) | S2.3, S2.4, S6.1 |
| T7 | Core ablation R0–R7 + B2ᵃ | S4.4 |
| T8 | OOD resistance results | S6.1 |
| T9 | Silent failures + detection AUROC | S5.1, S6.1 |
| T10 | Structured/persistent dropout and measurement noise | S6.1 |
| T11 | Model-mismatch robustness | S6.2 |
| T12 | Computational cost: params, FLOPs, train h, inference latency | S6.4 |
| T13 | Statistical tests: paired bootstrap CIs and p-values | S6.3 |

## Baseline ladder (T6) — the argument in one table

| # | Baseline | What it isolates | Status |
|---|---|---|---|
| B1ᵃ | Rank-1 sag-template matched filter, τ fitted on validation | direction-only physics | exists — F1 0.162 at every k |
| B1 | ΔV nearest-signature, train-split nominal reference, no profiling | raw table physics | S2.3 |
| B2 | PSM: argmin_f r̃_f, full (τ, R, L) profiling | what profiling and the load axis buy | S2.4 |
| B2ᵇ | Logistic regression / random forest on summary features | non-graph learning | exists — 0.713 / 0.633 |
| B3 | RGSAGE-Max (Paper 1) | learned representation | exists — 0.846 |
| B4 | Dropout-aware RGSAGE-Max (R0) | + dropout-aware training | exists — 0.839 |
| B5 | PC-STGNN (R5) | physics + learned representation | Stage E |

Reads as direction-only physics → table physics → profiled physics → non-graph learning →
learned representation → both. It is the cleanest answer to "is the GNN necessary" — and if B2
lands above B3 on the clean corpus, the ladder is still the right table; the paper's question
just moves to the noise, dropout and mismatch columns.
