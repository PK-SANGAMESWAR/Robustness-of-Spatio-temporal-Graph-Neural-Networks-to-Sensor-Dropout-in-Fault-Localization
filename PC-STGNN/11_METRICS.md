# Metric Definitions

Every metric used anywhere in the project, defined once. Report mean ± std over seeds 42/43/44
unless stated otherwise; 1-seed rows are marked.

## 1. Localization performance

| Metric | Definition |
|---|---|
| Macro-F1 (graph-level) | Per-window graph prediction from soft voting; macro-averaged F1 over the 26 classes. Primary metric, as in Paper 1 |
| Precision / recall / accuracy | Macro-averaged, same protocol |
| Per-location F1 | F1 for each of the 25 fault locations separately — feeds F-H1 |
| Detection / localization split | 26-class macro-F1 decomposed into class-0 F1 (detection) and 25-class macro-F1 (localization), as in Paper 1's `run_localization_metrics.py` |
| F1 by resistance | Macro-F1 restricted to events at each R. The 10 Ω column is the one that matters |
| F1 by fault type | Macro-F1 restricted to each of the 11 types |
| F1(k) | Macro-F1 at dropout level k ∈ {0,1,2,5,10} |
| Retention R(k) | F1(k) / F1(0) |
| AUDC (relative) | Area under R(k) vs k, normalised exactly as Paper 1 Eq. 8. Keep the identical normalisation or all cross-paper comparison breaks |
| AUDC (absolute) | The same integral on raw F1(k) (`scripts/build_absolute_audc.py`). Report both — Paper 1 found the relative form flatters weak models |
| Trivial reference | Always-"no fault" classifier: 26-class macro-F1 0.0256 |

## 2. Physics-operator diagnostics

| Metric | Definition |
|---|---|
| r̃_true | Normalised residual at the true location — lower is better |
| Residual rank | Rank of the true location when candidates are sorted by r̃. Rank-1 = PSM correct |
| PSM top-1 / top-3 accuracy | Fraction of windows where the true location is the best / among the 3 best by r̃ |
| Residual margin | r̃_(2) − r̃_(1) — the gap between best and runner-up. Measures decisiveness, not just correctness |
| argmin_R error | Distance in grid steps between the profiled R* and the true R (in-distribution only) |
| argmin_L error | Distance in grid steps between the profiled L* and the stored load multiplier (S1.5) |
| Dropped-sensor reconstruction RMSE | RMSE of μ at the best hypothesis against true values on D. Not a loss term; a diagnostic |

Report r̃_true and residual rank for every model, not just PSM — pass each model's predicted
location through the operator. It makes physical consistency comparable across models, including
ones with no physics.

## 3. Silent-failure detection

| Metric | Definition |
|---|---|
| AUROC | r̃ᵢ^LOO as a score separating corrupted from clean sensors. Primary. Hard gate at 0.75 |
| AUPRC | Better when few sensors are corrupted (s = 1 of 25) |
| TPR @ FPR = 0.05 | The operationally meaningful number — how many bad sensors you catch at a 5 % false-alarm rate |
| False-suppression rate | Fraction of clean sensors receiving gᵢ < 0.5. Report even when it is good |

## 4. Fidelity (Stage B)

| Metric | Definition | Pass |
|---|---|---|
| Bias | median and IQR of \|e_ip\| = \|s_ip − μ[f,τ,R,L,i,p]\| on training events | median \|e\|/sag < 0.3 at 10 Ω |
| L* recovery | fraction of training events with \|L* − L_true\| ≤ 1 grid step at f_true | > 90 % |
| τ* recovery | fraction with τ* = τ_true at f_true | > 90 % |
| L-grid interpolation error | max p.u. error of linear interpolation between grid points | < 1e-3 at step 0.05 |
| Residual floor calibration | median r̃_{f_true}(A) on the training split | ∈ [0.3, 3] |
| Snapshot equivalence | max \|V_snapshot − V_corpus\| over instrumented channels | < 1e-3 p.u.; sets σ_ε |

## 5. Identifiability

| Metric | Definition |
|---|---|
| D(f₁,f₂;A) | min over (τ,R,L) of Σ_{(i,p)∈A} (μ_f₁ − μ_f₂)² / (2(σ_ε² + σ₀²)) — Fisher-style, averaged over random sensor subsets |
| Indistinguishable-pair fraction | fraction of pairs with D < 1, per k, per R, per σ₀ |
| min_f₂ D(f₁, f₂) | per-location worst-case separability — identifies which locations are structurally hard |
| Empirical confusability | nearest-neighbour confusion rate between training events at different locations |

## 6. Statistics

- Seed spread: mean ± std over 42/43/44. Lead with this.
- Paired two-sided t-test over seeds (n = 3, df = 2) with Holm correction per family, as Paper 1
  (`scripts/run_significance_tests.py`).
- Paired cluster bootstrap over test **events** (6,175; not windows), 10,000 resamples, 95 % CI,
  for the pre-registered comparisons only: R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2.
- Report CIs, not bare p-values. Significance is cheap at this test size; seed variance is the
  binding constraint. Never upgrade an n = 3 non-significant ordering to "significant".

## 7. Cost

Parameter count · FLOPs per window · training GPU-hours · inference latency per window,
separately for the backbone, the residual computation, and (G2 only) the LOO pass plus second
forward. Baseline numbers exist in `results/runtime/`.

## 8. Calibration (optional, cheap)

Expected calibration error of P(f|X), before and after physics fusion. Reviewers like it, and
physics fusion plausibly improves it — a free extra result if it does.
