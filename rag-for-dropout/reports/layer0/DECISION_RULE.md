# DECISION_RULE.md — kNN agreement in the gate: pre-registration (fixed before any result)

Written 2026-10-30 08:50 IST, **before** any error-detection AUROC with kNN agreement was computed in this step. It is committed to
`gridsense-eval` before the analysis runs. Do not edit it after the results exist; write corrections as a dated addendum at the end.

## 1. The author's rule (2026-10-30, verbatim in substance)
On the **held-out store**, **post-fault windows**, **val split**: compare the error-detection AUROC of (a) the isotonic probability and
(b) a combination of probability and kNN agreement. Fit the combination on cal-A. Calculate the AUROC difference on cal-B with a
bootstrap by run (1000 samples, 95 % CI).
- CI includes 0 → remove kNN agreement from the gate; use probability and margin only; keep the neighbours in the report as precedents
  labelled "similar simulated cases"; do not use them as evidence of correctness.
- CI above 0 → keep τ_k in the gate; ask the author before running seeds 43 and 44.
Also (author): O20 — the held-out store is used for the gate and for τ_k; the standard store is secondary only. Report the decision and
the numbers; stop after step 2.

## 2. What the rule leaves open, and my reading (set now, before the result)
The rule does not name the dropout level k, the number of neighbours, the model of the combination, or the mask offset. I fix:
| Item | Fixed value | Reason |
|---|---|---|
| Windows | `val`, post-fault (w ≥ 21), w0020 excluded (`unlabelable_window_mask`); cal-A / cal-B = the 50/50 split by run, stratified by scenario, seed 42 (S2) | author rule; O12 |
| Store | Scenario-held-out 10 % train-run subsample, representation (b) mean + max | O20, O4, O15 |
| k_nn | 10 | the value used for selection in S2 and in the control |
| kNN agreement | fraction of the 10 neighbours whose label equals the STGNN top-1 class of the query window | `EVAL.md` § 2 |
| (a) | isotonic probability of the top-1 vote share (vote / 25), map fitted on cal-A at k = 0 in S2 | O2 |
| (b) | logistic regression of "prediction correct" on [isotonic probability, kNN agreement], unregularised (C = 1e6), **fitted on cal-A post-fault windows, separately for each k** | author: fit on cal-A |
| Score | error score = 1 − P(correct); positive class = error; AUROC counts ties as 0.5 | standard |
| Mask offset | offset 0 for k > 0 (offsets 1 and 2 reported as a stability check) | one realisation per window |
| Bootstrap | 1000 resamples of the cal-B **runs** (a run keeps all its windows at every k), seed 0, 95 % percentile CI | author rule |

### The dropout level k (the reading that decides)
**Primary statistic: D = mean over k ∈ {0, 1, 2, 5, 10} of the within-k AUROC difference AUROC_b(k) − AUROC_a(k).**
Reason: pooling windows of different k mixes the dropout level into the comparison, and agreement might then look useful only because it
tells k, which the gate already knows from the sensor mask. A within-k comparison asks the question the author asks: does agreement add
information about whether one prediction is right, given the probability?
Decision on D: kNN agreement is **kept** only if the lower end of the 95 % CI of D is above 0. Otherwise it is **removed**
(this includes a CI that is entirely below 0).

### Secondary readings (reported, never decisive)
Pooled over k (one AUROC on all five k, combination fitted on pooled cal-A); k = 0 only; each k alone; offsets 1 and 2; all windows
instead of post-fault; the standard store. If a secondary reading would give the other decision, the report says so in the first lines
and asks the author to confirm the reading; the primary reading still decides what is implemented.

## 3. O6 thresholds (fixed now)
- Selected on cal-B, post-fault windows. Objective: maximum coverage subject to accuracy ≥ 95 % on the CONFIDENT windows. Coverage at the
  98 % target is also reported. Gate variants: p only; p + margin (top-1 − top-2 vote share); p + kNN agreement. The variant chosen by
  § 2 is the gate; the others are secondary.
- One global threshold set (all five k pooled, equal weight per window) against per-k thresholds; both compared on cal-B, with the
  per-k accuracy of the CONFIDENT windows shown. Out-of-selection check on cal-A (thresholds are not tuned there).
- If the coverage at 95 % is below 50 %, the report flags it in its first lines and the author is asked; nothing downstream depends on it.
- `test` is not touched in step 2 (the gate is not frozen until the author reads the numbers).

## 4. Not decided here (asked, not chosen)
Conformal candidate sets (nominal coverage is an open O6 value) — not computed. Full train store — not built (O15: ask first).

## Addenda
(none)
