# 30 — Reporting rules: claims that may NOT be upgraded

The project's headline finding is genuinely strong. That makes it tempting to round the weaker
findings up to match. Don't. This file fixes the exact status of each claim.

## The gate does not win

**SDR-STGNN MATCHES but does NOT beat dropout-aware training.**
AUDC 0.892 vs 0.913; Ret@10 75.2 % vs 80.3 %; ΔAUDC = +0.021 for dropout-aware, **p = 0.098** —
a statistical tie. It also costs **+18 % inference latency** for zero robustness gain.

Never write that the gate wins, leads, or is "our best model". The mechanism for why it loses is
known and should be stated: silencing a dropped node at *every* hop also blocks **multi-hop
routing through it**, destroying information a healthy 2-hop neighbour could have relayed.

## May be stated as significant

- Dropout-aware > standard at k=10: Δ=+0.626, p_Holm=0.003
- Retention@10: +74.6 pp, p_Holm=0.003
- AUDC: +0.511, p_Holm=0.003
- Dropout-aware > imputation at k=10: +0.574, p_Holm=0.008
- The AUDC gain holds for **all five** architectures, p_Holm ≤ 0.027
- Clean-F1 cost of dropout-aware is **null**: −0.008, p=0.181 (a null the paper *claims*, so
  reporting it as null is correct)

## May NOT be called significant

- **The accuracy/robustness inversion** (RGCN over RGSAGE-Max at k=10): Δ=+0.116, **p=0.076**.
  Same sign in all 3 seeds and 2.4–6.7× in size, but the gap varies. Report as an **observed
  trend**, never as a proven law.
- **The k=0 architecture ranking**: best pair p=0.023, nothing survives Holm.
- **Imputation over zeroing at k=10**: +0.051, **p=0.102**.

## Targeted (worst-case) dropout — state both halves

Dropout-aware training **is penalized** by targeting (AUDC 0.913 → **0.747**, p=0.003; p≤0.014 at
every k). Worst-case loss is a **distinct stressor, not one the mitigation absorbs** — say so.

What may be claimed: it still retains +0.307 to +0.473 macro-F1 over standard under the same
targeting (p≤0.008), and from k=2 upward its *worst-case* score exceeds the standard model's
*average-case* score. At **k=1 the two are indistinguishable** (+0.018, p=0.33) — no win there.

Also state the caveat: no-fault windows have no fault bus and fall back to random selection, so
the reported macro-F1 blends worst-case and average-case windows and is a **conservative bound**.

## Persistent (per-event) dropout — a null, with a caveat

Nothing moves (< 0.003 macro-F1; min p_Holm = 0.46). The reason must be given: the decision rule
is **per window** — soft voting pools the 25 nodes of one window, never the 40 windows of an
event. And the caveat must be given: **this equivalence would not hold for an event-level rule**,
which per-window resampling would flatter.

## Accuracy is a floor

Best-validation epoch was **27.7–28.7 of 30** for RGCN/RGSAGE-*, i.e. those models were still
improving when the budget ended. Report their accuracies as a floor, not a converged ceiling.

## Cost claims are measured, not asserted

Dropout-aware has **identical** params (95,258) and latency (0.639 vs 0.640 ms @ B=1) — this was
measured in `results/runtime/`, so say "measured". Its only cost is **+2.7 %** training sec/epoch.

## Do not promise the green feeder

The reconfigured "green" feeder scenario was **CUT (2026-08-04)** as out of scope. No green
dataset, runs, or claims exist. `configure_green_feeder` is dead/future-work code. It may appear
only as a future-work clause ("cross-topology generalization to reconfigured feeders").

## Get the mask mechanism right

Mask channels are **static but not all-ones**: only 14 of 25 nodes are three-phase, the rest show
patterns like (1,0,0). Dropout zeroes all 6 channels, so a dropped node's mask becomes (0,0,0) —
a pattern that node never shows in standard training. That is the mechanism for why dropout-aware
training works. **Never restate it as "the mask is always 1".**
