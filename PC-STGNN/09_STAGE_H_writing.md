# Stage H — Analysis, Writing, Release (Weeks 19–26)

**Purpose.** Turn the tables into an argument, and release what a reader needs to check it.
**Compute.** None.

## S7.1 — Error attribution

**Do.** Overlay each model's confusion matrix on the S2.1 identifiability heatmap (F-C1, at the
matching k and σ₀).

**Check.** What fraction of each model's residual errors falls in pairs with D < 1?

**Target sentence.** "The residual errors concentrate in location pairs that are physically
indistinguishable under the available sensor subset." This is the deepest section in the paper,
and it works whether or not the model gains are large.

**Figure produced.** F-H1.

## S7.2 — Case studies

**Do.** Three events, each shown as the residual profile r̃_f across all 26 candidates plus the
network's P(f|X) before and after fusion:
1. one correctly gated silent failure (G2),
2. one gate failure (a clean sensor suppressed, or a corrupted one passed),
3. one 10 Ω event where physics flips the prediction.

**Figure produced.** F-H2. Reviewers remember these.

## S7.3 — Paper structure

Edit, don't restructure, from the Paper-1 skeleton (`paper/overleaf.tex`); the same
Elsevier/Procedia class unless the venue changes. Target sections:

| § | Content | Sources |
|---|---|---|
| 1 | Introduction — the three failure modes mask-based robustness cannot see (unseen R, noise, silent sensors) | Paper 1 limitations paragraph |
| 2 | Related work — T1: add "physics source" and "silent faults" columns to Paper 1's table | `PIGNN-SURVEY.md` |
| 3 | Physics — the operator, its fidelity (F-B3, F-B4) and identifiability (F-C1, F-C2); Y-bus for analysis only | Stage B, C |
| 4 | Method — PC-STGNN, PCRG-STGNN, F3 architecture figure, gradient audit table | `01_SPEC.md` |
| 5 | Setup — inherited constants, injectors, OOD sets (T2, T3), pre-registration | Stage A, D |
| 6 | Baseline ladder T6 and the training-free results | Stage C |
| 7 | Results — T5, T7, T8, T9, T10, T11, F-G1…F-G6, F-F1…F-F4 | Stage E–G |
| 8 | Analysis — F-H1, F-H2, cost T12, statistics T13 | Stage H |
| 9 | Limitations — same simulator (answered by T11), magnitudes only, scalar load, closed set of 25 buses, all 825 scenarios seen in training (optional: scenario-held-out split) | — |
| 10 | Conclusion | — |

**Claim discipline.** `12_CHECKLISTS.md` §D applies to every sentence. The framing under each
Gate-2 branch is written in `13_BUDGET_TIMELINE.md` "Fallback map" — pick the row that matches
the pre-registered outcome, not the most flattering one.

## S7.4 — Release

**Do.** Code, the signature table (`operator/`), the OOD generator and its seeds, the perturbed
tables, the silent-failure injector, the pre-registration note with its commit hash. PPGN set the
precedent for public code in this niche; releasing measurably improves reception.

**Check.** A fresh clone reproduces T6 (training-free rows) from the released table alone.
