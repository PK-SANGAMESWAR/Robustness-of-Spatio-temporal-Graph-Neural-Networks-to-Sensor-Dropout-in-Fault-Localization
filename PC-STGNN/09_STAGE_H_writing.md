# Stage H — Analysis, Writing, Release (Weeks 19–26)

**Purpose.** Turn the tables into an argument, and release what a reader needs to check it.
**Compute.** None.

## Framing, fixed before any of this is written `[Added 2026-09-23]`

Two sentences that must survive to the final draft, from `04_STAGE_C_gonogo.md` S2.4ᵇ and
`02_STAGE_A_audit.md` S0.3:

1. PSM's clean-corpus accuracy is a **sanity check on operator fidelity**, because the table and
   the corpus come from the same simulator. The contribution is where learning still helps, and
   where physics still helps — under noise, unseen resistance, dropout, silent sensors and a
   wrong table — not a leaderboard win on the clean split.
2. Every absolute macro-F1 in both papers sits under the **unlabelable-window ceiling**: the
   w = 20 windows carry a fault label with zero post-inception samples and nothing can beat
   chance on them. Quote the measured ceiling (**0.9744**) the first time an absolute number
   appears, together with the measured no-leakage result (all 33 Paper-1 runs score 0.0003–0.0418
   on that subset against a chance rate of 0.0385). This is a genuinely good result — it bounds
   every absolute number in both papers *and* demonstrates the absence of label leakage — so it
   belongs in the paper, not in a footnote.

3. `[Added 2026-09-23]` **The oracle/realistic pair is the paper's spine.** The operator is the
   corpus's own generator (S0.4: 1.02e-12 p.u.), so the exact-operator / noise-free column is an
   **oracle upper bound** and must be labelled, verbatim, *"oracle upper bound: perfect model
   knowledge, perfect measurements."* Headline tables lead with the **realistic** operating point
   — noise plus a perturbed table — and place the oracle column beside it. Written this way the
   degeneracy stops being a liability and becomes an asset almost no paper in this area can
   produce: an oracle and a realistic measurement of the same method, on the same corpus, with
   the gap between them quantified. Written the other way — oracle numbers presented as results —
   it invites the one-sentence rejection in `04_STAGE_C_gonogo.md` S2.0ᵃ.

4. `[Added 2026-09-23]` **Identifiability is a contribution, not a supporting figure.** *"How
   much can 25 sparse, magnitude-only µPMUs reveal about fault location, measured with an exact
   physics operator?"* is defensible, novel, and reviewer-proof precisely because it does not
   depend on any model gain — and the exactness that degenerates the accuracy comparisons is what
   makes it clean, since D(f₁,f₂;A) then measures the information content of the sensing
   infrastructure with zero model error mixed in. It is the one result that survives every
   Gate-2 branch, including the PIVOT row. Give it a numbered contribution in §1 and its own
   subsection in §3.

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
| 3 | Physics — the operator, its fidelity (F-B3 is a **construction** check, F-B4 the real one) and **identifiability as a contribution subsection** (F-C1, F-C2); Y-bus for analysis only | Stage B, C |
| 4 | Method — PC-STGNN, PCRG-STGNN, F3 architecture figure, gradient audit table | `01_SPEC.md` |
| 5 | Setup — inherited constants, injectors, OOD sets (T2, T3), pre-registration | Stage A, D |
| 6 | Baseline ladder T6 and the training-free results — PSM's clean k = 0 row is captioned as an **operator-fidelity check**, not as physics beating the network (S2.4ᵇ) | Stage C |
| 7 | Results — T5, T7, T8, T9, T10, T11, F-G1…F-G6, F-F1…F-F4. **Noise (F-G4) and model mismatch (F-G6) are main-body axes alongside OOD (F-G3)**, not appendix material: they are where physics and learning fail differently. `[Corrected 2026-09-23]` Every headline table leads with the realistic operating point and carries the oracle column beside it, labelled as in framing point 3; R5 is compared against **R0ⁿ** (`06_STAGE_E` S4.3ᵇ), and the R8 arm reports what happens to the fusion without noise | Stage E–G |
| 8 | Analysis — F-H1, F-H2, cost T12, statistics T13 | Stage H |
| 9 | Limitations — same simulator (**stated plainly as the oracle setting**, answered by T11), magnitudes only, scalar load, closed set of 25 buses, all 825 scenarios seen in training (optional: scenario-held-out split), and the **unlabelable-window ceiling** of S0.3 that caps every absolute macro-F1 reported | — |
| 10 | Conclusion | — |

**Claim discipline.** `12_CHECKLISTS.md` §D applies to every sentence. The framing under each
Gate-2 branch is written in `13_BUDGET_TIMELINE.md` "Fallback map" — pick the row that matches
the pre-registered outcome, not the most flattering one.

## S7.4 — Release

**Do.** Code, the signature table (`operator/`), the OOD generator and its seeds, the perturbed
tables, the silent-failure injector, the pre-registration note with its commit hash. PPGN set the
precedent for public code in this niche; releasing measurably improves reception.

**Check.** A fresh clone reproduces T6 (training-free rows) from the released table alone.
