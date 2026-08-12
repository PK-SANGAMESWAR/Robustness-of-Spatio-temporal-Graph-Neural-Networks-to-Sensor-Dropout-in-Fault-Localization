# Review & Page-Reduction Notes — SDR-STGNN Paper (ICMLDE 2026)

## Citation check (verified)

Both base-paper citations are real and correctly numbered:

- **Karabulut et al. (2026)** — `arXiv:2604.20403`. Title, authors, and ID all match (v1 22 Apr 2026, v2 8 May 2026). ✓
- **Karabulut et al. (2025)** — `arXiv:2510.03571` ("Generalization of GNN Models for Distribution Grid Fault Detection"). ✓

A reviewer who pulls these will find exactly what you cite.

> **Flag:** The base paper reports RGATv2/RGSAGE as strong with *tight* ±1.4% confidence intervals. Your reproduction has RGATv2 as the **weakest** GNN and the **most unstable** (±0.087). A reviewer familiar with the base paper will notice this divergence.

---

## Verdict

Publishable and solid applied work. For ICMLDE / Procedia CS (Scopus-indexed, mid-tier applied-ML venue), likely outcome: **accept with minor revisions**. Contribution is real and clearly scoped, experiments are honest, and the negative result on SDR-STGNN is reported rather than hidden. Main acceptance risk: a reviewer asking for an imputation baseline. Not top-venue *methodological* novelty, but ICMLDE doesn't require that.

---

## Cutting to ~12 pages

**Root problem:** the "internally consistent / 0.85 reference point / not comparable to base paper's absolute scale" framing appears ~7 times (abstract, intro, Deviations subsection, Table 3, Results intro, and multiple discussion/limitation spots). Cutting it reclaims close to a page **and** removes a defensive tone reviewers read as insecurity.

Prioritized cuts (highest yield first):

1. **Collapse the "internally consistent" refrain** to one statement in Section 3.2 (Deviations) + one pointer in Results. Delete from abstract tail, intro, and Limitations "Dataset scale." — *~0.5–0.75 page, improves tone.*
2. **Delete Table 3 (Deviations).** All content is already in prose. Replace with one sentence: *"Our setup differs from [Karabulut2026] in three documented respects — four message-passing hops rather than one, a 1.65 M-window corpus, and omission of the reconfigured green topology."* — *~0.25 page.*
3. **Merge the spatial-depth material** (currently appears 4×: config paragraph, "only difference" restatement, the depth sweep in Results, and Discussion 6.1). Keep the config paragraph + the sweep numbers; cut the two restatements.
4. **De-duplicate the max-pooling failure-mode explanation** (full version in both 5.2 and 6.1). Keep the detailed version in Discussion; compress Results to one sentence.
5. **Trim the "SDR-STGNN matches but does not exceed" conclusion** from 4 appearances (abstract, 5.3, 6.1, conclusion) to 3. The 6.1 restatement is most cuttable.
6. **Tighten the abstract** from ~260 to ~180–200 words.
7. *(Optional, if still over)* In Table `tab:baseline`, move no-fault precision/recall to prose ("all models reach no-fault recall 1.000, precision ≥0.972") and drop those two columns. Keep the per-resistance columns.

**Lower-yield levers:** Figure `fig:ablation` largely restates Table `tab:aware` — drop one if desperate. The six-item Limitations list can lose ~30% of its words.

**Do not cut:** Table 1 (positioning) — your clearest "here's the gap" argument. The algorithm and SDR-STGNN equations stay.

---

## Scientific review

### Strengths

- **Real, verified gap.** The base paper studies cross-run stability and cross-feeder generalization, not inference-time sensor loss. "First systematic study of STGNN robustness to μPMU dropout at inference" is a fair claim.
- **Strong headline result:** 7% → 81% retention at 40% loss, with ΔF1 < 0.01 clean.
- **Reusable evaluation contribution:** AUDC / retention framing.
- **Right rigor:** multi-seed reporting with shared mask realizations across architectures.
- **Integrity:** SDR-STGNN negative result reported honestly.

### Weaknesses & likely reviewer requests (by probability)

1. **No imputation baseline (most likely revision request).** Zhang 2025 and Fan 2024 are your nearest neighbors and you decline to compare. Even one impute-then-infer point on RGSAGE-Max would largely neutralize this. If you can't add it before the deadline, pre-empt more assertively — current "left to future work" reads as avoidance.
2. **RGATv2 reproduction divergence.** You lean on RGATv2 for the robustness ranking, then caveat it 3×. Foreground the *inversion at the extremes* (RGSAGE-Max most accurate → least robust — your firm result); demote RGATv2's mid-table position to "indicative" once, confidently.
3. **Modest core-technique novelty.** Dropout-aware training is structured input-node masking (adjacent to DropNode/DropEdge). **Dangling reference:** `\bibitem{Rong2020}` (DropEdge) is in the bibliography but never `\cite`d in the body — an editor will flag this. Fix it *and* use it: cite DropEdge/DropNode when introducing dropout-aware training, and state that your novelty is the *problem framing and systematic study*, not the masking primitive.
4. **Single feeder, iid uniform dropout.** Both acknowledged. Keep the correlated-failure limitation prominent (real μPMU losses cluster by substation/comms path).
5. **No code/data release stated.** For a robustness/reproducibility paper, add a "code available on acceptance" line if possible.

---

## Small fixes for a finished look

- Corresponding-author phone/fax still show template placeholder `+0-000-000-0000` — remove or fill.
- Acknowledgements section is currently empty (header only).
- Typos: "The four-spatio temporal GNN models" → "four spatio-temporal"; "For each input windows" → "each input window."
- Stray spaces before tildes, e.g. `Section ~\ref{...}` → `Section~\ref{...}`.
- Em-dash-as-`--` in "training method--dropout-aware" / "architecture--SDR-STGNN" reads oddly; use a colon or spaced em-dash.

---

## Bottom line

Tighten the "internally consistent" refrain and the depth/mechanism repetitions (this alone likely gets you to 12 pages and improves tone), fix the dangling DropEdge citation, and decide whether one imputation baseline is feasible. Do those and this is a confident submission.