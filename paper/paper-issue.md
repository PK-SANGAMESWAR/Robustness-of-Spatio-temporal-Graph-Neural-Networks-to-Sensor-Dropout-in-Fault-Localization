# ICMLDE 2026 — SDR-STGNN Manuscript Revision Checklist
**Deadline: 15 August 2026 | Target: ≤12 pages (Procedia two-column, incl. references & figures)**
**Current: ≈15.5 pages | After all cuts: ≈12.2 pages | +0.2 pp from spacing/figure cropping → within budget**

---

## 0. Overall Assessment (context — no action)

- Three clearly differentiated, internally consistent contributions:
  1. First systematic sensor-dropout robustness study for STGNN-based fault localisation.
  2. Dropout-aware training as a simple, architecture-agnostic mitigation.
  3. SDR-STGNN ablation that cleanly isolates training-time vs. architectural effects.
- Multi-seed, shared-mask evaluation protocol is a particular strength (ensures paired comparisons).
- Headline findings are both statistically meaningful and practically significant:
  - AUDC: **0.414 → 0.916**
  - Retention at k=10: **7% → 81%**
- Only problem: length. ICMLDE CFP caps regular papers at **12 pages including references and figures**.
- Estimated saving from all recommended cuts: **3.5–4 pages**, with **no primary result sacrificed**.

---

## 1. Abstract

**Status:** Structure is clean — problem → gap → method → result → conclusion. Correctly motivates partial observability, positions the gap (no prior STGNN robustness study), quantifies the headline result.

### Action items
- [ ] **A1. Remove redundant µPMU definition duplication.**
  Line 3 currently reads: *"…yet it remains difficult under partial observability, where only a limited number of micro-phasor measurement units (µPMUs) monitor networks of hundreds of buses."*
  The parenthetical definition is repeated **verbatim** in the Introduction.
  → **Keep the definition in the Abstract**; shorten the Introduction's first mention to a back-reference: *"…micro-phasor measurement units (µPMUs; see Abstract)"*, or simply rely on the Abstract's definition.
- [ ] **A2. Quantify the clean-accuracy cost explicitly.** Add one number:
  > "…at virtually no cost to clean performance (ΔF1 < 0.01 at k=0)."
  This is a key selling point reviewers may miss if it stays buried in Table 6.
- [ ] **A3. Fix the SDR-STGNN acronym introduction.** Do **not** introduce the acronym in the abstract if it is not used again there. Either:
  - drop the parenthetical "(SDR-STGNN)", **or**
  - use it in the very next sentence.

---

## 2. Introduction

### Action items
- [ ] **I1. Contribution bullet 1 (dataset + baselines) is a replication step, not a novel contribution.**
  → Re-label it as **"Experimental Infrastructure"** or integrate it into the setup description in Section 3.
  *Rationale: reviewers count bullets; four dilutes the three genuinely novel ones.*
- [ ] **I2. Trim background paragraphs 2 and 3 (~180 words).**
  Paragraph 2 ("Graph neural networks are a natural fit…") and Paragraph 3 (describing [1]) both belong in Section 2.
  → Move the summary of [1] to the **Related Work Table 1 row**; trim this paragraph to **two sentences**.
- [ ] **I3. Remove the five-sentence description of [1]'s architecture** ("…GRU for learning temporal patterns … 94.7–94.9"). It duplicates Table 1 and Section 3.2.
  → Replace with a single sentence plus forward reference:
  > "Building on the STGNN framework of [1] (see Section 3.2 and Table 1), we extend …"
- [ ] **I4. Collapse the roadmap paragraph** ("The remainder of this paper…", ~60 words — standard venue boilerplate). One sentence suffices, or omit entirely. Suggested:
  > "Section 2 reviews related work; Sections 3–4 describe the methodology and setup; Sections 5–6 present results; Sections 7–8 discuss and conclude."

**Estimated saving (I3 + I4): ~0.5 pp**

---

## 3. Related Work

**Status:** The seven-column table gives reviewers an at-a-glance comparison that clearly shows the unique combination of {sparse obs. + dropout @ test + mitigation} that only this work addresses. **Keep it.**

### Action items
- [ ] **R1. Add an interpretive paragraph below the table (2–3 sentences).** Currently nothing explains *why* existing methods fall short beyond what the checkmarks convey. Suggested text:
  > "The table reveals that no prior method simultaneously addresses sparse observability and test-time sensor dropout with active mitigation. Imputation approaches [5, 6] assume a fixed, known missingness pattern incompatible with random inference-time failures. This gap motivates the evaluation and mitigation study presented here."
- [ ] **R2. Resolve uncited reference [2]** (arXiv:2510.03571) — it appears in the reference list but is **not cited in the body text**. Either:
  - cite it explicitly in Section 2 as a companion generalisation study, **or**
  - remove it from the reference list to save space.
  *(See also §10 References.)*
- [ ] **R3. Verify the novelty claim before submission.** The paper claims: *"to the best of our knowledge, no previous work has studied how STGNN-based fault localization is affected when some sensors fail during inference."* This is a strong claim.
  → Run a brief search of **2024–2025 literature** on:
  - missing-sensor robustness in power systems
  - IEC 61850 communication-failure studies
  → Confirm the claim holds, or soften the wording.

---

## 4. Materials and Methods

**Status:** Dataset generation, the measured-only graph formulation, and the dropout-aware training algorithm (Algorithm 1) are described with sufficient reproducibility detail. The six-dimensional node feature including the availability mask is a technically elegant design choice, clearly motivated.

### 4.1 Dataset / Section 3.1

- [ ] **M1. Trim the voltage-response paragraph accompanying Fig. 2.** It restates the obvious ("low-impedance faults produce pronounced voltage drops").
  → **Retain the figure**, cut the paragraph to one sentence:
  > "Fig. 2 illustrates that high-impedance (10 Ω) faults produce shallow voltage sags (<0.02 p.u.), making them the hardest class to localise."
  **Saving: ~0.10 pp**
- [ ] **M2. Collapse Table 2's window-level distribution sub-table.** The train/val/test count rows consume vertical space disproportionate to their informational value.
  → Replace with one inline sentence and **delete the lower half of Table 2**:
  > "The 1.65M windows are partitioned 70/15/15 at the event level (train: 1,155k; val: 248k; test: 247k)."
  **Saving: ~0.15 pp**

### 4.2 Baselines / Section 3.2

- [ ] **M3. Condense the five baseline descriptions (RGCN, RGSAGE-Mean/Max, RGATv2, GRU) into a single paragraph.** They faithfully recapitulate [1], but this paper's contribution is robustness evaluation, not architecture design. The operators are standard and cited, so reproducibility is preserved. Suggested text:
  > "We adopt the five STGNN variants of [1]: a temporal-only GRU, RGCN [3], RGSAGE-Mean, RGSAGE-Max, and RGATv2 [4]. All share a GRU temporal encoder (128 hidden units) followed by an optional spatial stage over the 25-node measured-only graph; a shared classification head and soft voting produce the graph-level prediction. We extend the spatial depth to four hops (Section 5) and otherwise adopt the architectures unchanged."
  **Saving: ~0.4 pp**

### 4.3 Metrics — AUDC

- [ ] **M4. Fix the AUDC discretisation gap.** The continuous integral form,
```
  AUDC = 1/(k_max − k_min) ∫_{k_min}^{k_max} R(k) dk
```
  implies a continuous curve, but k is discrete.
  → The sentence *"In practice we evaluate AUDC via trapezoidal integration over k ∈ {0,1,2,5,10}"* already exists but appears **three lines after** the integral. **Move it to immediately after the equation.**

### 4.4 Equations / Section 3.5

- [ ] **M5. Standardise equation cross-references.** Eq. (1) defines *aᵢ*; Eqs. (2) and (3) define gated aggregation. The text currently references **"Eq. 3" and "Eq. 3.5" interchangeably**.
  → Use `\eqref` consistently throughout.
  → Verify the equation counter resets/increments correctly in the Procedia template.

---

## 5. Experimental Setup (Section 4)

- [ ] **E1. Move exact software versions to a footnote or Reproducibility Statement.** The hardware paragraph (NVIDIA T1000, Python 3.12, PyTorch 2.5.1, CUDA 12.1, PyG 2.8, NumPy 2.5, scikit-learn 1.9, OpenDSSDirect 0.9.4) is important for reproducibility but reads like a version-number list. Some venues now prefer a Reproducibility Statement at the end.
  → Retain in-body only:
  > "All experiments were run on a single NVIDIA T1000 GPU (8 GB) using PyTorch 2.5.1 with PyTorch Geometric 2.8."
- [ ] **E2. Delete the duplicate "Dropout-aware training" paragraph in Section 4.** It repeats Section 3.4. A cross-reference suffices:
  > "Dropout-aware training follows Section 3.4 with no additional hyperparameter tuning."

**Estimated saving (E1 + E2): ~0.25 pp**

---

## 6. Differences from Base Paper (Section 5) — **BIGGEST SINGLE SAVING**

**Status:** This section pre-empts reviewer questions about why the reported F1 (0.85) differs from [1]'s (0.947). Legitimate concern, but it does **not** warrant a dedicated section in a 12-page paper.

### Action items
- [ ] **D1. Fold the section into Section 3 or Section 4.** Add a clearly marked paragraph titled **"Deviations from [1]"** at the end of Section 3.2 (baselines), containing a compact table:

  | Aspect | [1] | This work |
  |---|---|---|
  | Spatial depth | 1 hop | 4 hops |
  | Training loss | Not disclosed | Class-weighted CE → Unweighted CE ⚠️ |
  | Corpus size | Not disclosed | 1.65M windows (50 runs/scenario) |
  | Green topology | Evaluated | Not evaluated |

  ⚠️ **Verify the "Training loss" row against your own manuscript** — the source review renders this row ambiguously (class-weighted CE vs. unweighted CE). Confirm which loss [1] used and which you used before finalising.
  ⚠️ Also confirm the "Corpus size" entry for [1]; the source review does not state it explicitly.

- [ ] **D2. Follow the table with a single sentence:**
  > "All comparisons in this work are conducted on our own corpus and splits, so robustness effects are internally consistent regardless of absolute F1 differences from the base paper."
- [ ] **D3. Delete the long prose paragraphs justifying each difference** (Spatial depth, Training Loss, Corpus size, Scope not covered) — ~60 lines replaced by <10 lines of table.

**Estimated saving: 0.9–1.0 pp**

---

## 7. Results (Section 6)

**Status:** The three-stage structure (baseline accuracy → degradation curves → mitigation comparison) follows a logical progression. Tables 3, 5, and 6 are clean, informative, and well-captioned. Figures 3 and 4 are publication-quality and directly support the key claims.

### 7.1 Section 6.1

- [ ] **S1. Demote Table 4 (depth sweep).** It is described as "diagnostic" but given nearly the same visual weight as Table 3.
  → Move to an appendix (if the venue allows), **or** shrink to an inline sentence:
  > "A single-seed depth sweep on RGSAGE-Max shows macro-F1 increasing monotonically from 0.509 (1 hop) to 0.846 (4 hops, our operating point)."
  **Saving: ~0.15 pp**
- [ ] **S2. Address RGATv2 training instability (±0.087).** Currently only cautioned against, never investigated. Add one sentence to pre-empt the inevitable reviewer question:
  > "Stabilising RGATv2 at depth 4 (e.g., via attention dropout tuning or warmup scheduling) is left as future work."

### 7.2 Section 6.2

- [ ] **S3. Replace the number-reciting paragraph with interpretation.** The paragraph beginning *"RGSAGE-Max, the strongest model under full observability, is also the least robust…"* re-states every number already in Table 5. Suggested replacement:
  > "Table 5 reveals a clear inversion: accuracy rank (RGSAGE-Max > RGSAGE-Mean > RGCN > RGATv2 > GRU) is exactly reversed by robustness rank (RGATv2 ≈ RGCN > RGSAGE-Mean > GRU > RGSAGE-Max). No baseline retains more than 32% of its full-observability F1 at 40% sensor loss. The mechanism is the aggregation operator: max-pooling allows a single corrupted (zeroed) neighbour embedding to dominate a live node's update, while mean and normalised-sum dilute it. This motivates both mitigations in Section 6.3."
  **Saving: ~0.25 pp**

### 7.3 Table 6 — per-resistance breakdown (**reviewer-rejection risk**)

- [ ] **S4. Add per-resistance robustness data.** Table 6 currently reports only aggregate macro-F1. Since Table 3 shows 10 Ω high-impedance faults are the hardest class (F1 drops 0.949 → 0.676 as resistance increases), reviewers will ask: *does dropout-aware training help equally across fault resistances, or does it fail first on high-impedance faults?*
  → Add a single per-resistance row to Table 6, **or** state explicitly:
  > "Dropout-aware training preserves per-resistance ranking: at k=10 the 10 Ω class retains X% of its full-sensor F1, compared with Y% for the 0.1 Ω class (see supplementary Table S1)."
  → **Even a qualitative statement prevents a reviewer rejection on this point.**

---

## 8. Discussion (Section 7)

**Status:** Section 7.1 ("Which Architectures Are Most Robust to Sensor Loss") re-explains, in prose, results already visible in Tables 5 and 6 and Figures 3 and 4.

### Action items
- [ ] **G1. Keep the mechanistic paragraph.** The paragraph beginning *"This ordering follows from how each operator treats the zeroed embedding…"* is insightful mechanistic reasoning and **must be kept**. Cut everything else that re-narrates numbers.
- [ ] **G2. Restructure trimmed Section 7 as:**
  1. Two-sentence summary of the accuracy-vs-robustness inversion (keeping the mechanistic explanation of max-pooling fragility).
  2. One paragraph on **why dropout-aware training works** — currently split across 7.1 and the caption of Fig. 4; **unify here**.
  3. One paragraph on the SDR-STGNN finding (gating removes multi-hop rerouting paths, hence slightly lower AUDC than ungated) — already well-stated in Section 6.3, so **do not repeat it**.
- [ ] **G3. Keep Section 7.2 (Limitations) as-is** — complete and well-scoped.

**Estimated saving: 0.5–0.7 pp**

- [ ] **G4. Add a practical deployment guideline (two sentences).** The paper shows robustness to 40% sensor loss; this has direct practical implication for µPMU network operators (*what minimum sensor availability guarantees acceptable service?*). Suggested text:
  > "For the IEEE 123-bus feeder with 25 µPMUs, dropout-aware training maintains macro-F1 > 0.67 even at 40% loss (k=10). This suggests that operators can tolerate losing up to ∼10 of 25 sensors during scheduled maintenance without degrading the localisation system below an operationally useful threshold, provided the model is retrained with dropout-aware augmentation."
  This connects the academic result to **SDG 7 (Affordable and Clean Energy)** and **SDG 11 (Sustainable Cities)** and strengthens the impact statement.

---

## 9. Conclusion and Future Work

**Status:** Correctly identifies the three contributions, quantifies the headline result, and proposes four concrete future directions (correlated failures, larger feeders, cross-topology generalisation, adaptive gating). **No content changes needed.**

### Action items
- [ ] **C1. Retain the opening** — "Building on the STGNN fault-localization methodology of [1]" is correct.
- [ ] **C2. Fix the citation style to match the body text.** Use `\cite{Karabulut2026}` rather than a manual "[1]" in the final manuscript, so the bibliography auto-numbers correctly in the Procedia template.

---

## 10. References

- [ ] **F1. Cite or remove reference [2]** (Karabulut et al., arXiv:2510.03571, *"Generalization of GNN models for distribution grid fault detection"*) — currently listed but never cited. Options:
  - Cite in Section 2 as the companion generalisation study:
    > "Karabulut et al. [2] separately study cross-feeder generalisation; our work is complementary, addressing sensor dropout robustness on a fixed topology."
  - **Or** remove it to save one reference slot. (Procedia does not enforce a minimum, but an uncited reference raises reviewer eyebrows.)
- [ ] **F2. Verify preprint status of [1] and [2].** Both are arXiv preprints. Before camera-ready, confirm whether [1] has appeared in a peer-reviewed venue (IEEE Transactions on Smart Grid, etc.) and update the entry. Conference proceedings rank preprint citations lower.

---

## 11. Figures and Tables

**Status:** Figs. 3 and 4 use consistent colours, labelled axes, error bars (±1 std), and clear legends. The non-linear x-axis (k ∈ {0,1,2,5,10}) correctly emphasises low-dropout behaviour where differences are most interpretable. **No changes needed.**

### Action items
- [ ] **V1. Re-export Fig. 1 (measured-only graph) at ≥300 dpi as PDF or EPS** (not PNG) for the Procedia camera-ready, to avoid pixelation in print.
- [ ] **V2. Verify Fig. 1's caption against the actual figure.** Caption states *"Red rings mark the 6 µPMU buses that are also candidate fault locations"* — confirm this matches; some red rings may be partially obscured by node labels in the submitted PDF.
- [ ] **V3. (Optional but recommended) Add an architecture schematic.** One column width, ≤6 cm tall, showing the **GRU → GNN → classifier → soft-vote** pipeline, with the dropout-aware masking step annotated on the input. Helps readers coming from a power-systems rather than a GNN background; strengthens accessibility.
  ⚠️ *Note: this ADDS space — weigh against the page budget; only include if you land comfortably under 12 pages.*

---

## 12. Page-Saving Roadmap

| # | Action | Est. saving |
|---|---|---|
| 1 | §1: Remove base-paper architecture recap + shorten roadmap | ~0.50 pp |
| 2 | §3.1: Collapse Table 2 split sub-table to one sentence | ~0.15 pp |
| 3 | §3.1: Trim voltage-sag paragraph to one sentence | ~0.10 pp |
| 4 | §3.2: Condense 5-baseline descriptions to one paragraph | ~0.40 pp |
| 5 | §4: Remove duplicate dropout-aware paragraph; footnote versions | ~0.25 pp |
| 6 | §5: Replace with compact deviation table (**biggest single saving**) | ~0.90 pp |
| 7 | §6.1: Demote Table 4 to inline sentence | ~0.15 pp |
| 8 | §6.2: Replace number-recitation with interpretive paragraph | ~0.25 pp |
| 9 | §7.1: Trim to mechanistic explanation only | ~0.55 pp |
| | **Total estimated saving** | **~3.25 pp** |

- Starting from **≈15.5 pages**, these cuts bring the manuscript to **≈12.2 pages**.
- A further **~0.2 pp** is easily achieved by reducing inter-paragraph spacing and cropping all figures tightly — the Procedia template handles this automatically if bounding boxes are set correctly.

---

## 13. Priority Action List (do these first)

1. [ ] **Trim Section 5 (Differences from Base Paper) to a compact four-row table** — single largest page saving. *(→ D1, D2, D3)*
2. [ ] **Add per-resistance robustness data for the dropout-aware model** (Table 6 extension or supplementary) — closes an obvious reviewer gap. *(→ S4)*
3. [ ] **Cite or remove Reference [2] (Karabulut2025).** *(→ R2, F1)*
4. [ ] **Condense Section 3.2 baseline descriptions to one paragraph.** *(→ M3)*
5. [ ] **Add practical deployment implication paragraph in Section 7** (connects to SDG 7/11, strengthens impact). *(→ G4)*
6. [ ] **Verify arXiv references [1] and [2] for peer-reviewed publication status.** *(→ F2)*
7. [ ] **Export Fig. 1 at ≥300 dpi as vector PDF.** *(→ V1)*
8. [ ] **Fix equation cross-reference style in Section 3.5 (use `\eqref` consistently).** *(→ M5)*

---

## 14. Master Checklist (all items, flat)

**Abstract:** A1 µPMU dedup · A2 ΔF1 < 0.01 at k=0 · A3 SDR-STGNN acronym
**Introduction:** I1 relabel bullet 1 · I2 trim paras 2–3 · I3 cut [1] architecture recap · I4 collapse roadmap
**Related Work:** R1 interpretive paragraph · R2 uncited [2] · R3 verify novelty claim (2024–25 lit + IEC 61850)
**Methods:** M1 voltage-sag paragraph · M2 Table 2 sub-table · M3 baseline condensation · M4 AUDC trapezoid sentence placement · M5 `\eqref` standardisation + counter check
**Setup:** E1 versions → footnote · E2 delete duplicate dropout paragraph
**Deviations:** D1 fold + table · D2 internal-consistency sentence · D3 delete prose justifications
**Results:** S1 demote Table 4 · S2 RGATv2 future-work sentence · S3 interpretive replacement · S4 per-resistance data
**Discussion:** G1 keep mechanism · G2 restructure 7.1 · G3 keep Limitations · G4 deployment guideline + SDG
**Conclusion:** C1 keep opening · C2 `\cite{}` not manual [1]
**References:** F1 cite/remove [2] · F2 preprint status check
**Figures:** V1 Fig. 1 vector 300 dpi · V2 caption/red-rings verification · V3 optional pipeline schematic