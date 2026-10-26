# Thesis page cuts — proposal only

**Written for:** the thesis author (decision on which cuts to make). Proposal only. `MTech_thesis.tex` was not
changed for page cuts. Date: 2026-10-24.

## How the numbers were measured

Each cut was applied to a scratch copy of `MTech_thesis.tex` (pdflatex, two passes) and the end page of the main
text was read from the PDF. "Pages saved" is the fractional length of the chapter before and after the cut. Each
chapter starts on a new page, so a saving counts as a whole page only when it lowers the chapter's page count.
The replacement sentences that a cut needs (about 0.1 page each) are **not** in these numbers. Treat the final
page counts as ±1.

## 1. Pages per chapter (main text, now 64 pages; limit 60; target ≤ 57)

| Chapter | Printed pages | Pages | Content length (fraction of last page used) |
|---|---|---|---|
| 1 Introduction | 1–11 | 11 | 10.20 |
| 2 Literature Survey | 12–19 | 8 | 7.81 |
| 3 Proposed Work | 20–40 | 21 | 20.75 |
| 4 Results and Inferences | 41–60 | 20 | 19.26 |
| 5 Conclusion | 61–64 | 4 | 3.69 |
| **Main text** | 1–64 | **64** | |

The remaining [PENDING] markers need room. Four arms report results (a held-out-term-only model, a model without
the auxiliary loss, a shuffled-table model, a learned-decoder model), and one more (a type-head model) if it is
run; I estimate **1.0 to 1.5 pages**. The wider-backbone model was dropped from the plan, so the two
`[PENDING: R7]` markers should be removed, not filled (two markers). Target ≤ 57 leaves 3 pages for this.

## 2. Version A — appendices do not count (move first, then cut little)

Step 1 is "move tables to an appendix first".

| # | Cut | Where | Pages saved | Main-text pages after | What the reader loses |
|---|---|---|---|---|---|
| A1 | 8 secondary Ch. 4 tables to an appendix: depth diagnostic, loss patterns, significance tests, signature-table fidelity, physics-only ladder, inference-time fusion, graft screening, loss-weight tuning | 4.1, 4.5.1, 4.6, 4.8.1–4.8.5, 4.9.3 | 1.8 | | The exact numbers beside the text; they stay one page-turn away. The text still states the findings. |
| A2 | 5 configuration tables to an appendix: corpus, event-level partition, training configuration, comparison methods, signature-table axes | 3.1.1, 3.1.3, 3.2.3, 3.6, 3.7.1 | 1.55 | | The setup at a glance. Section text must keep the key values. |
| A3 | Table 1.1 (positioning of prior work) and Table 2.1 (families of methods) to an appendix | 1.5, 2.1 | 0.87 | **60** (A1–A3 together: 64 → 60) | The visual comparison with prior work. The gap is still argued in prose (2.8). |
| A4 | Move text of 4.8.5 (graft screening) to an appendix | 4.8.5 | 0.61 | 59 | The screening that rejected loss-injection grafts. One sentence stays in 4.9. |
| A5 | Move text of 4.9.3 (loss-weight selection) to an appendix | 4.9.3 | 0.45 | 59 | How λ₁ = 0 and λ₂ = 0.1 were chosen. One sentence stays. |
| A6 | Move text of 4.8.4 (inference-time fusion, preliminary) to an appendix | 4.8.4 | 0.55 | 58 | The preliminary evidence that fusion helps under noise (+0.213). The Discussion still cites it, so keep one sentence with the number. |
| A7 | Move 3.1.4 (data quality and reference bounds) to an appendix | 3.1.4 | 0.45 | **57** | The unlabelable-window ceiling (0.9744) and settling bounds. |
| A8 (optional margin) | Move 3.7.4 and 3.9 to an appendix | 3.7.4, 3.9 | 0.55 + 0.31 | **56** | The description of the inference-time fusion method (Eq. fusion stays in 3.8); the pipeline-verification paragraph. |

**Version A result: 57 pages with A1–A7, 56 with A8.** Keep the Gate-4 tables (4.14, 4.15) in the main text.
Moving them as well saves 1.13 pages but removes the main result of Chapter 4 from the chapter.
Check first that the `ugthesis` template allows an appendix and that the 60-page limit excludes it.

## 3. Version B — appendices count (shorten; nothing moves)

| # | Cut | Where | Pages saved | Main-text pages after | What the reader loses |
|---|---|---|---|---|---|
| B1 | Delete 1.10 (organization of the thesis) | 1.10 | 0.48 | 63 | A roadmap; the table of contents covers it. |
| B2 | Delete Figure 1.4 (workflow) and Figure 1.3 (stressors); keep the four rules of 1.9 | 1.4, 1.9 | 0.32 + 0.35 | 63 | Two overview pictures. |
| B3 | Delete Figure 2.1 (taxonomy) | 2.1 | 0.21 | 63 | A picture of method families; Table 2.1 stays. |
| B4 | Delete Table 1.1 (prior-work positioning) | 1.5 | 0.46 | 62 | The check-mark comparison. 2.8 states the gap. |
| B5 | Delete 3.9 (verification); fold one sentence into 3.3.1 | 3.9 | 0.30 | 62 | The mask-seed bug history. |
| B6 | Cut 3.6.4 (classical-baseline method) to two sentences | 3.6.4 | 0.65 | 61 | The method detail; results stay in 4.5.4. |
| B7 | Cut 3.7.4 (physics-based imputation and fusion method) to two sentences | 3.7.4 | 0.56 | 60 | The method detail for the inference-time baselines. |
| B8 | Delete 4.8.5 (graft screening) with its table; one sentence in 4.9 | 4.8.5 | 0.69 | 59 | The evidence that loss-injection grafts fail. |
| B9 | Delete 4.9.3 (loss-weight selection) with its table; one sentence | 4.9.3 | 0.73 | 58 | How the final loss weights were chosen. |
| B10 | Delete the text and table of 4.8.4 (preliminary fusion); keep the +0.213 in one sentence | 4.8.4 | 0.82 | 58 | The preliminary-fusion table. |
| B11 | Delete 5.1 (summary of the work) | 5.1 | 0.46 | 58 | A recap that 5.2 repeats. |
| B12 | Replace Table 4.14 (per-seed, 20 conditions) by a 5-row mean table | 4.9.4 | 0.57 | **57** | Per-seed values. The sign-consistency claim then rests on the text. **This is the cut I like least.** |
| B13 (optional) | Shorten 2.7 (physics-aware learning) by two thirds | 2.7 | ≈ 0.6 | 56 | Literature coverage for the physics component. |

**Version B result: 57 pages only with B1–B12, and the replacement sentences add 0.5 to 1 page, so B is likely 57
to 58.** B cannot reach 57 without deleting preliminary-result subsections and condensing the main Chapter-4 table.
If both versions are possible, Version A loses far less.

## 4. My recommendation

Use Version A (A1–A7, with A8 for margin) if the appendix does not count. If it does, the cost is high and I
would ask the supervisor whether 60 is a hard limit for a document that includes an appendix.

## 5. Other things found while measuring

- Two `[PENDING: R7]` markers remain although the wider-backbone model is dropped from the plan. Remove them.
- The figures the author inserted are in the build: the main text is 64 pages with them (it was 64 before).
- Chapter 1 has 11 pages for an introduction (9 sections); B1–B4 alone bring it to 9.
