# 20 — Editing the paper (`paper/overleaf.tex`)

Rules rescued from the deleted `paper/ISSUES.md` and `paper/.rules/`, updated to current state.

## Edit, don't restructure

Make targeted edits. Do **not** perform large structural rewrites of sections that are already
written — bulk-regenerated prose carries AI-plagiarism/similarity risk, and the draft has already
been through review. Rewriting a paragraph to say the same thing differently is not an
improvement; it is a risk.

## Never invent a number

Re-check every quoted result against the CSV/JSON on disk before it goes in the `.tex`.
See `10-integrity.md`.

## Constraints

- **Page cap = 12** (ICMLDE). Currently **13 pp** — trimming is the open task. The deleted
  `.rules/50-pagelimit.md` said 10; that file is gone and 12 is the number to work to.
- **It does not compile locally** — `ecrc.sty` (Elsevier CRC) is not installed on this machine.
  Page count can only be verified on Overleaf. `paper/preview-local.pdf` is the last export.
- Document class is `\documentclass[3p,times,procedia]{elsarticle}`.
- Wide tables must use `table*` (full width) or they overflow the two-column layout.
- Figures must be ≥300 dpi; both embedded PNGs currently export at 400.
- Equation cross-references use `\eqref`.

## Framing — non-negotiable

This paper **adopts** Karabulut et al. (2026)'s methodology and reports **our own** numbers on a
corpus we regenerate ourselves.

- **Never** use "reproduce"/"reproduction" language, and never claim to match their 0.95.
- **Always** keep the attribution to the base paper.
- Our 1.65 M windows vs their reported 2.5 M is a **documented deviation**, not a discrepancy to
  paper over. The 2.5 M figure is not reconstructable from the 825-scenario grid.
- Depth 4 (vs their 1 layer) is likewise a documented deviation — depth 1 was unstable here.

## Framing of the contribution

Dropout-aware training is the **headline**. SDR-STGNN is an **honest ablation**. Do not reorder
that emphasis, and see `30-reporting.md` for what may not be claimed.
