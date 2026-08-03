# Project: ICMLDE 2026 Paper — STGNN Fault Localization under Sensor Dropout

This repository is an **academic paper**, not a software product. The deliverable is a
camera-ready Elsevier *Procedia Computer Science* manuscript. Correctness of claims
matters more than speed of output.

## Read these before doing anything

- `.rules/00-core.md` — non-negotiables. Read every session.
- `.rules/10-reading.md` — how to read the repo before editing.
- `.rules/20-research.md` — web search + citation verification protocol.
- `.rules/30-writing.md` — prose, LaTeX, and Procedia/Vancouver conventions.
- `.rules/40-figures.md` — finding, selecting, and captioning figures.
- `.rules/50-page-limit.md` — the 10-page limit. **Applies after the full draft, not during.**

## Paper at a glance

| Field | Value |
|---|---|
| Venue | ICMLDE (Elsevier Procedia Computer Science) |
| Citation style | Vancouver, numeric, square brackets, ordered by first appearance |
| Page limit | 10 pages compiled — enforced only at the final gate, see `50-page-limit.md` |
| Template | `elsarticle` / Procedia CS `ecrc` class |
| Topic | Spatio-temporal GNN fault localization on power distribution feeders |
| Testbed | IEEE 123-bus feeder |
| Core contribution | Robustness of STGNN fault localization under sensor dropout |
| Baseline extended | Karabulut et al. benchmark |
| Editing surface | Overleaf (assume a human edits in parallel) |

## Default workflow for any task

1. Read the relevant `.tex` files **in full** — not with `grep`.
2. Restate the task and what you found, before writing anything.
3. Do the work in the smallest possible diff.
4. Report what you changed, what you verified, and what you could **not** verify.