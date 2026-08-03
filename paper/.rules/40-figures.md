# 40 — Figures

## Where figures live

Generated figures are **scattered across the project**, not centralized. At minimum:

- `figures/` — hand-made and finalized plots
- `results/` — figures emitted by training and evaluation runs, often alongside the
  `.csv` / `.json` that produced them

Assume there are more. Before any figure work, sweep the whole repo:

```bash
find . -type f \( -iname '*.png' -o -iname '*.pdf' -o -iname '*.svg' \
  -o -iname '*.eps' -o -iname '*.jpg' \) -not -path './.git/*' | sort
```

Do not assume `figures/` is complete. A plot sitting in `results/dropout_sweep/` is just
as likely to be the one the paper needs.

## Inventory before selecting

Produce this table before proposing anything:

| Path | What it shows | Source script / notebook | Date | Already in paper? |
|---|---|---|---|---|

Flag in the inventory:

- **Duplicates** — the same plot saved twice under different names or folders.
- **Stale versions** — an older run's figure whose numbers no longer match the results
  tables. Check the timestamp against the results file, and check the values in the plot
  against the reported metrics.
- **Orphans** — a figure with no identifiable generating script.

A stale figure is more dangerous than a missing one, because it looks finished.

## Selection: inclusion is earned, not automatic

**There is no requirement to include every figure.** Most generated plots are diagnostic
artifacts for the authors, not evidence for the reader. Procedia page limits are real, and
a padded figure set reads as an author who could not identify their own contribution.

Include a figure only if it clears **all** of these:

1. It supports a claim the paper actually makes.
2. It communicates something prose or a table cannot do as compactly.
3. It is readable at single-column width in print, in greyscale.
4. Its numbers agree with the results tables.

For this paper, the figures that usually earn their place are:

- The system / architecture diagram (one, not several).
- The IEEE 123-bus feeder topology with sensor placement.
- The headline result: localization performance vs. **sensor dropout rate** — this is the
  contribution, so it gets the most space.
- One comparison against the Karabulut et al. baseline.

Figures that usually do **not** earn their place: training-loss curves (unless
convergence is itself a claim), per-epoch metric traces, every ablation as its own plot
when a table does it better, and any variant plot that repeats a point already made.

## Propose, don't insert

Never add figures to the manuscript unilaterally. Present:

```
Recommend including: <path> — <the claim it supports> — <where it goes>
Recommend excluding: <path> — <why it doesn't earn space>
Uncertain:           <path> — <what I'd need to decide>
```

Wait for the author's call. Cutting a figure the author cares about is worse than asking.

## Never fabricate a figure

Do not `\includegraphics` a path that does not exist on disk. Do not describe in a caption
what a plot "will show." If a needed figure is missing, say so and name the results file
that could generate it.

## LaTeX conventions

- Set `\graphicspath{{figures/}{results/}}` in the preamble rather than writing long
  relative paths at each call site.
- Prefer vector (`.pdf`, `.eps`) over raster. If only `.png` exists, check it is ≥300 dpi
  at print size; flag it if not.
- Procedia CS: figure captions go **below** the figure, table captions **above** the table.
- Every figure gets a `\label` and is referenced in the prose as `Fig.~\ref{...}` **before**
  it appears. An unreferenced figure is a defect.
- Use `[htbp]` placement and let LaTeX float. Do not fight the placement with `[H]`.

## Captions

A caption is a standalone sentence, not a title. `Localization F1 across sensor dropout
rates on the IEEE 123-bus feeder; shaded region denotes ±1 std. over five seeds.` — not
`F1 vs dropout`.

Define every axis, unit, and abbreviation in the caption. State the number of runs and
what any error band represents. If you don't know the seed count, flag it rather than
writing a plausible one.