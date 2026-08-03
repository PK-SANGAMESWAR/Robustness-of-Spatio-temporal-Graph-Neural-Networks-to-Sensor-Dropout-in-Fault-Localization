# 10 — Reading the Repository

## Read fully, not partially

`grep` tells you a string exists. It does not tell you what the section argues, what the
notation already is, or whether the claim you're about to write is already made two
paragraphs up. **Open and read the whole file.**

Mandatory full reads before editing any section:

- The `.tex` file you are editing — start to finish.
- `overleaf.tex` (or the root file) — for `\input` order, packages, macros, custom commands.
- The `.bib` file — before adding, citing, or renumbering any reference.[.bib here is written inside the overleaf.text file]
- Any results `.csv` / `.json` / log file you are about to quote a number from.

## Session start checklist

Before your first edit in a session, produce this:

1. **File map** — every `.tex`, `.bib`, figure, and results file, with a one-line purpose.
2. **Notation table** — every symbol already in use and its meaning. Reuse these. Never
   introduce a second symbol for a quantity that already has one.
3. **Claim inventory** — for the section in scope: each factual claim, and where its
   support lives (repo file / reference key / unsupported).
4. **Open gaps** — every `[TODO]`, `\todo{}`, `XXX`, commented-out block, and any number
   that appears in two places with two different values.

Show this before writing. It is cheap and it prevents contradictions.

## Numbers are traced, never recalled

When a metric enters the prose:

- Find it in a results file or log. Quote the file path.
- Check it against every other place the same metric appears in the manuscript.
- If two places disagree, **stop and report the conflict.** Do not silently pick one.

Metrics that need special care in this paper: accuracy, precision, recall, **F1**,
localization error, dropout rate, bus counts, and any baseline number attributed to
Karabulut et al.

## Reporting reads

State what you read and what you did not. "I read Sections 1–3 and the `.bib`; I did not
open the appendix" is useful. Silence implies you read everything, which is a lie if you
didn't.