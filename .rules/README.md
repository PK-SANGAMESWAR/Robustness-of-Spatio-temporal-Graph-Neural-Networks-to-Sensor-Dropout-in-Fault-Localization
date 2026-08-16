# .rules/ — operating rules for this repo

Standing rules for anyone (human or assistant) doing work here. `CLAUDE.md` is the *map*
(what exists, where); these are the *constraints* (what you may and may not do).

| File | Covers |
|---|---|
| `00-execution.md` | What may be launched on the GPU without asking, and what must stop and report first |
| `10-integrity.md` | Scientific method rules: one variable at a time, frozen splits, never fabricate a number |
| `20-paper.md` | Editing `paper/overleaf.tex`: edit-don't-restructure, page cap, framing |
| `30-reporting.md` | How results must be stated — the claims that are NOT allowed to be upgraded |
| `40-data.md` | The frozen corpus, splits, and bugfix provenance |

**Precedence when sources disagree:** a CSV/JSON in `results/` beats `paper/overleaf.tex`,
which beats any markdown file, which beats memory. Never resolve a conflict by picking the
number that sounds better.

**History:** this folder previously held only `CUDA.md` (a single execution rule, unchanged
since the initial commit). Its intent is carried forward in `00-execution.md`, amended per the
author's decision on 2026-08-16. A separate, richer `paper/.rules/` set was deleted in `6a00c4e`;
the rules from it that still matter were rescued into `20-paper.md` here.
