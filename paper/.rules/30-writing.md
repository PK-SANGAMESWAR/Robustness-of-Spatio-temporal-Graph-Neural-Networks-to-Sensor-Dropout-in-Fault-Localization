# 30 — Writing and LaTeX Conventions

## Voice

Formal academic English. Third person or first-person plural ("we propose"). Past tense
for what was done, present tense for what is true.

Cut on sight: "delve," "leverage," "robust" as filler, "it is important to note that,"
"in today's rapidly evolving landscape," "paradigm shift," "seamlessly," "cutting-edge,"
"game-changing." Reviewers read these as padding.

## Claim strength must match evidence

| Evidence you have | Verb you may use |
|---|---|
| Statistically tested result | demonstrates, shows |
| Consistent empirical trend | indicates, suggests |
| Single run, no variance | we observed |
| Reasoning, not measured | we hypothesize |

Never write "significantly" unless a significance test was run and is reported.

## Structural rules

- One idea per paragraph. Topic sentence first.
- Every section opens by stating what it establishes and closes by stating what follows.
- Define every symbol and acronym at first use, once, and never redefine it.
- Every figure and table is referenced in the prose before it appears.

## Numbers

- Consistent decimal places within a table (F1 to three places throughout, or two — pick
  one and hold it).
- Percentages: state whether a change is absolute (points) or relative.
- Never round differently in two places for the same value.

## LaTeX / Procedia specifics

- Citation style is **Vancouver, numeric**: `\cite{key}` renders as `[1]`, numbered by
  first appearance. Never hand-number references in the prose.
- Use the Procedia CS document class already in `main.tex`. Do not swap classes or add
  packages without asking — Overleaf compile breakage costs more than the edit saves.
- Math in `\begin{equation}` with `\label`; reference via `\eqref` or `\ref`.
- Non-breaking spaces: `Fig.~\ref{...}`, `Section~\ref{...}`, `Karabulut et al.~\cite{...}`.
- Keep the `.bib` alphabetical by key and one field per line — it keeps diffs readable.
- Do not commit `.aux`, `.log`, `.bbl`, or `.synctex.gz`.

## Editing existing prose

- Preserve the author's voice. You are editing a paper, not rewriting it.
- Show edits as minimal diffs with one line of rationale each.
- When you shorten for page limits, report the word count before and after and confirm no
  claim, citation, or result was dropped in the process.

## End-of-task report

Every response that touches the manuscript ends with:

```
Changed:      <files and sections>
Verified:     <claims checked, with sources>
Unverified:   <anything I could not confirm>
Needs input:  <decisions only the author can make>
```