# 00 — Core Rules (non-negotiable)

## The fabrication rule

**Never invent a number, citation, author, year, DOI, dataset statistic, or experimental
result.** Not as a placeholder, not as an illustration, not "to be replaced later."

If a value is needed and not found in the repo:

- Write `[TODO: value not found in repo — needs author input]` in the text.
- Add the item to a `NEEDS-VERIFICATION.md` list at the end of your reply.

A fabricated F1 score that survives into submission is a retraction risk. A `[TODO]`
marker is a two-minute fix. Always choose the `[TODO]`.

## The verification rule

Every factual claim in the manuscript falls into exactly one bucket:

| Bucket | Source | What you must do |
|---|---|---|
| Our result | Repo logs, results tables, notebooks | Cite the file and line you read it from |
| Prior work | A real paper | Verify it exists via web search before citing |
| Common knowledge | Textbook-level | State it plainly, no citation needed |

If a claim doesn't fit any bucket, it does not belong in the paper.

## Scope discipline

- Change **only** what was asked. Do not "improve" adjacent paragraphs, retitle sections,
  reorder the bibliography, or reformat untouched files.
- Never delete a `\cite`, a figure, a table, or a paragraph without asking first.
- Never reflow or rewrap lines you didn't otherwise edit — it destroys the Overleaf diff.

## Uncertainty is reported, not smoothed over

Say "I could not verify this" plainly. Do not hedge a claim into vagueness to make it
technically defensible. Vague prose is worse than a flagged gap.

## Parallel editing

A human is editing the same files in Overleaf. Before a large edit, re-read the file from
disk rather than trusting an earlier read in the same session.