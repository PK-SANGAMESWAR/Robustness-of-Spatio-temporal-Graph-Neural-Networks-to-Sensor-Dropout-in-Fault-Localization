# 50 — Page Limit

## The limit

**10 pages maximum**, in the Procedia CS template, compiled.

Verify against the ICMLDE call for papers before the first trim pass — confirm whether the
10 pages **include** references and appendices, since Elsevier venues differ on this. Do
not assume. Search, then record the answer here:

```
Verified: <date> — <CFP URL> — references included / excluded / not stated
```

## Ordering: draft first, measure second

**The limit is a gate at the end, not a constraint during drafting.**

While drafting, write the complete argument. Say everything the paper needs to say. Do
**not**:

- Shorten a section because you estimate the paper is "getting long."
- Drop a figure, a citation, or an ablation preemptively.
- Compress related work before it is complete.
- Mention the page count in the middle of a writing task.

Cutting a full draft down is a mechanical, reversible edit. Never-written material is
invisible and lost. Under-writing to hit a limit produces a paper that is short *and*
unconvincing, which fails twice.

An over-length first draft is the expected outcome. Treat it as normal.

## The measurement gate

Run this **only** when the full draft exists — all sections written, figures placed,
bibliography complete.

Measure the compiled PDF. Word counts and `.tex` line counts are not page counts.

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdfinfo main.pdf | grep Pages
```

Report: current page count, the limit, the overflow, and **where** the overflow sits —
which sections and floats occupy the excess pages. Overflow of half a page is a different
problem from three pages, and the fix is different.

If the draft is at or under the limit, say so and stop. Do not trim a paper that fits.

## Trim order

When over, cut in this sequence. Exhaust each level before moving down.

1. **Float and layout waste** — an oversized figure, a table that fits at
   `\small`, a figure that could move from double-column to single-column, orphaned
   headings. Usually recovers the most pages for zero content loss.
2. **Redundant figures** — anything a table already conveys, per `40-figures.md`.
3. **Verbose prose** — padding, restated points, over-long transitions, sentences that
   preview what the next sentence says.
4. **Related work compression** — group similar references into one sentence with multiple
   `\cite` keys rather than one sentence each. Keeps the citations, loses the lines.
5. **Merge or shorten the introduction's motivation** — usually the most compressible
   substantive section.

## Never cut, and never do

Off-limits regardless of overflow:

- Any experimental result, including negative or unfavorable ones.
- Reproducibility details: hyperparameters, dataset splits, seed counts, hardware.
- Any claim's supporting citation. Cutting the citation and keeping the claim is worse
  than cutting both.
- Limitations and threats-to-validity discussion.
- Figure caption content — captions must stay standalone.

Formatting hacks that cause desk rejection, so never attempt them:

- Changing font size, margins, line spacing, or the document class.
- `\vspace{-Xpt}` to claw back whitespace around floats or headings.
- Shrinking the bibliography font.
- Moving required content into an unlabeled appendix to dodge the count.

If the paper cannot reach 10 pages without breaking a rule above, that is a **scope
decision for the author** — report it and stop. Do not resolve it by quiet deletion.

## Report after trimming

```
Before:   <N> pages
After:    <M> pages
Cut from: <section — what was removed — trim level used>
Preserved: no result, citation, or reproducibility detail was removed
Still over / now fits
```