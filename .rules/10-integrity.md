# 10 — Scientific integrity

## Never fabricate a number

Every number that appears in the paper, a doc, a slide, or a message must be traceable to a
CSV/JSON on disk. If you cannot point at the file it came from, it does not go in. This includes
"about", "roughly", and numbers recalled from earlier in a conversation — re-read the file.

Several deleted markdown docs quoted **pre-bugfix** values. Any number inherited from a document
rather than from `results/` is suspect until re-checked.

## Precedence

`results/*.{csv,json}` → `paper/overleaf.tex` → markdown docs → memory.
Resolve every disagreement downward, never upward.

## Change one variable at a time

Identical frozen data and splits across all models. If you change the corpus, do not also change
the architecture; if you change depth, do not also change the epoch budget. A result that moved
for two reasons at once measures nothing.

## Tests are not validation

Code passing tests ≠ validated. A masking policy, evaluator, or metric is only confirmed by a
**real run** against known-good numbers. The standing sanity check for any dropout-style
evaluation: **F1 at k=0 must equal the recorded clean F1** for that checkpoint. If it doesn't,
the loader or checkpoint path is wrong — stop and fix it before reporting anything.

## Fairness of paired comparisons

Any comparison between dropout policies is valid only if the **same sensors are dropped**.
When adding a policy, verify the dropped-node sets match the reference policy byte-for-byte
across every `(k, offset)` before reporting a comparison — this is how the imputation baseline
was validated (`IMPUTATION.md` §4).

## Statistics discipline at n=3

Three training seeds is a floor, not a comfort. Report paired t-tests with **Holm correction per
family** and 95 % CIs. Wilcoxon is computed but must **not** be relied on — its p-value floor at
n=3 is 0.25. Which specific claims survive and which do not is fixed in `30-reporting.md`;
do not re-litigate them from a fresh read of the numbers.
