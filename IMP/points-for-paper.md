## Following the protocol of [Karabulut2026], we generate a dataset of the full factorial 25×11×3 = 825 fault scenarios with N load draws each (N=50 in our frozen configuration), yielding 1.65M windows. Note this differs from the 2.5M windows reported in the base paper: their figure is not reconstructable from the stated factor grid (825 scenarios × 100 runs × 40 windows = 3.3M), so we report our own exact count rather than forcing a match.

## ✅ Approach: "we reproduced the protocol and built our own baseline; all robustness results are relative to that baseline.
## ✅ Cite the paper's numbers for motivation, not as your success criterion.
## ✅ One-line honest disclosure of the size difference + depth deviation in Reproduction Notes.
## ❌ Don't claim numerical replication of their 94.7%.


## 
All your numeric comparisons stay internal — your 5 models against each other, and standard vs. dropout-aware, across k, on identical data/splits. You never put your F1 in a table next to their 94.7% as if competing. ✅

"Their way of creating the dataset" = you followed their protocol, not that you got an identical dataset. Word it as "we regenerate the dataset following the base paper's specification" and disclose that the resulting size differs (1.65M vs their reported 2.5M). Don't say "we used the same dataset" — you can't, it's synthetic and you regenerated it. "Same protocol, our own regeneration" is the honest and accurate claim.

You still report the paper's numbers once — for motivation only (the intro paragraph you already have). That's citing, not comparing. Keep it there; just don't repeat them in your results tables as a baseline row.

