# QUESTIONS.md — author's open questions + answers

**Q1: ✅ RESOLVED (2026-08-13).** The 3-seed requirement below was met in full: all 5 baselines,
all 5 dropout-aware variants, and SDR-STGNN are trained at seeds **42/43/44**, and *every*
degradation curve is 3 training seeds × 3 mask seeds. The paper reports mean ± std throughout,
plus paired t-tests with Holm correction (`results/statistics/`). Nothing was shipped
single-seed.

---

1. Next doubt is currently we are runing all models on 1 seed for 100 full dataset , so if i had to write a conf is this comparison of all seed with my new contibution enough or had i to test with all thew seeds mentioned ? 

Answered like : 
-> Short answer: one seed is enough to build and demonstrate the story, but not enough for the final submission — you need ≥3 training seeds with mean ± std on your headline results, especially the contribution. Here's the reasoning.

Two different sources of randomness — don't confuse them
Training-seed variance — different weight init + data-shuffle produces a different model. You currently have only seed 42 for this. ← the gap.
Mask-seed variance — which k μPMUs get dropped. Your dropout evaluator already runs 3 mask seeds per k (n_mask_seeds=3), so this axis is covered.
You have (2) but not (1). A conference paper needs both.

Why one training seed isn't enough for your specific paper
Your whole theme is robustness, and your key novel claim is quantitative: "dropout-aware training recovers X% of the accuracy lost to sensor dropout." With a single training seed you cannot tell whether that recovery is real or just training-seed noise. If seed 42's dropout-aware model happens to beat seed 42's standard model by 2 points, a reviewer will ask: "is that within the noise of re-running with a different seed?" — and you'd have no answer. The base paper itself reports error bars over 5 seeds (their Fig. 4: 94.7 ± 0.3, GRU 86.3 ± 4.2), so that's the bar you're being compared against.

What's actually required (conference standard)
Minimum defensible: 3 training seeds (42, 43, 44), reported as mean ± std, on your main results:
the baseline table,
the degradation curves (F1 vs k),
the standard-vs-dropout-aware comparison.
You do not need 3 seeds on every diagnostic/ablation — just the headline claims.
Where this already fits your plan
This is exactly the seed-42-first / backfill-later split your plan encodes:

Now (Step 2 + Step 4): seed 42 only → gets you a complete draft with the full mechanism and figures. Perfect for writing and internal review.
Step 5 (backfill): re-run seeds 43 + 44 for the baseline and the dropout study (4a + 4b), then recompute everything as mean ± std → the error bars the paper actually ships with.
My recommendation, given the deadline
Finish the seed-42 pipeline end to end first (baseline → degradation → dropout-aware). That's your story, and it de-risks everything.
Then prioritize multi-seed on the contribution (degradation + dropout-aware recovery) over exhaustive multi-seed everywhere — because that's the claim reviewers will scrutinize hardest.
Report only what you actually ran — if 43/44 don't finish in time, present single-seed results labeled as such (a known limitation) rather than implying error bars you don't have. But plan to get to 3.
Bottom line: single seed = enough to write the draft and prove the mechanism; 3 seeds with mean ± std = what the final conference version needs, and it's non-negotiable specifically for the dropout-aware recovery number. It's already Step 5 in your plan — just don't let it get cut from the contribution figures.

Want me to note in QUESTIONS.md that seeds 43/44 are required for the contribution figures (not optional), so it doesn't slip?

2. Will the architecture of models change if we try full scale training to 100 percent of dataset -> 3.30 M or will we do the dataset training on same architecture as we did for 1.5 M ie 50 percent of full dataset?

Answered:
-> **Same architecture, unchanged.** The 100-run sweep is staged in
`scripts/run_fullscale100_seed42_sweep.py` and it differs from the 50-run sweep in **one thing
only: `DATA_ROOT`**. Identical model definitions, identical frozen config (GNN depth 4, 30
epochs, unweighted CE, grad-clip 1.0, 3 warm-up epochs), identical splits logic.

That is deliberate, and it is what makes the comparison meaningful. The project's hard rule is
*change one variable at a time* — if you scaled the corpus **and** retuned the architecture, a
change in accuracy could not be attributed to either. Doubling the data with everything else
frozen answers exactly one question: *does more data help at this architecture?*

Two caveats worth knowing before launching it:
- Best-validation epoch is already **27.7–28.7 of 30** for RGCN/RGSAGE-*, i.e. those models are
  still improving when the budget ends. On 2× the data they may be *more* under-trained, so a
  flat or worse result would be ambiguous — it could mean "more data doesn't help" or "30 epochs
  is now too few". Consider raising the epoch budget if you run it, and say so explicitly.
- Cost: the 50-run sweep was 4.7–12.3 GPU-hours *per run*, ×5 models ×3 seeds. At 2× data,
  budget roughly double.

**Status: the 100-run corpus is generated (`dataset_fullscale_100`, 3.30M windows) but the sweep
is PARKED — do not launch it without being asked.**