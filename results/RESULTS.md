blation result: decisive GO signal
edges ON	edges OFF	ratio
graph_macro_f1 (soft-voted)	0.343	0.124	2.8x
per-window macro_f1 (node diag)	0.227	0.082	2.8x
f1 @ 0.1Ω (easy)	0.471	0.152	3.1x
f1 @ 1Ω	0.352	0.132	2.7x
f1 @ 10Ω (hard)	0.153	0.049	3.1x
no-fault precision	0.994	0.922	—
no-fault recall	1.000	1.000	—


diag_depth2 finished (197.7 min ≈ 3.3h) — big result:

baseline (depth=1)	diag_depth2	delta
graph_macro_f1	0.439	0.571	+0.132
f1 @0.1Ω	0.584	0.726	+0.142
f1 @1Ω	0.457	0.578	+0.121
f1 @10Ω	0.199	0.371	+0.172 (nearly doubled)
no-fault precision/recall	0.969 / 1.000	0.992 / 0.9998	improved / preserved
This is a real jump — a ~30% relative gain, and it lifted the hardest bucket (10Ω) the most. It's just short of the 0.6–0.8 target band (0.571 vs. 0.60), but this strongly confirms your hypothesis: 1-hop depth was a genuine architectural ceiling, not primarily an underfitting or graph-mechanism issue.

diag_depth3 is running now — this is the one that tells us whether depth 3 pushes further into the target band, or whether over-smoothing kicks in and it comes back below depth-2 (in which case depth-2 is the sweet spot, not "more depth is better"). I'll report as soon as it finishes.