Execution Plan: Paper 1 → PC-STGNN / PCRG-STGNN

Every step has a Do, a Check with a numeric pass criterion, and a Branch telling you what to do if it fails. Nothing expensive runs before the cheap thing that could kill it. Five gates; the decisive one is Gate 2, at about week 6, before any GPU time is committed.

0. Frozen specification — the single source of truth

Everything below refers to this. If you change any line here, restart from Gate 1.

Physics operator (fixed, zero parameters, zero gradient)

Axes:  f ∈ 25 locations · τ ∈ 11 types · R ∈ 15 values (0.05–30 Ω, log-spaced)
For each (f, τ, R):  solve N_L = 12 times, each with an independent
per-bus load draw from U(0.5, 1.3) — the corpus's own load distribution
Store:   μ[f,τ,R,i,p]        mean post-fault magnitude, per µPMU phase
         σ_load[f,τ,R,i,p]   std over the load ensemble
Cost:    25 × 11 × 15 × 12 = 49,500 snapshot solves ≈ 15–20 min, ≈ 2.5 MB

Observation vector. s_ip = mean of the post-inception timesteps in the window. A = {(i,p) : m_ip = 1}. D = the synthetically dropped set (training only).

Residual — inputs only, no network output on the right-hand side:

r_f(A) = min over (τ, R) of   Σ_{(i,p) ∈ A}  (s_ip − μ[f,τ,R,i,p])²
                                             ──────────────────────────
                                             σ_load[f,τ,R,i,p]² + σ₀²

r̃_f(A) = r_f(A) / |A|

Load is marginalised, not minimised. (τ, R) are profiled. σ₀ = injected measurement-noise level (0 on the clean corpus).

Held-out residual r̃_f(D): take the (τ*, R*) profiled at f using A alone, then score at the sensors in D.

PC-STGNN. Paper-1 backbone untouched: GRU(128) → 4 × RGSAGE-Max(64) → BN/ReLU/Drop 0.35/Linear(26) → soft vote → z ∈ ℝ²⁶.

z'_f = z_f − β · r̃_f(A)

L = CE(z', y)  +  λ₁ · Σ_f P(f|X)·r̃_f(A)  +  λ₂ · Σ_f P(f|X)·r̃_f(D)

λ₁ = λ₂ = β = 0   ⟹   exactly dropout-aware RGSAGE-Max

PCRG-STGNN. Pass 1 gives f̂ = argmax z'. For each available sensor i: refit (τ, R) at f̂ on A \ {i}, read the normalised error at i → r̃ᵢ^LOO. Gate gᵢ = mᵢ · σ(−α(r̃ᵢ^LOO − c)). Pass 2 scales sensor i's input embedding and soft-vote weight by gᵢ — never its messages at every hop. Three new learnable scalars total: β, α, c.

Architectural rule. The physics operator is a function of the inputs only. No network output is ever an argument to μ or σ_load. The type head, if kept, is auxiliary supervision and is never read by the physics.

STAGE A — Verify what you have (Week 1, no new science)

Nothing here produces a result. All of it prevents a month of wasted work.

S0.1 — Reproduce Paper 1

Do. Load the seed-42/43/44 checkpoints, re-run the dropout sweep k ∈ {0,1,2,5,10} with the stored mask seeds.
Check. Every number in Paper 1's Table 6 reproduces to the printed precision.
Branch. If it does not, stop the project and fix reproducibility. Every comparison in this paper is paired against these numbers; if they drift, nothing downstream is interpretable.

S0.2 — Metadata audit

Do. For all 41,250 events, confirm you stored: fault location, fault type, fault resistance, the per-bus load multiplier vector, and the RNG seed.
Check. 100 % coverage, and re-running the generator on 5 events with the stored seed reproduces the record bit-for-bit.
Branch. If the per-bus load vectors are missing, you cannot do the S1.5 dispersion check event-by-event. Fall back to the distributional version (compare the corpus within-scenario spread across the 50 runs against σ_load) — still valid, slightly weaker. If the seeds are missing, the partial-regeneration escape hatch (voltage angles) is closed; note it and continue.

S0.3 — Window census

Do. Count fault-labelled windows by the number of post-inception samples they contain. With a 60 ms record, inception at 40 ms, and 40 windows of 20 steps, some fault-labelled windows sit entirely before inception.
Check. Produce the histogram. Record the fraction with 0 post-inception samples, and with 1–5.
Branch. Whatever the number, adopt this policy and state it in the paper: windows with 0 post-inception samples have the physics term disabled (r̃ = 0, excluded from L₁, L₂). Windows with 1–5 samples get a noisier s; do not special-case them, but report their F1 separately in the appendix. If more than ~25 % have zero, add a "physics-active window" column to every results table so a reviewer sees the operator only acts where it can.

S0.4 — Snapshot-mode equivalence

Do. Take one known corpus event. Re-solve it in OpenDSS snapshot mode using your existing fault-injection code, at the event's own stored per-bus load multipliers, same (f, τ, R).
Check. max |V_snapshot − V_corpus_postfault| < 1e-3 p.u. across all 75 channels.
Branch. If it fails, the fault model differs between modes (common causes: fault element applied before vs after the first solve, Solve mode=snap not re-initialising the control iteration, different MaxControlIter). Fix this before building the table — the whole operator inherits the error. Repeat for 5 events spanning 0.1/1/10 Ω and at least 3 fault types.

S0.5 — Timing calibration

Do. Time one 30-epoch training run (or one epoch × 30), and 100 snapshot solves on your machine.
Check. Record both numbers.
Branch. Rescale the GPU budget in Section 6 linearly. If one full run exceeds ~6 h, drop the 3-seed requirement to 3 seeds only for R0/R5/G0/G2 and 1 seed for the rest, and say so in the paper.

GATE 0 — proceed only if

S0.1 reproduces exactly · S0.4 matches to 1e-3 p.u. · the window census is documented.
Deliverable: a one-page audit note. Time: 1 week.

STAGE B — Build and validate the physics operator (Weeks 2–4)

This stage answers one question: is a magnitude-only signature table a faithful description of your corpus? If not, nothing after it works, and you find out for the price of a few hours of CPU.

S1.1 — Network matrices

Do. Extract the three-phase Y-bus from the .dss model; invert (or factorise) to get the Z-bus entries you need for the sensitivity figure |Z_pf| / |Z_ff|.
Check. Y-bus is symmetric; reconstructing one known branch current from a solved power flow via I = Y·V matches OpenDSS's own reported current to < 0.1 %.
Use. This is for the paper's physics section and the identifiability discussion. It is not used in the residual — say so explicitly, because reviewers will assume otherwise.

S1.2 — N_L convergence

Do. For 3 representative (f, τ, R) combinations (near sensor / far lateral / 10 Ω), build σ_load with N_L ∈ {4, 8, 12, 20, 40}.
Check. σ_load stabilises — the change from N_L=12 to N_L=40 is under ~10 % on most channels.
Branch. If it has not converged by 12, raise N_L to the knee. Cost scales linearly and is still under an hour at N_L = 24. Report the chosen N_L and this curve in the appendix.

S1.3 — Build the table

Do. Generate μ and σ_load over the full grid. Parallelise across 4–6 processes, one OpenDSS engine each.
Check. No NaNs; no non-converged solves (log and re-solve any); monotonicity sanity — sag depth at the nearest sensor decreases monotonically with R for each (f, τ).
Branch. Non-monotonic sag depth usually means a control-loop (regulator/cap-bank) interaction. Decide once, and match the corpus: either freeze controls in snapshot mode exactly as the corpus daily-mode run had them, or allow them and document it. Re-run S0.4 after any change.

S1.4 — Bias check on μ

Do. For every training event, compute e_ip = s_ip − μ[f_true, τ_true, R_true, i, p].
Check. Report median and IQR of |e| by fault type, by R, and by sensor.
Pass criterion. At 10 Ω — the hard case — the median |e| at the deepest-sagging channels must be well below the 10 Ω sag itself. Your Paper 1 Fig. 2 puts that sag near 0.02 p.u., so target median |e| / sag < 0.3, i.e. roughly |e| < 0.006 p.u.
Branch (in order). (a) Raise N_L. (b) Check for a systematic offset — a constant bias means a source/regulator setpoint mismatch between snapshot and daily mode; fix it rather than absorbing it into σ. (c) Only as a last resort, calibrate μ from training-split event means per (f,τ,R) and describe it honestly as a model-derived template calibrated on training data, weakening the "no data used" claim.

S1.5 — Dispersion check on σ_load ← the new decisive test

This is the test that validates Change 3, and it is exact because your corpus already contains 50 independent load draws per scenario.

Do. For each (f, τ, R) scenario, compute the empirical standard deviation of the post-fault level across its 50 runs, per channel. Plot it against σ_load[f,τ,R,i,p].
Check. Points cluster on the 1:1 line. Report the slope of a through-origin regression and the fraction of channels where the table under-disperses (σ_load < empirical).
Pass criterion. Slope in ~[0.8, 1.25]; under-dispersion on fewer than ~20 % of channels.
Branch. Systematic under-dispersion means the residual denominator is too small, the χ² is inflated, and PSM will look over-confident and brittle. Fix by (a) confirming the load draw really is per-bus independent U(0.5,1.3) in your generator — re-read the code, do not trust the paper text; (b) including any other per-run randomness the corpus has (e.g. inception phase angle) in the ensemble; (c) inflating with an empirical floor σ_ip² ← σ_load² + σ_emp² estimated on the training split only, and documenting it.

This scatter plot goes in the paper. It is the figure that converts "we assumed a load model" into "we tested our load model."

S1.6 — Correlation check (diagonal vs whitened)

Do. From the same N_L load samples, estimate the channel-channel correlation of the load-induced deviation for a few hypotheses.
Check. Largest off-diagonal correlations.
Branch. If correlations are mostly < 0.3, keep the diagonal form and state the approximation. If strong block structure appears along laterals, estimate a rank-k (k ≈ 3–5) covariance from the existing samples and whiten. Costs nothing extra — the samples are already computed — and it strictly improves the score.

S1.7 — R-grid sufficiency

Do. Two sub-tests, both free.

Discretisation: rebuild the residual using only every second grid point (7 values), re-run S2.4's PSM later and compare F1.
Interpolation: for the corpus's own R ∈ {0.1, 1, 10}, check where the profiled argmin_R lands.

Check. Discretisation costs < 0.01 macro-F1; argmin_R lands within one grid step of the true value on the large majority of events.
Branch. If either fails, densify to 25 log-spaced points. Cost rises to 82,500 solves — still under an hour. Report the chosen grid and this test; it is exactly what their point 6 asks for.

GATE 1 — Fidelity. Proceed only if

S1.4 bias passes at 10 Ω · S1.5 dispersion slope in [0.8, 1.25] · S1.7 discretisation loss < 0.01 F1.

If Gate 1 fails after all branches: magnitude-only physics is too weak for this corpus. Switch to the partial-regeneration route — re-log the same events (needs S0.2 seeds) with voltage angles at the 25 µPMUs — and rebuild the operator on complex quantities. This is a fork in the road, not a failure; take it here, at week 4, rather than at week 16.

Time: 3 weeks, CPU only.

STAGE C — Training-free science (Weeks 4–6). The main GO/NO-GO

Everything here runs on existing checkpoints and CPU. It decides whether the project proceeds.

S2.0 — Generate the OOD set, and a throwaway pilot

Do. Two separate generations, with different RNG seeds:

OOD-pilot (~1,100 events): R ∈ {5, 20} Ω × 25 locations × 11 types × ~2 runs. Used at Gate 2, then discarded and never reported.
OOD-test (~11,000 events): R ∈ {3, 5, 20, 30} Ω × 25 × 11 × 10 runs. Sealed. Never opened until Stage G.

Why two. You need OOD evidence to make the Gate-2 decision, but touching the real OOD set that early burns it — any decision you make after seeing it contaminates it. The pilot buys the signal; the sealed set carries the claim.
Check. Same generator, same load distribution, ~0.3–1.5 h of simulation.

S2.1 — Identifiability matrix

Do. For every location pair, compute the separability score under the new statistics:

D(f₁, f₂; A) = min over (τ,R) pairs of  Σ_{(i,p)∈A}  (μ_f₁ − μ_f₂)²
                                                     ─────────────────
                                                     σ_f₁² + σ_f₂²

The min over hypotheses makes it the hardest case — the most confusable (τ,R) combination. Average over random sensor subsets at k ∈ {0, 2, 5, 10}, reported separately per R.
Check. Produce the 25×25 heatmap per k. Count pairs with D < 1 (indistinguishable at the load-noise scale).
Read it against the within-class spread, not against zero. On a noise-free corpus everything looks separable if you compare to zero; the Fisher-style ratio above is the honest comparison.

S2.2 — Empirical confusability

Do. Independently of the table: nearest-neighbour confusion between training events at different locations, using the raw observation vectors.
Check. Does the empirical confusion structure match the S2.1 heatmap?
Why. It separates the two failure causes cleanly. Table says separable + corpus says confusable → your table is wrong. Both say confusable → a genuine information limit, which is itself a publishable result.

S2.3 — B1: ΔV nearest-signature baseline

Do. Δs = s − v_nominal, with v_nominal the per-channel no-fault mean from the training split. Nearest table signature at nominal load, no profiling over τ, R, or load.
Check. Full Paper-1 evaluation: F1 at every k, F1 by R and type, AUDC.
Purpose. This is the classical method. B1 → B2 measures exactly what profiling and the load model buy you.

S2.4 — B2: PSM

Do. f̂ = argmin_f r̃_f(A), no network at all. Full Paper-1 evaluation.
Check. Compare against B1 and against the Paper-1 models.
Also run. PSM on the OOD-pilot at R ∈ {5, 20} Ω.

S2.5 — The extrapolation contrast

Do. Evaluate the existing dropout-aware RGSAGE-Max checkpoints on the OOD-pilot.
Check. Compare the neural model's F1 drop from in-distribution → OOD against PSM's.
Why this matters more than anything else in Stage C. The network was trained on {0.1, 1, 10} Ω. PSM has a dense, continuous representation of R. If the network collapses on unseen resistance and PSM does not, you have your paper's central claim demonstrated before writing a line of model code.

S2.6 — Inference-only fusion

Do. Apply z'_f = z_f − β·r̃_f(A) to the existing R0 checkpoints. Tune β by grid search on the validation split only.
Check. ΔF1 at k = 0, at k = 10, and at 10 Ω, on validation.
Time. Minutes. This is the cheapest possible test of "does physics know something the network doesn't."

S2.7 — Pre-registration

Do. Before opening any test-split results, write down — dated, in the repo — the conditions where you predict the gain will appear (10 Ω, k = 10, OOD R, noise, silent failures), the ablations that must fail (R6 shuffled, R2 learned decoder), and the statistical test.
Why. It costs an hour and it is the difference between "we found an effect" and "we searched until we found one." It also makes the results section write itself.

GATE 2 — the decision
Observation (validation + pilot only)	Reading	Action
S2.6 improves F1 at 10 Ω or k = 10	Physics carries information the network lacks	GO — full plan
S2.6 flat, but PSM holds up on the OOD pilot while the network collapses (S2.5)	The network already learned the in-distribution physics; physics buys extrapolation	GO — reframe around OOD resistance + silent failures. Phases E–G unchanged
PSM ≈ or > the STGNN everywhere	Physics dominates; the network may be unnecessary	GO — but change the question to "where does learning still help?" (dropout, noise, structured loss). Be honest; this is still a good paper
PSM weak and S2.1 shows many pairs with D < 1 at k = 10	You are at the information limit of 25 magnitude-only µPMUs	PIVOT to the identifiability + silent-failure paper (Stage F still runs; Stage E is dropped)

Deliverables: identifiability heatmap · B1 vs B2 vs Paper-1 table · the S2.6 number · the S2.5 contrast · the pre-registration note.
Time: 2–3 weeks. GPU used: none.

STAGE D — Infrastructure and stronger baselines (Weeks 5–9, runs in parallel)

Start the GPU work here, during Stage C, since it is needed under every Gate-2 branch.

S3.1 — Failure injectors

Do. One data-loader hook, extending your Paper-1 masking. No new simulation for any of it.

Injector	Settings
Random dropout	k ∈ {0,1,2,5,10} (Paper 1)
Structured	whole sub-trees of the measured-only graph
Persistent	the same k sensors missing for all 40 windows of an event
Silent, m = 1	stuck at pre-fault value · bias ±2 %, ±5 % · Gaussian σ = 0.02 p.u. · swapped phase channels — on s ∈ {1,2,3} sensors
Measurement noise	σ ∈ {0.001, 0.005, 0.01} p.u. on all sensors

Check. Unit tests: with a fixed seed, the injector is deterministic and reproducible across models; k=0 and no corruption returns the tensor byte-identical to the un-injected batch.
Note. Stuck-at-pre-fault is the most realistic silent failure — a sensor that stops updating. Weight the discussion toward it.

S3.2 — Dropout-aware baselines

Do. Train GRU, RGCN, RGSAGE-Mean, RGATv2 with dropout-aware training, 3 seeds each — 12 runs. RGSAGE-Max already exists.
Check. Each beats its standard-trained counterpart on AUDC.
Why. Without this, a reviewer says "physics only helps Max aggregation." It also strengthens Paper 1's own claim.

S3.3 — Impute-then-infer baseline

Do. Fill dropped sensors with μ at the PSM-best hypothesis, then run unmodified RGSAGE-Max.
Check. No training needed. This closes the gap Paper 1 flagged as missing.

S3.4 — Residual computation layer

Do. Implement r̃(A) and r̃(D) as batched tensor ops: broadcast the [25, 11, 15, 75] μ and σ tensors against [B, 75] observations, mask, sum, min over the (τ,R) axes → [B, 25].
Check. Correctness against a slow reference loop on 100 samples. Speed: under ~5 ms per batch of 32 on the T1000. LOO adds one more pass: [B, 25 sensors] by removing each sensor's term before the min.
Branch. If it is slow, precompute and cache r̃ per window — it depends only on inputs, so it can be computed once for the whole corpus and stored. That is the single most useful optimisation available, and it is only possible because of the architectural rule.

GATE 3 — infrastructure correctness

Injectors deterministic and shared-seeded across models · residual matches the reference loop · residual under 5 ms/batch or cached.

STAGE E — PC-STGNN (Weeks 8–13)
S4.1 — The null test ← run this before anything else

Do. Train the new model with λ₁ = λ₂ = β = 0, gate off, seed 42.
Check. It reproduces R0 exactly — same loss curve, same validation F1 per epoch.
Branch. If it does not, you have an unintended change in the data path, the seeding, or the graph. Find it now. Every attribution claim in the paper rests on this single property.

S4.2 — Gradient audit ← guards the bug that was in the earlier draft

Do. For a single batch, finite-difference check: perturb a network weight, confirm ∂L₁/∂w ≠ 0 and ∂L₂/∂w ≠ 0. Separately confirm ∂L/∂(table)  does not exist — the table must not be a leaf with requires_grad.
Check. Both λ terms produce non-zero gradients through P(f|X); β, α, c receive gradients; the table receives none.
Why. The earlier formulation of λ₂ (reconstruction at the arg-min hypothesis) has zero gradient — the arg-min is an integer index, constant in the weights. R4 would have been silently identical to R3. This two-hour check prevents weeks of confusion.

S4.3 — Tuning

Do. Single seed, 15-epoch budget (as in your Paper 1 depth study). Sweep λ₁ ∈ {0.1, 0.3, 1}, then λ₂ ∈ {0.1, 1} at the best λ₁. β is learned. ≈ 5–6 half-cost runs.
Check. Select on validation F1 averaged over k, never on test, never on OOD.
Branch. If the best λ₁ sits at a boundary, extend the grid one step in that direction.

S4.4 — Core ablation, 3 seeds × 30 epochs, everything else frozen
#	Variant	Answers the objection
R0	Dropout-aware RGSAGE-Max	reference (already trained)
R1	+ type head only	"the auxiliary task is doing it"
R2	+ learned MLP decoder on D, parameter-matched	"any decoder / masked autoencoding would do it"
R3	+ λ₁ (residual on A)	physics as a training constraint
R4	+ λ₂ (residual on held-out D)	does held-out consistency add over in-sample?
R5	+ β fusion — full PC-STGNN	physics as evidence at test time
R6	R5 with a shuffled location→signature map	"it's regularisation, not physics"
R7	Size-matched wider RGSAGE-Max	"it's just a bigger model"
B2ᵃ	learned R̂ head instead of profiling, 1 seed	demonstrates why Option A wins on OOD R

R6 and R2 are the two that carry the burden of proof. R7 matters less than it looks — the physics path adds three scalars — but run it anyway, because it is a one-line objection and a one-row answer.

GATE 4

S4.1 null test exact · S4.2 gradients non-zero on both λ terms · R5 > R0 on validation in the pre-registered conditions, sign-consistent across 3 seeds.
If R5 ≈ R0 on validation: do not proceed to Stage F expecting a rescue. Go straight to Stage G, run the OOD and noise evaluations, and write the "physics buys generalisation, not in-distribution accuracy" paper.

STAGE F — Trust gate (Weeks 12–16)
S5.1 — Training-free detection ← hard gate

Do. Using PSM hypotheses only, no training: compute r̃ᵢ^LOO on clean vs corrupted sensors, for every corruption type.
Check. AUROC per corruption type and per severity.
Hard threshold. If AUROC < 0.75 under the mildest corruption (±2 % bias), do not train PCRG.
Branch on failure. Publish PC-STGNN alone and report this as a bounded negative finding: "physics residuals from magnitude-only measurements cannot detect sensor bias below X %." Include the AUROC-vs-severity curve. That is a genuinely useful sentence for the field, it costs nothing, and it is a better outcome than a gate that does not work.

S5.2 — LOO vs in-sample (free)

Do. Compute both residual forms and compare their detection AUROC.
Check. LOO should be clearly better, because an in-sample fit is pulled toward the corrupted sensor and masks its own error.
Why. This is a one-paragraph, one-figure justification of the LOO design — a reviewer will ask, and the answer is free.

S5.3 — Gate training, 3 seeds each
#	Variant	Purpose
G0	dropout-aware + corruption augmentation, no physics	the critical control: "why not just augment with corruption?"
G1	PC-STGNN + the same augmentation	isolates the gate from the augmentation
G2	PCRG-STGNN — gate on evidence and vote weight	the headline result
G3	gate at every hop, 1 seed	re-tests the Paper-1 SDR-STGNN lesson

G0 is the one that matters. If G2 does not beat G0, the gate is not earning its complexity and the honest move is to report it.

GATE 5

S5.1 AUROC ≥ 0.75 at the mildest corruption · G2 > G0 under silent failures, sign-consistent across seeds.

STAGE G — Full evaluation and robustness (Weeks 15–19)

Mostly inference. Minutes per model per condition.

S6.1 — Master sweep

Every model × every condition, shared seeds throughout.

Condition	Settings
Random dropout	k ∈ {0,1,2,5,10}
Structured / persistent	sub-trees; per-event persistent
Silent failures	4 types × s ∈ {1,2,3}
Measurement noise	3 levels
OOD resistance	{3,5,20,30} Ω × k ∈ {0,5,10} — open the sealed set here, once
S6.2 — Model-mismatch study ← answers the leakage objection with evidence

The sharpest criticism you will face is not "you used test events" — your checklist handles that. It is: "the table comes from the same simulator that made the data; a real utility has an approximate model." No paragraph answers that. An experiment does.

Do. Rebuild the signature table from a perturbed feeder model; leave the corpus and the trained models untouched.

Perturbation	Levels
Line impedances, random per segment	±2 %, ±5 %, ±10 %
Load composition (constant-P vs a ZIP mix)	one alternative
One unmodelled lateral, or a small unmodelled DER	one scenario

Check. Plot the gain over R0 as a function of mismatch level.
Cost. ~20 min of solves per perturbed table. The fusion-only variant needs no retraining at all. One retrain at the ±5 % level confirms the trained variant behaves the same way.
Either outcome is publishable. Graceful degradation → "the operator tolerates realistic feeder-model error," which is exactly what a power-systems reviewer wants to read. Collapse at ±2 % → a real limitation, found in a week rather than after submission.

S6.3 — Metrics and statistics
Graph-level macro-F1; F1 by R, by type, per location; retention; AUDC.
Silent-failure detection AUROC.
Mean ± std over the 3 seeds, and paired bootstrap over test events for the pre-registered comparisons only: R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2.
With ~6,175 test events, the bootstrap resolves small differences easily — so seed variance, not significance, is your binding constraint. Report both and lead with the seed spread.
STAGE H — Analysis and writing (Weeks 19–26)
S7.1 — Error attribution

Overlay each model's confusion matrix on the S2.1 identifiability heatmap. The target sentence: "the residual errors concentrate in location pairs that are physically indistinguishable under the available sensor subset." This is the deepest section in the paper, and it works whether or not the model gains are large.

S7.2 — Case studies

One correctly gated silent failure, one gate failure, one 10 Ω event where physics flips the prediction. Show the residual profile r̃_f across all 25 candidates for each. Reviewers remember these.

S7.3 — Release

Code, the signature table, the generator for the OOD set, and the perturbed tables. PPGN set the precedent for public code in this niche; releasing measurably improves reception.

Success criteria — the pattern, not a threshold

A fixed "≥ 0.03–0.05 macro-F1" bar is the wrong instrument. With your test-set size, significance is cheap; what makes the result physical is where the gain appears. Judge against all six:

The gain concentrates in the pre-registered conditions: 10 Ω, k = 10, OOD R, injected noise, silent failures.
Clean k = 0 accuracy does not drop by more than one seed standard deviation.
Sign-consistent across all three seeds.
Paired bootstrap p < 0.05 on the pre-registered comparisons only.
R6 (shuffled physics) removes the gain and R2 (learned decoder) does not reproduce it.
It survives ±5 % model mismatch.

+0.01 in-distribution with +0.10 on unseen resistance is a stronger paper than +0.05 everywhere — only the first pattern is evidence that the mechanism is physical rather than regularisation.

Fallback map — every branch still ends in a paper
Where it lands	The paper you write
Gains on OOD R and noise, flat in-distribution	"Physics buys generalisation, not in-distribution accuracy." Honest, and the most likely outcome
Gains only under silent failures	Refocus on silent-failure robustness; PC is the enabling component, not the headline
Gate AUROC < 0.75	PC-STGNN paper + a bounded negative result on magnitude-only silent-failure detection
No model gains, clear identifiability limit	Identifiability + benchmark paper: how much 25 sparse magnitude-only µPMUs can reveal. IEEE Access or SEGAN
PSM ≈ STGNN	Physics dominates in-distribution; show where learning still helps (dropout, structured loss, noise)
Gate 1 fidelity fails	Fork to the voltage-angle route at week 4, with the metadata from S0.2
Budget and timeline
Block	Runs	GPU hours
Dropout-aware baselines (4 × 3 seeds)	12	40–48
λ tuning (half cost)	6	10–12
Core ablation R1–R7 × 3 seeds	21	65–85
B2ᵃ learned-R̂ head, 1 seed	1	3–4
Gate study G0–G2 × 3 + G3 × 1	10	32–40
Model-mismatch retrain at ±5 %	1	3–4
Total	~51	~155–195 h ≈ 7–8 days of continuous T1000 time

CPU-only work: ~50 k table solves (20 min) + N_L study + 3–5 perturbed tables (~1.5 h) + OOD generation (~1.5 h). All evaluation is inference, minutes per model.

Weeks	Stage	GPU in background
1	A — audit	—
2–4	B — operator + fidelity → Gate 1	—
4–6	C — identifiability, PSM, fusion → Gate 2	dropout-aware baselines start
5–9	D — injectors, OOD sets, impute baseline	dropout-aware baselines
8–13	E — PC-STGNN, tuning, R0–R7 → Gate 4	ablation runs
12–16	F — detection AUROC, G0–G3 → Gate 5	gate runs
15–19	G — full sweep, OOD, model mismatch	—
19–26	H — analysis, figures, writing, release	—

About 6.5 months. The decisive commitment point is week 6, after roughly five weeks of CPU-only work. That is the property that makes this plan low-risk: if the physics does not carry information, you learn it before spending a single significant GPU hour, and you still have four publishable fallbacks.

Leakage and fairness checklist — verify before every table
 The true R, type, or load multiplier is never supplied to the model or the residual at test time.
 σ, β, λ₁, λ₂, the gate parameters α, c, and the R grid are all fixed using train/validation only.
 The sealed OOD set was opened once, in Stage G, and never used for any selection. The Gate-2 pilot used different seeds and is not reported.
 If any model receives extra input, all baselines receive the same input.
 Mask seeds, corruption seeds, and noise seeds are shared across all models.
 Checkpoint selection is validation F1 at k = 0 for every model, as in Paper 1.
 The paper states that the table comes from the nominal feeder model, and backs it with the S6.2 mismatch study.
 Any "first" claim is narrowed against Karabulut et al. 2025, which already studied sensor-configuration robustness for detection.