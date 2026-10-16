This is a strong implementation plan — it was clearly written against the actual code, not against my spec text, and it caught something important that I got wrong. Approve it with the corrections below. Four of the issues will bite during execution if not fixed now.

The headline: your spec and your code disagree about the load model

The implementer says runner.py:116 applies one scalar load multiplier per run. Paper 1 §3.1 says "the per-bus load independently scaled by a factor drawn from U(0.5, 1.3)."

Both cannot be true, and it matters twice over.

If the code is right (a scalar), then the implementer's correction is correct and my σ_load ensemble was solving a problem you do not have. A scalar L is a profiled nuisance parameter exactly like τ and R — min over a 17-point grid, not marginalisation over an ensemble. Everything downstream simplifies: σ_ε becomes a small table-vs-corpus fidelity floor rather than a load-variation term; the S1.5 check correctly becomes L* recovery rather than a dispersion scatter; and the reviewer's "biggest weakness in the plan" dissolves entirely.

But Paper 1 then misdescribes its own dataset. That is a correction you need to make regardless of this project — in an erratum if it is already in press, and certainly by not repeating the claim in Paper 2. Make this the very first item in Stage A, before audit_metadata.py: read runner.py and the stored per-event metadata, and settle it in writing. If the stored metadata has a scalar load_multiplier field rather than a 123-vector, that is your answer.

Decision: confirm the scalar L axis, conditional on that verification. If it turns out the code really is per-bus, revert to the ensemble form in 01_SPEC.md §2 and tell me — the rest of this review still stands.

The four §1 decisions
#	Decision	Verdict
1	Scalar L axis, 17 points	Confirm, after the code-vs-paper check above
2	Build via the corpus's own Daily-mode path, snapshot as cross-check	Confirm. Good call — it makes snapshot-vs-Daily equivalence a non-issue by construction rather than something to prove
3	Seed plan	Confirm with one swap: give R4 three seeds and drop R7 to one. λ₂ is the novel loss term and R4 is where its effect is isolated; R7 answers a one-line objection and a single seed suffices
4	No GPU before Gate 2	Confirm. Non-negotiable
Issue 1 — the residual will overflow the loss (fix before Stage E)

With a scalar load profiled out and σ_ε ≈ 1e-3 p.u., a wrong hypothesis produces per-channel terms of order (0.1/0.001)² = 10⁴. Summed over 75 channels, r_f for wrong locations reaches 10⁵–10⁶ while the true location sits near |A|. Then:

Σ_f P(f|X)·r̃_f is dominated by whichever wrong location happens to have mass, and λ₁ would need to be ~1e-5 to be stable.
z − β·r̃ saturates the softmax instantly; β has no usable range.

Fix — and it is also the more principled formulation. Shift and halve the residual, then treat it as a log-likelihood:

ρ_f      = ( r_f − min_g r_g ) / 2          bounded below by 0, scale-free
P_phys   = softmax( −ρ )
fusion   z'_nf = z_nf − β · ρ_f              (β a learned temperature)
L₁       = Σ_f P(f|X) · ρ_f

This is not a different objective. Because log P_phys,f = −ρ_f − log Z, the expected residual already is the cross-entropy between the network posterior and the physics likelihood, up to an additive constant. Shifting by min_g r_g removes that constant, and halving puts it in Gaussian log-likelihood units. β = 0 still recovers the baseline exactly, so the null test (S4.1) is unaffected.

Apply the same shift inside residual.py so PSM, the loss, the fusion and the LOO score all use one scale. Update 01_SPEC.md §4 and §6, and 11_METRICS.md §2 (report ρ_true and the margin ρ_(2) − ρ_(1), which becomes the natural decisiveness metric).

Issue 2 — residual_loo: subtract then min, never min then subtract

The plan describes "the subtract-one-term trick". That is valid only in one order.

✗  WRONG:  take r_f at the global argmin (τ*,R*,L*), then subtract channel i's term
✓  RIGHT:  build the per-channel error tensor, subtract channel i's term,
           THEN re-minimise over (τ,R,L)

The wrong order is precisely the in-sample residual — the thing S5.2 exists to show is inferior. Implementing it by accident would make S5.2 compare a variant against itself and silently kill the gate's justification.

Cost is manageable: build [B, f-chunk, 70125/25, 75] once per f chunk, then do 25 masked re-mins over the hypothesis axis. Add an explicit assertion to test_physics_residual.py: plant a corrupted sensor, verify ρᵢ^LOO ≫ ρᵢ^in-sample for that sensor.

Issue 3 — per-node fusion is a different object; describe what you implement

Broadcasting β·ρ[B,1,26] to every node is the right engineering choice — it keeps per-node CE, soft voting and the existing evaluator untouched. But be clear about what it computes. Soft voting sums softmaxes, so

Σ_n softmax(z_n − β·ρ)_f  ∝  Σ_n exp(z_nf) · exp(−β·ρ_f) / Z_n

Each node's distribution is reweighted multiplicatively by the same physics prior before summing. That is a product of experts applied per node, not z' = z − βρ on the graph-level logit vector. It is arguably better, and β being learned absorbs the scale difference — but the paper must state it as implemented. One sentence in §4, and one line in 01_SPEC.md §6. A reviewer who reimplements from the graph-level equation and gets different numbers is a reviewer you do not want.

Issue 4 — the noise-free-corpus trap around PSM

The plan's expected base case is "PSM ≈ or > STGNN at k = 0 on the clean corpus." Prepare for that, because taken at face value it is not a result — it is a tautology. On a noise-free corpus, with the table built from the exact generative model, with σ_ε ≈ 1e-3, nearest-signature matching should be near-perfect. A reviewer will say so in one line.

The consequence is strategic, and it should shape the paper now rather than in month five:

Do not lead with PSM's clean-corpus number. Report it as a sanity check confirming operator fidelity — which is genuinely what it is — and say so explicitly.
Promote the noise experiments from a robustness sub-table to a primary axis. Physics degrades under measurement noise; the network degrades under OOD resistance and severe dropout; the hybrid should own the union. That is the paper.
Promote the model-mismatch study (S6.2) alongside it. PSM on a perturbed table is where "physics alone is enough" dies, and it is the honest answer to the leakage objection.
Check real µPMU accuracy-class specifications when choosing the noise levels, and state which of {0.001, 0.005, 0.01} p.u. are realistic. At 10 Ω the sag is ~0.02 p.u., so σ = 0.01 is a ~2:1 signal-to-noise regime — that is the interesting operating point, not an extreme.

If Gate 2 lands on this branch, the reframed question is not "where does learning still help" in the abstract — it is "physics and learning fail under different perturbations; does fusing them cover both?" That is a better paper than the one originally scoped.

Issue 5 — the unlabelable-window ceiling (add to Stage A)

The census expects ~5 % of fault-labelled windows to contain zero post-inception samples. Excluding them from L₁/L₂ is right. But note what they are: windows that are physically identical to no-fault windows yet carry a fault-location label. No model can classify them above chance. That is a ceiling on macro-F1 built into the Paper-1 protocol.

Two cheap additions, both worth having:

Report F1 on that subset for every model. It should be ≈ chance. If any model does better, you have found label leakage somewhere.
Report the implied macro-F1 ceiling given that subset. It reframes every absolute number in both papers and makes the error-attribution section (F-H1) stronger: some residual error is physically unrecoverable by construction, not a model failure.

Add to 02_STAGE_A_audit.md S0.3 and to 11_METRICS.md §1.

Issue 6 — off-by-one on the pre-fault load-shape freeze

PREFAULT_FREEZE = 1 + 0.003·sin(2π·39/40) = 0.99953 is the shape value at step 39. The fault is injected at step 40, where the shape value is 1 + 0.003·sin(2π·40/40) = 1.0.

Make sure the fault solve uses the step-40 value and the pre-fault solve uses step 39 — whatever the corpus actually does. The error is only ~5e-4 p.u., but it is a systematic bias on every entry of μ, it is of the same order as σ_ε, and it will quietly inflate every residual. Add it as an explicit assertion inside audit_snapshot_equivalence.py rather than a comment.

Smaller points

s_ip = last timestep. Fine for the clean corpus — the post-fault level is flat to <1e-6 p.u., so the last timestep is the level, and it avoids needing inception position inside the residual (preserving spec §5). But under injected noise a single timestep is √n noisier than an average. Add a --obs-reduce {last, last5} flag, default last, and compare the two in the noise experiments only. Cheap, and it pre-empts the obvious question.

L-grid step. With σ_ε ≈ 1e-3 and a 0.05 load step producing maybe 2–5e-3 p.u. of voltage change, discretisation, not fidelity, becomes the dominant error term. Make S1.2's convergence check decide this rather than defaulting to 17 points, and test --l-step 0.025 (33 points, ~140 k solves — still comfortable on 24 cores).

The μ₀[L] no-fault row is a good addition I had not specified. The 26th class needs a signature too, otherwise PSM cannot express "no fault" and the comparison against 26-class models is unfair. Keep it, and make sure ρ is computed over all 26 candidates including f = 0 — the test file already checks for the f = 0 row, good.

"S3.2 dropout-aware baselines are done — nothing to run." Verify this. Paper 1 reports dropout-aware training for RGSAGE-Max only. If the repo genuinely has dropout-aware GRU / RGCN / RGSAGE-Mean / RGATv2 at 3 seeds, excellent — that is 12 runs saved. If it does not, they must be run, because "physics only helps Max aggregation" is a guaranteed reviewer objection.

Add to the master sweep: the physics residual ρ at each model's predicted location, for every model including the ones with no physics. It costs nothing from the cache and makes physical consistency a comparable column across the whole results table.

Spec files to update before execution starts
File	Change
01_SPEC.md §2	Scalar L axis (17 pts) replaces the per-bus ensemble; σ_load → σ_ε; add the μ₀[L] no-fault row
01_SPEC.md §4	Shifted/halved residual ρ_f; P_phys = softmax(−ρ)
01_SPEC.md §6	Fusion stated as per-node broadcast (product of experts); L₁, L₂ on ρ
01_SPEC.md §7	LOO defined as subtract-then-re-minimise
03_STAGE_B_operator.md S1.5	Dispersion scatter → L*/τ* recovery; F-B4 changes accordingly
02_STAGE_A_audit.md S0.3	Add the unlabelable-window ceiling analysis
11_METRICS.md §1, §2, §4	Ceiling metric; ρ_true and margin; σ_ε replaces dispersion metrics

Everything else in the 16 files survives unchanged, including all five gates, the ablation logic, the baseline ladder, the leakage checklist and the fallback map.