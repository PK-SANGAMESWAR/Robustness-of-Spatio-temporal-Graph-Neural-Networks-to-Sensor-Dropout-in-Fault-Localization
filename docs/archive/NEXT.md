**Stale as of 2026-10-15.** This file's "next prompt" sequence stops at "don't start Stage E
yet" / "once S4.1 lands" / "before committing to S4.3's ~21 GPU-h sweep" — all of that has long
since happened. Current actual status: S4.1 through S4.3ᵉ are done, Gate 4 (R0ⁿ vs R5, 30
epochs, 3 seeds each) started 2026-10-09, was amended to 30 epochs and restarted 2026-10-12, and
is in progress (`06_STAGE_E_pcstgnn.md` "S4.4 — Frozen settings for Gate 4 runs" and its two
amendments are the current source of truth; `results/s4_4/gate4_progress.json` has the live
state). Kept below, not deleted, as the historical handoff sequence it was.

Report back the physics-imputation sweep results (macro-F1, AUDC relative and absolute, Ret@10,
at each k, each seed) and the S3.4 speed benchmark number, side by side with the existing
neighbour-mean imputation baseline (AUDC 0.476, Ret@10 11.8%) for direct comparison.

Then check Gate 3's checklist in 05_STAGE_D_infra.md against what's actually done:
- [ ] Injectors deterministic and shared-seeded across all models
- [ ] Silent injectors verified to leave m=1 and skip uninstrumented phases
- [ ] Residual matches the reference loop
- [ ] Residual under 5ms/batch, or cached
- [ ] Physics-imputation baseline evaluated on the full sweep

For each item, confirm pass/fail with a pointer to the evidence (test file, benchmark number, or
result file). This is a mechanical infrastructure checklist, not a project-direction decision, so
if all five are genuinely satisfied, tick them in the file yourself and commit — no need to wait
for my sign-off on this one, unlike Gate 2.

Do not start Stage E (S4.1 null test) yet — report the sweep results and the Gate 3 status first
and stop there.



*************************************************************************************

After S4.1 finishes and passes, here's the sequence — hold each prompt until the prior one confirms.

If S4.1 passes (exact reproduction of R0's per-epoch curve):

S4.1 passed. Proceed to S4.2 — the gradient audit, per 06_STAGE_E_pcstgnn.md.

Run finite-difference checks on THREE separate batches, not one:
1. A batch containing w=20 windows (zero post-inception samples), so the L1/L2 loss mask is
   active and non-trivial.
2. A batch at k=10, so |A| is small and |D| is large -- lambda_2's intended regime.
3. A batch at k=0 with no synthetic drops, so |D|=0 -- the commonest training case, and the one
   a naive reconstruction-loss formulation would turn into NaN or zero gradient.

For each batch confirm:
- dL1/dw is non-zero for network weights w
- dL2/dw is non-zero
- beta, alpha, c receive non-zero gradients
- mu, mu_0, sigma_eps receive NO gradient (they must not be leaves with requires_grad=True)

Report pass/fail on all four checks for all three batches -- 12 confirmations total. This is a
hard gate per the project's own framing ("two hours here prevents weeks of silently identical
results") -- do not proceed to S4.3 if any of the 12 fail.

Once S4.2 passes, tuning (S4.3) is next, but that's a good point for you to make a call rather than just handing off blindly — it's the first step that starts consuming real GPU budget on new hyperparameter search (≈21 GPU-h across 6 runs) rather than a single validation run. Worth a quick check-in prompt first:

S4.2 passed (12/12). Report current cumulative GPU-hours spent this project against the budget in
13_BUDGET_TIMELINE.md's ~255-270h estimate, so I know where we stand before committing to S4.3's
~21 GPU-h tuning sweep (6 runs: lambda_1 in {0.1,0.3,1}, then lambda_2 in {0.1,1} at best
lambda_1, beta learned, 15-epoch budget each, validation F1 averaged over k selection). Don't
start the sweep yet -- just report the budget position.

If S4.1 fails instead — don't let it proceed past this point at all:

S4.1 failed to reproduce R0 exactly. Report the specific point of divergence: at which epoch does
the validation F1 curve first differ, by how much, and is the divergence in the loss value, the
gradient path, the data loading order, or the checkpoint comparison itself. Do not attempt S4.2
or any further Stage E work until the source of the mismatch is identified and fixed -- this
property is what every later attribution claim in the paper rests on, so a silent near-miss here
is worse than an obvious failure.

Send whichever applies once S4.1 lands, and I'll give you the next step from there.