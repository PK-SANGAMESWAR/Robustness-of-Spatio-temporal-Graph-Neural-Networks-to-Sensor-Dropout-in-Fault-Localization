# Checklists

## A. Leakage and fairness — verify before every table

- [ ] The true R, fault type, or load multiplier is never supplied to the model or the residual
      at test time (the stored `load_multiplier` is used only in S1.4/S1.5 diagnostics)
- [ ] σ_ε, β, λ₁, λ₂, the gate parameters α, c, and the R and L grids are all fixed using
      train/validation only
- [ ] The residual is computed from the input tensor alone — no window index, event id, or
      inception offset (`01_SPEC.md` §3, §5)
- [ ] The sealed OOD set was opened once, in Stage G. The Gate-2 pilot used different seeds and
      is not reported
- [ ] The sealed set was never used for tuning, checkpoint selection, or any decision
- [ ] If any model receives extra input, all baselines receive the same input
- [ ] Mask seeds, corruption seeds, and noise seeds are shared across all models
- [ ] Checkpoint selection is validation F1 at k = 0 for every model, as in Paper 1; λ selection
      on validation F1 averaged over k is stated separately
- [ ] The paper states the table comes from the nominal feeder model, and backs it with the S6.2
      mismatch study
- [ ] The Y-bus is stated as used for analysis only, not inside the residual
- [ ] Any "first" claim is narrowed against Karabulut et al. (arXiv 2510.03571 — sensor
      configuration and topology generalisation for detection) and PPGN
- [ ] Pre-bugfix Paper-1 files (`*.prebugfix.*`) are never mixed with current numbers

## B. Reproducibility

- [ ] Paper 1 tables reproduce exactly (S0.1)
- [ ] Snapshot mode matches Daily mode to 1e-3 p.u. on 5 events (S0.4)
- [ ] λ₁ = λ₂ = β = 0 reproduces R0 exactly (S4.1)
- [ ] Residual layer matches the slow reference loop (S3.4)
- [ ] Injectors deterministic under a fixed seed, across models (S3.1)
- [ ] `data/ood_test.sha256` verified before the set is opened
- [ ] `14_PREREGISTRATION.md` committed before any test-split result was viewed; commit hash
      recorded in the paper

## C. Correctness traps specific to this method

- [ ] Gradient audit passed (S4.2). A reconstruction loss at the arg-min hypothesis has zero
      gradient — do not use one
- [ ] Silent injectors leave m = 1. Verify explicitly; if they zero the mask you are testing
      ordinary dropout
- [ ] Silent injectors never touch uninstrumented phases (V = 0, m = 0)
- [ ] LOO residual refits on A \ {i}, not on A — otherwise the corrupted sensor explains itself
- [ ] The gate scales the input embedding and vote weight, not messages at every hop
- [ ] The observation vector is the last timestep, de-normalised to p.u. with the loader's stored
      stats (`data/dataset.py:30–31`); the table is in p.u.
- [ ] Windows with zero post-inception samples (w = 20) are excluded from L₁/L₂
- [ ] σ₀ in the residual denominator is set to the injected noise level under the noise
      experiments, not left at 0
- [ ] The no-fault hypothesis f = 0 is in the residual (μ₀[L]); PSM can predict class 0
- [ ] The snapshot solve uses L × 0.99953 (the frozen pre-fault modulation)
- [ ] r̃ cache entries are keyed by (window, mask seed, k) — r̃(A) changes with the mask

## D. Claim discipline

- [ ] "Physics-consistent", never "PINN" or "physics-informed PINN"
- [ ] No claim of KCL, KVL, or power-flow residuals — the data cannot support them
- [ ] No claim of physics-informed temporal dynamics — OpenDSS Daily mode is quasi-static, there
      are no transients inside a window
- [ ] Resistance invariance stated as approximate, under the adopted fault model and operating
      conditions
- [ ] The signature table is disclosed as privileged model knowledge, with the mismatch study as
      evidence it degrades gracefully
- [ ] The load model is disclosed as a single global multiplier (the corpus's own), not a per-bus
      load model
- [ ] The contribution is framed as physics-consistent decision making on a spatio-temporal graph
      representation, not a new GNN layer
- [ ] If PSM ≥ STGNN on the clean corpus, say so in the abstract; the contribution is then where
      learning still helps, not "the GNN beats physics"
