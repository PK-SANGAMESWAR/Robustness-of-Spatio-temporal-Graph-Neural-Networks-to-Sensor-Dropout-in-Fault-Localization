# PC-STGNN / PCRG-STGNN — Implementation Plan (rev. 2, with author corrections 2026-09-22)

> On approval this file is saved verbatim as `PC-STGNN-PLAN.md` in the repo root and nothing
> else is done today. Execution starts tomorrow, after the author has read it.

## 0. Context

Paper 1 (dropout-aware STGNN, IEEE-123, 25 µPMUs) is a complete draft. `NEW-STRATEGY.md` is the
first single-file draft of the follow-up; `PC-STGNN/` (16 files, reviewed against the repo on
2026-09-22) supersedes it and is the specification this plan implements. The method: a
**fixed, zero-parameter physics operator** (an OpenDSS signature table μ[f,τ,R,L] over
25 locations × 11 types × 15 resistances × L load levels, plus a no-fault row μ₀[L]) turned
into a per-window, scale-free residual ρ_f that depends on the **inputs only**; it is used
(a) alone as a training-free classifier (PSM), (b) fused per node into the logits of the
*unchanged* Paper-1 backbone (PC-STGNN, with λ₁/λ₂ consistency terms), and (c) as a
leave-one-out per-sensor trust score gating silent-failure sensors (PCRG-STGNN). Three learnable
scalars in total (β, α, c). Five hard gates; the decisive one (Gate 2) is CPU-only and happens
before any GPU training.

This plan translates the 16 spec files into **concrete code, scripts, outputs and an execution
order for this repo**, reusing what exists. It does not re-argue the science.

### Verified against the code (read-only, 2026-09-22)

| Question | Finding | Consequence |
|---|---|---|
| Is the corpus load per-bus or scalar? | **Scalar.** `runner.py:116` draws one `rng.uniform(0.5,1.3)` per event; stored `meta["load_multiplier"]` is a `float`; `paper/overleaf.tex:224` says "U(0.5, 1.3) **per run**" and l.205 "differ only in the load multiplier drawn per run" — the draft on disk never says per-bus | Decision 1 (scalar L axis) is confirmed. No erratum needed for the draft on disk; grep the submitted copy once to be sure Paper 2 never repeats a per-bus claim |
| Which load-shape value applies during the fault? | The fault loop (`runner.py:138–150`) **never rescales loads**; the last rescale was pre-fault step 39, so every fault solve runs at `L × (1 + 0.003·sin(2π·39/40)) = L × 0.99953`, **not** the step-40 value 1.0 | `PREFAULT_FREEZE = 0.99953` is right for this code; `audit_snapshot_equivalence.py` asserts it explicitly (Issue 6) |
| Do the 12 dropout-aware baselines exist? | **Yes.** `results/dropout_aware/{gru,rgcn,rgsage_mean,rgsage_max,rgatv2}/seed_000{42,43,44}/metrics.json` all present; `results/dropout_study_aware/degradation_curves.csv` has `n_train_seeds=3` | S3.2: nothing to run; cite |
| `.rules/` | Deleted in `f4b7e7b`; CLAUDE.md is stale on this | The five guiding rules in `PC-STGNN/00_INDEX.md` are the operating rules for this project |
| OOD corpora location | Corpora live at repo root as `dataset_*/`; `fault_localization/data/` is code | `dataset_ood_pilot/`, `dataset_ood_test/` (not `data/ood_*`) |
| OOD seeds | `constants.py:173` `deterministic_seed()` does `FAULT_RESISTANCES.index(R)` → **raises** for R ∉ {0.1,1,10} | New `deterministic_seed_ood()` with a disjoint offset; existing seeds untouched |
| Where is the loss applied? | Per-node `expanded_cross_entropy(logits[B,N,26], y)` (`models/common.py:52`); soft vote = Σ_nodes softmax (`voting.py`) | Fusion is per-node (Issue 3 below) |
| Table-build cost | Corpus code re-compiles the feeder per event (`runner.py:118`) | S0.5 times "recompile per hypothesis" vs "compile once per worker, re-solve"; S0.4 checks the cheaper one still matches to 1e-3 p.u. |
| Training-time metadata | Batches carry `window_id` and `metadata` lists (`dataset.py:137`); the DA loader (`dropout/training.py:62`) has the clean `x` in hand before masking | w = 20 loss mask from `window_id` (training only, never inside the residual); add `batch["x_clean"]` |
| Memory | 15 GB RAM, T1000 8 GB; a full `[B, 70 125, 75]` broadcast is 670 MB at B = 32 | Residual layer chunks over f |

Machine: 24 CPU cores, T1000 8 GB, `uv run python`.

---

## 1. Decisions (confirmed by the author 2026-09-22)

1. **Operator = profiled scalar load axis L** (`01_SPEC.md` §2 as corrected). Verified against
   code and paper above. L is a profiled nuisance parameter exactly like τ and R — min over a
   grid, no ensemble, no σ_load. σ_ε is a small table-vs-corpus fidelity floor only.
   The grid step is **decided by S1.2**, not defaulted: test `--l-step 0.05` (17 pts) and
   `0.025` (33 pts, ≈ 140 k solves, fine on 24 cores); with σ_ε ≈ 1e-3 and ≈ 2–5e-3 p.u. of
   voltage change per 0.05 of load, discretisation — not fidelity — is the likely dominant error.
2. **Table built by the corpus's own Daily-mode path** (compile → 1 pre-fault step at
   L × 0.99953 → inject → 1 fault step); snapshot mode kept only as an S0.4 cross-check.
3. **Seed plan** (one run ≈ 7 h): 3 seeds for R0 (done), R2, **R4**, R5, R6, G0, G2; 1 seed for
   R1, R3, **R7**, G1, G3, B2ᵃ. (Swap vs. the spec: λ₂ is the novel loss term and R4 isolates it;
   R7 answers a one-line objection.) ≈ 33 runs ≈ 220–235 GPU-h.
4. **No GPU training before Gate 2.** Non-negotiable.
5. New results under `results/pcstgnn/<stage>/…`; the operator under `operator/`; the ρ cache
   under `cache/` (git-ignored, regenerable).

---

## 2. Method corrections that change the code (from the author's review)

### 2.1 Scale-free residual ρ — fixes the loss/fusion overflow (Issue 1)

With L profiled and σ_ε ≈ 1e-3 p.u., a wrong hypothesis gives per-channel terms ≈ (0.1/1e-3)² =
10⁴; summed over ≤ 75 channels, r_f for wrong locations is 10⁵–10⁶ while the true one is ≈ |A|.
Raw r̃ would need λ₁ ≈ 1e-5 and would saturate the softmax at any usable β. So the single scale
used everywhere is

```
r_f     = min over (τ,R,L) of Σ_{(i,p)∈A} (s_ip − μ[f,τ,R,L,i,p])² / (σ_ε² + σ₀²)     f = 0…25
ρ_f     = ( r_f − min_g r_g ) / 2            ≥ 0, scale-free, Gaussian log-likelihood units
P_phys  = softmax(−ρ)
```

`residual.py` returns ρ (and the argmin hypothesis per f). PSM = argmin_f ρ_f. Fusion, L₁, L₂,
the LOO score, the identifiability score and the metrics (ρ_true, margin ρ_(2) − ρ_(1)) all use ρ.
β = 0 still recovers the baseline exactly, so S4.1 is unaffected.

### 2.2 Fusion is per node — a product of experts — and is described as such (Issue 3)

```
z'_{n,f} = z_{n,f} − β · ρ_f          broadcast to every node n     (β learned, init 0)
Σ_n softmax(z'_n)_f ∝ Σ_n exp(z_{n,f}) · exp(−β ρ_f) / Z_n
```

Each node's distribution is reweighted multiplicatively by the same physics prior before the
soft-vote sum. This is what is implemented, and it is what `01_SPEC.md` §6 and the paper's §4
must state (not the graph-level `z' = z − βρ`). Engineering reason: the per-node CE, soft vote,
`dropout/evaluator.py` and every loader stay untouched.

```
L = CE(z', y) + λ₁ · Σ_f P(f|X)·ρ_f(A) + λ₂ · Σ_f P(f|X)·ρ_f(D)      P(f|X) = normalised soft-vote scores
```

w = 20 windows (0 post-inception samples) are excluded from the λ terms via a loss mask from
`window_id`.

### 2.3 LOO residual: subtract, **then** re-minimise (Issue 2)

```
✗ WRONG : r_f at the global argmin (τ*,R*,L*), minus channel i's term        ← this is the in-sample score
✓ RIGHT : per-channel error tensor E[B, hyp, 75]; for each sensor i zero its channels,
          THEN min over hyp → r_f(A∖{i}); ρᵢ^LOO = normalised error of sensor i at that refit
```

Cost: build `E` once per f-chunk (`[B, 2 805, 75]`), then 25 masked re-mins over the hypothesis
axis. Test: plant a corrupted sensor and assert ρᵢ^LOO ≫ ρᵢ^in-sample for it — otherwise S5.2
would compare a variant against itself.

### 2.4 Observation vector

`s_ip` = last timestep of the window, de-normalised to p.u. (default). Add `--obs-reduce
{last,last5}`; `last5` is compared to `last` **only in the noise experiments** (a single sample
is √n noisier than an average; on the clean corpus the post-fault level is flat to < 1e-6 p.u.).
Neither uses the inception offset — spec §5 holds.

### 2.5 Framing to build in from day one (Issue 4)

PSM ≈ perfect at k = 0 on the clean corpus is a **tautology** (same simulator, noise-free,
σ_ε ≈ 1e-3), not a result. Therefore:
- report PSM's clean number as an operator-fidelity sanity check and say so;
- **noise** (σ₀) and **model mismatch** (S6.2) are primary axes, not sub-tables. Physics degrades
  under noise; the network degrades under OOD R and severe dropout; the hybrid should own the
  union. The Gate-2 reframed question is "physics and learning fail under different
  perturbations; does fusing them cover both?";
- noise levels: keep the existing grid σ ∈ {5e-4, 2e-3, 1e-2} p.u. so Paper-1 curves are
  reused, and state which are realistic against µPMU accuracy classes (IEEE C37.118 1 % TVE cap
  ⇒ 1e-2 is the spec limit; at 10 Ω the sag is ≈ 0.01–0.02 p.u., so σ = 1e-2 is a ~1–2 : 1 SNR
  regime — the interesting operating point, not an extreme). Check the datasheets before the
  paper text is written.

### 2.6 Unlabelable-window ceiling (Issue 5)

w = 20 fault-labelled windows contain zero post-inception samples: physically identical to
no-fault windows but carrying a fault label. No model can beat chance on them. Stage A computes
(a) every model's F1 on that subset (≈ chance; anything better ⇒ label leakage) and (b) the
implied macro-F1 ceiling. Goes in `02_STAGE_A_audit.md` S0.3, `11_METRICS.md` §1, and reframes
every absolute number in both papers (and F-H1).

---

## 3. Code map — what gets written, where, reusing what

### New package `fault_localization/physics/`

| Module | Contents | Reuses |
|---|---|---|
| `grids.py` | `R_GRID` (15 log-spaced 0.05–30 Ω), `l_grid(step)`, `PREFAULT_FREEZE = 1 + 0.003·sin(2π·39/40)` with a docstring pointing at `runner.py:138` (loads never rescaled during the fault), axis-order constants | `config/constants.py` |
| `signature_solver.py` | `solve_signature(f, τ, R, L, mode="daily"\|"snapshot") → [25,3] p.u. + converged flag`; `solve_nofault(L)`; a `worker(chunk)` for `multiprocessing`, one OpenDSS engine per process | `simulation/feeder.compile_feeder`, `fault_injection.inject_fault/clear_all_faults`, `monitors.solve_prefault_timeseries/_scale_loads_from_base/extract_pmu_voltages_with_mask` |
| `table.py` | `build_table(out_dir, workers, mode, l_step, r_grid, master)` → `operator/mu.npy [25,11,15,L,25,3]`, `mu0.npy [L,25,3]`, `build_log.csv`, `grid.json`; `load_table()`; `--master` accepts a perturbed `.dss` for S6.2 | — |
| `observation.py` | `observe(x, mean, std, reduce="last") → (s [B,25,3] p.u., a [B,25,3])`; `observe_raw(record, step)` for the offline cache | `processed/norm_stats.json` |
| `residual.py` | Torch, no parameters: `rho(s, a, table, sigma_eps, sigma0, chunk_f=1) → ρ [B,26], hyp_star [B,26,3]` (§2.1; chunked over f); `rho_heldout(s_clean, a_clean, d_mask, hyp_star) → ρ(D) [B,26]`; `rho_loo(s, a, f_hat) → ρᵢ^LOO [B,25]` (§2.3, subtract-then-re-min) and `rho_insample(...)` for S5.2; `reference_loop()` slow NumPy version | — |
| `identifiability.py` | `separability(table, sigma_eps, sigma0, k, n_subsets) → D [26,26]` (min over (τ,R,L) pairs, in ρ units); empirical nearest-neighbour confusability | `dropout/masking.py` subset sampling |
| `cache.py` | `build_rho_cache(split)` → `cache/rho_{split}.npz` keyed by `window_id` (k = 0, clean; from `raw_sim/*.npz` step w+19 so it equals what `observe()` sees) | `simulation/runner.load_event` |

### Other new / touched modules

| Module | Contents |
|---|---|
| `fault_localization/dropout/silent.py` | `apply_silent_failure_batch(x, s, kind, severity, seed, norm_stats)` — kinds `stuck` (freeze at the window's first timestep), `bias` (±2 %, ±5 % p.u.), `gauss` (σ = 0.02 p.u.), `swap` (permute the node's instrumented phases); leaves m = 1; never touches m = 0 phases; de-normalise → perturb → re-normalise. Seed derivation `seed·10000 + batch_idx` as `masking.py`. |
| `fault_localization/models/pc_stgnn.py` | `PCSTGNN(RGSAGE)` — subclass exactly like `models/pignn.py:PIGNNSage`. Buffers μ, μ₀, σ_ε, mean, std (never `requires_grad`). Params: `beta` (init 0; `use_fusion`), optional `type_head` (R1), optional parameter-matched `decoder` on D (R2), `shuffle_map` (R6: fixed seeded permutation of the location axis). `forward(x)` → node logits with the per-node fusion of §2.2; `forward_pc(x, x_clean=None, drop_mask=None)` → `{logits, p_graph, rho_A, rho_D, hyp_star}`. All switches off ⇒ byte-identical to `RGSAGE(aggregator="max")`. |
| `fault_localization/models/pcrg_stgnn.py` | `PCRGSTGNN(PCSTGNN)` — pass 1 → f̂ → `rho_loo` → `g_i = m_i·σ(−α(ρᵢ^LOO − c))` (α, c learnable) → pass 2 with the **input embedding** and **soft-vote weight** scaled by g_i (G2). `gate_every_hop=True` for G3. |
| `fault_localization/models/common.py` | `pc_loss(out, y, lam1, lam2, loss_mask)` per §2.2. |
| `fault_localization/experiments/runner.py`, `train.py` | `_model()` names `pc_stgnn`, `pcrg_stgnn`; `physics_operator:` config block (table path, σ_ε, λ₁, λ₂, fusion, shuffle, type_head, decoder, gate, obs_reduce); `train_epoch(pc=...)` hook mirroring the existing `physics=` hook (reads `x_clean`, `dropout_nodes`, `window_id`); logs `train_l1`, `train_l2`, `beta`, `alpha`, `c` per epoch; corruption-augmentation loader for G0–G3 wrapped like `DropoutAugmentedLoader`. |
| `fault_localization/dropout/training.py` | `batch["x_clean"] = x` before masking (one line; nothing existing reads it). |
| `fault_localization/config/constants.py` | `deterministic_seed_ood(...)` — separate hash with offset ≥ 10⁹, provably disjoint from every corpus seed; `deterministic_seed` untouched. |
| `config/pc_stgnn.yaml`, `config/pcrg_stgnn.yaml`, `config/pcstgnn_ablations/{R1..R7,B2a,G0..G3}.yaml` | `base: dropout_aware/rgsage_max.yaml` + the switch map. |

### New scripts (`scripts/`, all `uv run python scripts/<x>.py`, resumable, JSON + CSV out, `--max-batches`/`--epochs` smoke flags on anything long)

| Stage | Script | Output |
|---|---|---|
| A | `audit_load_model.py` | `results/pcstgnn/audit/load_model.md` — the code-vs-paper check written down (scalar field over all 41,250 `.npz`; the two `overleaf.tex` lines) |
| A | `audit_paper1_repro.py` | `audit/paper1_repro.csv` (re-runs the `run_dropout_study.py` logic on the checkpoint sets and diffs against the on-disk CSVs) |
| A | `audit_metadata.py` | `audit/metadata_coverage.md` (7 meta keys; regenerates 5 events bit-for-bit) |
| A | `audit_window_census.py` | `audit/window_census.csv` + F-A1; within-window post-fault spread on 100 events; **unlabelable subset F1 per existing model + implied ceiling** (§2.6) |
| A | `audit_snapshot_equivalence.py` | `audit/snapshot_equivalence.csv` (5 events × {daily-path, snapshot, reused-circuit}); **asserts** the corpus fault steps sit at `L×0.99953` (recomputes the pre-fault step-39 load from `meta` and compares); sets σ_ε |
| A | `audit_timing.py` | `audit/timing.md` (100 solves per mode, 100 Daily events) |
| B | `build_signature_table.py` | `operator/` + T3; `--r-grid`, `--l-step`, `--mode`, `--workers`, `--master` |
| B | `check_operator_fidelity.py --checks {lgrid,bias,load,floor,rgrid}` | S1.2 (F-B2, decides the L step), S1.4 (F-B3, T4), S1.5 L*/τ* recovery (F-B4), S1.6 floor (F-B5, median ρ-scale sanity), S1.7 (F-B6) |
| B | `build_network_matrices.py` | S1.1 Y-bus / \|Z_pf\|/\|Z_ff\| map (F-B1); analysis only |
| C | `generate_ood_set.py --set {pilot,test}` | `dataset_ood_pilot/`, `dataset_ood_test/` + `dataset_ood_test.sha256`; **copies** `norm_stats.json`, `edge_index.npy`, `node_mapping.json` from the frozen corpus; single `test.txt` split |
| C | `build_rho_cache.py` | `cache/rho_{train,val,test,ood_pilot}.npz` |
| C | `run_identifiability.py` | F-C1, F-C2, F-C3 |
| C | `run_psm_baselines.py` | B1 (ΔV nearest signature), B2 (PSM) at every k / mask seed / σ₀ / `--obs-reduce`, by R and type, top-3, ρ rank, margin, AUDC rel + abs; `--data-root dataset_ood_pilot` for S2.4-OOD → T6 rows |
| C | `run_ood_contrast.py` | S2.5 (F-C4, internal) |
| C | `run_inference_fusion.py` | S2.6: β grid on validation, clean and σ₀ ∈ {5e-4, 2e-3, 1e-2}, ΔF1 at k = 0/10 and 10 Ω per seed |
| D | `run_physics_impute_baseline.py` | S3.3 row 2 |
| D | `bench_residual_layer.py` | S3.4: reference-loop equality (100 samples) + ms/batch |
| E | `run_pcstgnn_null_test.py` | S4.1 (seed 42, all off; epoch-by-epoch diff vs `results/dropout_aware/rgsage_max/seed_00042/metrics_by_epoch.csv`) |
| E | `audit_gradients.py` | S4.2 finite-difference audit |
| E | `run_pcstgnn_tuning.py` | S4.3 λ sweep, 15 epochs, seed 42, val F1 averaged over k (F-E1) |
| E | `run_pcstgnn_ablation.py --arms --seeds` | S4.4 R1–R7 + B2ᵃ → T7 |
| F | `run_silent_detection.py` | S5.1/S5.2 AUROC, AUPRC, TPR@FPR=0.05, LOO vs in-sample (F-F1…F-F3) |
| F | `run_gate_study.py --arms --seeds` | S5.3 G0–G3 (F-F4) |
| G | `run_master_sweep.py` | S6.1 every new model × condition (Paper-1 CSVs reused); opens the sealed set once (hash + date → `14_PREREGISTRATION.md`); **adds a ρ-at-predicted-location column for every model, physics or not** (from the cache) → T5, T8, T9, T10, F-G1…F-G5 |
| G | `run_model_mismatch.py` | S6.2 perturbed tables → T11, F-G6 |
| G | extend `run_significance_tests.py`, `run_runtime_benchmark.py` | T13, T12 (+ residual / LOO / second-pass timings) |
| H | `build_error_attribution.py`, `build_case_studies.py` | F-H1, F-H2 |

### Tests (`tests/`)

`test_physics_residual.py` (reference loop = torch layer; dropped channels contribute 0; f = 0
row present; ρ ≥ 0 with exactly one zero; argmin recovers a planted hypothesis; **planted
corrupted sensor ⇒ ρᵢ^LOO ≫ ρᵢ^in-sample**), `test_signature_solver.py` (one solve converges;
shape; mask pattern equals the corpus's), `test_silent.py` (deterministic; k = 0/no-corruption
byte-identical; m stays 1; m = 0 phases untouched), `test_pc_stgnn.py` (all-off ≡ RGSAGE
byte-for-byte — copy `tests/test_pignn.py`; β/α/c receive gradients; μ receives none; shuffle
map is a permutation), `test_ood_seed.py` (no collision with any corpus seed).

---

## 4. Spec files to update **before** execution starts (first task tomorrow, after §1 is confirmed)

| File | Change |
|---|---|
| `01_SPEC.md` §2 | Scalar L axis (step decided by S1.2) replaces the ensemble; σ_load → σ_ε; μ₀[L] no-fault row (already there — keep) |
| `01_SPEC.md` §4 | Shifted/halved residual ρ_f; P_phys = softmax(−ρ) |
| `01_SPEC.md` §6 | Fusion stated as per-node broadcast (product of experts); L₁, L₂ on ρ |
| `01_SPEC.md` §7 | LOO defined as subtract-then-re-minimise |
| `02_STAGE_A_audit.md` S0.3 | Unlabelable-window ceiling analysis; S0.4 freeze assertion; new S0.0 load-model check |
| `03_STAGE_B_operator.md` S1.2/S1.5 | L-step decided by convergence (0.05 vs 0.025); F-B4 = L*/τ* recovery |
| `11_METRICS.md` §1, §2, §4 | Ceiling metric; ρ_true and margin ρ_(2) − ρ_(1); σ_ε replaces dispersion metrics |
| `06_STAGE_E_pcstgnn.md`, `13_BUDGET_TIMELINE.md` | Seed swap R4 ↔ R7 |
| `04_STAGE_C_gonogo.md`, `09_STAGE_H_writing.md` | Framing of §2.5: PSM-clean as sanity check; noise + mismatch as primary axes |

Everything else in the 16 files stays as is (all five gates, ablation logic, baseline ladder,
leakage checklist, fallback map).

---

## 5. Execution order (what to run, in what order, with the gate after each)

### Stage A — audit (week 1, CPU + inference) → **Gate 0**
0. `audit_load_model.py` — settle scalar-vs-per-bus in writing (already verified; this makes it
   an artefact).
1. `audit_paper1_repro.py` — must match `results/fullscale_final/benchmark_summary.csv` and
   `results/dropout_study{,_aware,_sdr}/degradation_curves.csv` to printed precision. *Stop the
   project if not.*
2. `audit_metadata.py`, `audit_window_census.py` (expected 5 % / 25 %; spread < 1e-6 p.u.;
   ceiling numbers).
3. `audit_snapshot_equivalence.py` on 5 events (0.1/1/10 Ω, ≥ 3 types): Daily-path signature
   vs corpus post-fault level < 1e-3 p.u.; freeze assertion; record σ_ε; pick the cheapest
   faithful solve mode.
4. `audit_timing.py` → rescale `13_BUDGET_TIMELINE.md`.
5. Deliverable: `results/pcstgnn/audit/AUDIT-NOTE.md` (one page).

### Stage B — operator (weeks 2–4, CPU) → **Gate 1**
1. `build_network_matrices.py` (S1.1, analysis only).
2. `check_operator_fidelity.py --checks lgrid` on 3 (f,τ,R) at steps 0.1/0.05/0.025 → **choose
   the L step** (pass: interpolation error < 1e-3 p.u. at the chosen step).
3. `build_signature_table.py --workers 20 --l-step <chosen>` → `operator/` (non-converged
   solves logged and re-solved; monotonic sag vs R check).
4. `check_operator_fidelity.py --checks bias load floor rgrid` on the **training split** only.
   Pass: median |e|/sag < 0.3 at 10 Ω; L* within one step and τ* correct on > 90 %; ρ-scale
   sanity; halving the R grid costs < 0.01 F1.
5. Fallback after all branches: voltage-angle route (`monitors.py:78–80` already computes
   angles; seeds stored) — a fork taken now, not at week 16.

### Stage C — training-free science (weeks 4–6, CPU + inference) → **Gate 2**
1. `generate_ood_set.py --set pilot` (R ∈ {5,20} × 25 × 11 × 2 runs, run_ids 1000–1001) and
   `--set test` (R ∈ {3,5,20,30} × 25 × 11 × 10 runs, run_ids 2000–2009; ≈ 4–5 h); seal
   (`chmod -R a-w`, `sha256`, log in `14_PREREGISTRATION.md` §6).
2. `build_rho_cache.py` for train/val/test/ood_pilot.
3. `run_identifiability.py` (S2.1 + S2.2; diagnosis table in `04_STAGE_C_gonogo.md`).
4. `run_psm_baselines.py` — validation + pilot for the decision; test only for the T6 rows the
   pre-registration commits to reporting regardless. PSM-clean is reported as a fidelity check.
5. `run_ood_contrast.py` (S2.5) and `run_inference_fusion.py` (S2.6) — validation + pilot only;
   the noise arm of S2.6 is the one Paper 1 could not answer.
6. Fill in `14_PREREGISTRATION.md`, commit, record the hash (S2.7).
7. Apply the Gate-2 table; expected branch: "PSM ≥ STGNN clean" → GO with the §2.5 framing.

### Stage D — infrastructure (weeks 5–9, overlaps C; CPU) → **Gate 3**
`dropout/silent.py` + tests; `residual.py` benchmark (< 5 ms/batch of 32 or cached);
`run_physics_impute_baseline.py`; `PCSTGNN`; `x_clean`; `pc_loss`; configs. S3.2 is done.

### Stage E — PC-STGNN (weeks 8–13, GPU) → **Gate 4**
1. `run_pcstgnn_null_test.py` — must reproduce the R0 seed-42 curve epoch by epoch.
2. `audit_gradients.py` — ∂L₁/∂w ≠ 0, ∂L₂/∂w ≠ 0, β gets a gradient, μ gets none.
3. `run_pcstgnn_tuning.py` (λ₁ ∈ {0.1, 0.3, 1} then λ₂ ∈ {0.1, 1}; 15 epochs; ≈ 21 GPU-h) —
   with ρ scale-free these λ ranges are O(1)-meaningful.
4. `run_pcstgnn_ablation.py --arms R2 R4 R5 R6 --seeds 42 43 44`, then
   `--arms R1 R3 R7 B2a --seeds 42` (background, one run at a time on the 8 GB card).
5. Gate 4: R5 > R0 on validation in the pre-registered conditions, sign-consistent. If
   R5 ≈ R0 → skip Stage F training, go to G, write the "physics buys generalisation" paper.

### Stage F — trust gate (weeks 12–16) → **Gate 5**
1. `run_silent_detection.py` (training-free; AUROC ≥ 0.75 at ±2 % bias or PCRG is not trained
   and the AUROC-vs-severity curve is a bounded negative).
2. `run_gate_study.py --arms G0 G2 --seeds 42 43 44`, then `--arms G1 G3 --seeds 42`.

### Stage G — evaluation (weeks 15–19)
`run_master_sweep.py` (sealed set opened once; ρ column for all models),
`run_model_mismatch.py` (3–5 perturbed tables + one R5 retrain at ±5 %), statistics and cost.
Noise (F-G4) and mismatch (F-G6) are headline figures alongside OOD (F-G3).

### Stage H — writing (weeks 19–26)
Error attribution (with the ceiling from §2.6), case studies, paper from the Paper-1 skeleton
(edit, don't restructure), release.

---

## 6. Non-negotiables carried into every script (from `12_CHECKLISTS.md`)

- The residual reads **only** the `[25,20,6]` tensor (last timestep(s) + mask) and the fixed
  table. `window_id` is used for the L₁/L₂ loss mask at training time and for nothing else.
- True R, τ, L are used only in S1.4/S1.5 diagnostics.
- Mask seeds `mask_seed·10000 + batch_idx`, corruption seeds and noise seeds shared across
  models (reuse `dropout/evaluator.py`, `dropout/noise.py` unchanged).
- Checkpoint selection = validation graph-F1 at k = 0; λ selection = validation F1 averaged
  over k; both stated.
- σ₀ in the residual denominator equals the injected noise level in the noise experiments.
- Sealed OOD set opened once, in Stage G, after `sha256` verification; the pilot is never
  reported.
- Never mix `*.prebugfix.*` files with current numbers. Never fabricate a number.
- "Physics-consistent", never "PINN"; no KCL/KVL/power-flow-residual claims; the table is
  disclosed as nominal-model knowledge and backed by S6.2; the fusion is described as the
  per-node product of experts it is.

## 7. Verification

- **Unit**: the five new test files + the existing suite (`uv run pytest`) green.
- **Gate 0**: `paper1_repro.csv` diff = 0 at printed precision; `snapshot_equivalence.csv`
  max error < 1e-3 p.u. on all 5 events and the freeze assertion passes.
- **Gate 1**: the fidelity summary JSON from `check_operator_fidelity.py`; chosen L step
  recorded in `operator/grid.json`.
- **Gate 2**: S2.6 ΔF1 per seed (clean + 3 noise levels) and the S2.5 pilot contrast;
  pre-registration committed.
- **Gate 3**: `bench_residual_layer.py` reports `max_abs_diff_vs_reference` ≈ 0, the LOO ≫
  in-sample assertion, and ms/batch.
- **Gate 4**: null-test epoch-by-epoch diff = 0; gradient audit; R5 vs R0 sign table.
- **Gate 5**: AUROC table; G2 vs G0 sign table; false-suppression rate.
- Smoke every training script on the GPU with `--epochs 1 --max-batches 50` before any 7-h run.

## 8. First actions tomorrow

1. Save this file as `PC-STGNN-PLAN.md` (done on approval) and apply the §4 spec-file edits.
2. Create the `fault_localization/physics/` skeleton, `dropout/silent.py`, and the five test
   files (no results yet).
3. Run Stage A scripts 0–4 (all cheap) and write `AUDIT-NOTE.md`.
4. Do **not** launch any GPU training.
