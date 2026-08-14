# IMPUTATION.md — Imputation baseline for the sensor-dropout study

**Status:** ✅ complete (standard-trained RGSAGE-Max, seeds 42/43/44). Wired into the
paper (Table `tab:aware` row + §Effect-of-Dropout-Aware-Training paragraph + limitation
rewrite). **No existing result, checkpoint, or CSV was modified** — this only *adds*
one comparison.

---

## 1. What this is

A **reviewer-requested imputation baseline** for the sensor-dropout study. When a
μPMU drops out at inference we must feed the model *something* for that node. The
paper's degradation study **zeroes** the dropped node (voltages + availability mask →
0, i.e. "this node is blank"). The natural alternative a reviewer will ask about is
**impute-then-infer**: *guess* the missing readings and present them to the model as
if observed.

This baseline implements **graph neighbour-mean imputation**: for each dropped node,
its three voltage channels are filled with the **per-timestep mean of its still-alive
graph neighbours**, then the **standard-trained** RGSAGE-Max is run on the
reconstructed input. It is the lightweight, no-training analogue of the Dirichlet /
GDIA imputation of **Zhang et al. (2025)** — one of our cited related works — so it
directly answers "why didn't you just impute?".

> Note: this is an **inference-time** method on the *standard-trained* model. It is
> **not** a new training scheme. It is added as a comparison *against* our
> dropout-aware **training** contribution, to show that training beats patching the
> input at test time.

## 2. Why we added it

Both the reviewer notes and the supervisor file flagged the same top acceptance risk:

> *"No imputation baseline — Zhang 2025 / Fan 2024 are your nearest neighbours and you
> decline to compare. Current 'left to future work' reads as avoidance."*

Adding one row to Table `tab:aware` converts that gap into a **defended design
choice**: on the *same* backbone and *same* dropout sweep we now compare four ways to
cope with sensor loss — zeroing, imputation, dropout-aware training, and the
availability-gated SDR-STGNN.

## 3. Method (exactly what the code does)

For a dropped node $j$ in window $i$:

- **Fill:** $x_{i,j,\,0:3} \leftarrow \operatorname{mean}_{m \in \mathcal N(j),\ a_m=1}\ x_{i,m,\,0:3}$
  (per timestep, per phase), where $\mathcal N(j)$ are $j$'s **undirected 1-hop**
  graph neighbours and $a_m$ marks nodes that are still available.
- **Present as observed:** $x_{i,j,\,3:6} \leftarrow 1$ (availability mask = 1). This
  is the standard impute-then-infer convention and the **generous** choice — it gives
  imputation its best shot, so the comparison can't be accused of sandbagging it.
- **Fallback:** if a dropped node's whole neighbourhood is also dropped (rare, low
  degree in the 24-edge tree), it falls back to 0.
- Implemented as batched matmuls over the $25\times25$ adjacency (no Python inner loop
  in the hot path).

**Design choices to state in the paper (per reviewer):**
1. Imputed nodes are presented as *observed* (mask = 1) — the most favourable setting
   for imputation.
2. Imputation uses **undirected 1-hop** neighbours as a preprocessing step, independent
   of whether an operator aggregates directionally.

## 4. Fairness — proven, not assumed

The whole comparison is valid only if imputation drops **the same sensors** as the
zeroing study. It does. The node selection reuses the evaluator's seeding
(`mask_seed = 10000*k + offset`, per-sample `rng.sample`, batch 32, `shuffle=False`),
and this was verified **byte-for-byte** against `apply_sensor_dropout_batch` for every
`(k, offset)`:

```
k=2 off=0 first 5 samples  zeroing: [[11,13],[9,23],[4,12],[7,18],[3,24]]
k=2 off=0 first 5 samples  impute : [[11,13],[9,23],[4,12],[7,18],[3,24]]
ALL (k,offset) dropped-sets identical: True
```

So the **only** difference between the zeroing run and this one is the fill
(neighbour-mean vs zero). The validated `masking.py` / `evaluator.py` were left
untouched; the selection equivalence is documented in the script.

**Sanity check:** F1 at $k=0$ must equal the standard clean number. It does — seed 42
returns **0.8526** (its exact recorded clean F1), confirming the loader/checkpoint path
is correct.

## 5. Files

- **Script:** `scripts/run_imputation_baseline.py` — self-contained; loads frozen
  standard-trained checkpoints from `results/fullscale_final/rgsage_max/seed_000{42,43,44}/`,
  runs the impute-then-infer sweep, writes `results/dropout_study_impute/imputation_summary.json`.
- **Output:** `results/dropout_study_impute/imputation_summary.json`
  (per-seed and seed-averaged F1 by $k$, AUDC, Ret@10).

**Reproduce:**
```bash
uv run python scripts/run_imputation_baseline.py \
    --results-dir results/fullscale_final \
    --data-root dataset_fullscale_50 \
    --seeds 42 43 44 \
    --output-dir results/dropout_study_impute
```

## 6. Results (RGSAGE-Max backbone, soft-voted macro-F1)

Seed-averaged over 42/43/44 (mean ± std, sample std / ddof=1), three mask realizations
each. **Final — post mask-diversity bugfix; these match `tab:aware` in the paper.**

| Variant | $k{=}0$ | $k{=}1$ | $k{=}2$ | $k{=}5$ | $k{=}10$ | AUDC | Ret@10 |
|---|---|---|---|---|---|---|---|
| Standard training (zero) | 0.846 | 0.736 | 0.590 | 0.236 | 0.048 | 0.402 | 5.7% |
| **Mean-imputation (standard)** | **0.846**±0.007 | **0.739**±0.017 | **0.637**±0.028 | **0.336**±0.043 | **0.099**±0.033 | **0.476**±0.042 | **11.8%**±4.0% |
| Dropout-aware training | 0.839 | 0.827 | 0.815 | 0.774 | 0.674 | 0.913 | 80.3% |
| SDR-STGNN (gated) | 0.839 | 0.827 | 0.813 | 0.758 | 0.631 | 0.892 | 75.2% |

At $k{=}0$ imputation is a no-op, so its clean column is the standard row's number
(identical per-seed values, hence an identical ±0.007).

**Read:** graph mean-imputation gives a *small* gain over zeroing (AUDC 0.402 → 0.476,
Ret@10 5.7% → 11.8%) but stays **far below dropout-aware training** (AUDC 0.913,
Ret@10 80.3%). It helps most at moderate dropout (k=2: 0.637 vs 0.590; k=5: 0.336 vs
0.236) and collapses with zeroing at the extreme (k=10: 0.099 vs 0.048); the gain over
zeroing at k=10 is **not significant** (+0.051, p=0.102). Conclusion:
*naive graph imputation does not recover the lost accuracy — the reconstructed values
are still fed to a model never trained to distrust them, so robustness must come from
training, not from patching the input at test time.*

## 7. What it changes in the paper

- **Adds one row** ("Mean-imputation + standard training") to Table `tab:aware`.
- **Rewrites the "No imputation baseline" limitation** (Discussion) from an
  unexplored gap into a reported comparison; learned/iterative imputation (full GDIA)
  can be named as the remaining future-work item.
- Adds the two design-choice sentences from §3 to the methods/results text.
- Nothing else changes — all existing tables, figures, and numbers are untouched.

## 8. Honest framing (do not overclaim)

Report faithfully, consistent with the project's rule set: this is a **single, simple**
imputation baseline (neighbour-mean), not a reimplementation of a learned imputation
model. It is the generous, fair, lightweight comparison that neutralizes the obvious
reviewer objection; it does **not** claim to be the strongest possible imputation
scheme. The conclusion it supports is that **dropout-aware training** is the more
effective and practical mitigation.
