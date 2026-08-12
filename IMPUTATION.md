# IMPUTATION.md — Imputation baseline for the sensor-dropout study

**Status:** experiment running (standard-trained RGSAGE-Max, seeds 42/43/44).
Methodology + fairness are final; the results table is filled once the seed-averaged
run completes. **No existing result, checkpoint, or CSV is modified** — this only
*adds* one comparison.

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

Seed-averaged over 42/43/44, three mask realizations each. **Imputation column pending
run completion** — partial seed-42 values shown in _italics_ as an interim indication;
final seed-averaged numbers replace this table when the run finishes.

| Variant | $k{=}0$ | $k{=}2$ | $k{=}5$ | $k{=}10$ | AUDC | Ret@10 |
|---|---|---|---|---|---|---|
| Standard training (zero) | 0.846 | 0.570 | 0.267 | 0.059 | 0.414 | 7.0% |
| **Mean-imputation (standard)** | _0.853*_ | _0.631*_ | _pending_ | _pending_ | _pending_ | _pending_ |
| Dropout-aware training | 0.839 | 0.816 | 0.778 | 0.676 | **0.916** | **80.6%** |
| SDR-STGNN (gated) | 0.839 | 0.812 | 0.758 | 0.643 | 0.895 | 76.7% |

_*seed-42 only, interim._

**Interim read:** imputation helps modestly over zeroing at moderate dropout (seed-42
$k{=}2$: 0.631 vs 0.537) but stays far below dropout-aware training (0.816 at $k{=}2$).
If this holds seed-averaged, the claim is: *naïve graph imputation does not recover the
lost accuracy — robustness must come from training, not from patching the input.*

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
