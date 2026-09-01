# FaultQuery-STGNN — architecture, rationale, and what makes it new

**Status (2026-08-21):** implemented, 15/15 unit tests passing, parameter budget
measured. Not yet trained — the stage-3 pilot is chained behind the tier-3
retrain queue. **No accuracy number for this model exists yet.** Every result
quoted below is from an *existing* model in `results/`, and is there to justify a
design choice, never to claim a FaultQuery outcome.

Figure: `paper/figures/ppt_faultquery_architecture.png`
(built by `scripts/build_faultquery_figure.py`).
Plan: `~/.claude/plans/zesty-tickling-metcalfe.md`.

---

## 1. One paragraph

FaultQuery-STGNN keeps the GRU temporal encoder and the four-hop RGSAGE-Max
spatial block of our strongest baseline, and replaces the per-node classifier +
soft-voting decoder with **cross-attention over 26 learned fault-bus queries**
(25 candidate fault buses + no-fault). Each query attends over the 25 sensor
embeddings, and the attention logit is biased by three physically meaningful
terms: **hop distance** from that candidate fault to that sensor over the full
123-bus feeder, a learned per-sensor **reliability**, and a hard **availability**
mask that gives a dropped sensor exactly zero weight. The decoder emits the 26
class logits directly — there is no voting step.

---

## 2. Why the existing architecture needed replacing

The guide's criticism was that SDR-STGNN is not a new architecture. That is
correct, and our own results say so:

| | SDR-STGNN |
|---|---|
| What it changes | availability-weighted aggregation in RGSAGE-Max |
| Extra parameters | **0** |
| Behaviour at k=0 | provably identical to RGSAGE-Max (`max abs diff = 0.0`) |
| AUDC | 0.892 vs dropout-aware training's 0.913 |
| Statistical verdict | **matches, does not beat** (p = 0.098) |
| Inference cost | **+18% latency** for no robustness gain |

So the real contribution of the paper to date is *dropout-aware training*, a
training method. The architecture contribution is a negative result.

But the baselines have three specific, measurable weaknesses that live in the
**decoder**, not the encoder — and no amount of re-gating the message passing
touches any of them.

### 2.1 The model has no representation of what it predicts

The measured-only graph has 25 nodes, one per μPMU. There are 25 candidate fault
buses. **Only 6 of them are instrumented**, so **19 fault locations are not nodes
of the graph at all.** In the baselines a class is never an object — it is only a
column index in a per-sensor logit vector, and the final answer is produced by
summing 25 independent softmaxes.

### 2.2 The receptive field is smaller than the graph

The measured-only graph is a tree of 25 nodes and 24 edges with **diameter 10**.
The frozen protocol stacks **4 hops**. Roughly **44% of node pairs cannot exchange
information at all**, at any depth of the network as configured. Soft voting
cannot repair this: it pools *predictions*, not evidence, so a node that never
saw the relevant sensor still votes from ignorance.

### 2.3 Proximity — which the study proved is decisive — is never made available

`results/dropout_study_targeted/` shows that dropping the k μPMUs **nearest the
fault** is far more damaging than dropping k at random:

| | k=1 | k=2 |
|---|---|---|
| random dropout | 0.736 ± 0.013 | 0.590 |
| targeted (nearest) | **0.448 ± 0.039** | 0.361 |
| p (paired, n=3) | 0.010 | 0.017 |

A 3.6× larger drop from clean. So the sensors near a fault carry its signature —
yet nothing in the model knows which sensors those are. The measured-only graph
construction walked the real feeder to derive its 24 edges and then **discarded
the geometry it walked over**.

---

## 3. The architecture, layer by layer

```
x  [B, 25, 20, 6]                 3 RMS voltages + 3 availability-mask channels
 │
 ├─ TemporalEncoder — shared GRU-128                        52,224 params
 │     [B,25,20,6] → [B,25,128]                             UNCHANGED
 │
 ├─ Local spatial block — RGSAGE-Max × 4 hops, UNGATED      41,216 params
 │     h′ = ReLU( W [ h ‖ max_{i∈N(j)} h_i ] )
 │     Linear 256→64, then 128→64 ×3
 │     [B,25,128] → [B,25,64]                               UNCHANGED
 │
 ├─ FAULT-QUERY CROSS-ATTENTION DECODER                     ← the new part
 │     Q [26, 64] learned queries                            1,664 params
 │     W_Q, W_K, W_V  64→64                                  4,160 each
 │
 │     logit(q,i) =  (W_Q q_q)·(W_K h_i)/√d_head       content
 │                 −  softplus(α_h) · D[q,i]           physics prior
 │                 +  β · log r_i                      reliability
 │                 ←  MASK_VALUE  if sensor i is dark  availability
 │
 │     A = softmax_i(logit)        [B, 4, 26, 25]
 │     ctx_q = Σ_i A_{q,i} · W_V h_i
 │
 └─ z_q = MLP( LayerNorm[ ctx_q ‖ q_q ] )                    8,321 + 256 params
       one scalar per query  →  [B, 26]  →  [B, 1, 26]
```

Total **118,279** parameters vs RGSAGE-Max's **95,258** (**+24.2%**).

### The output shape is a deliberate engineering decision

The decoder produces one graph-level prediction, but returns it as `[B, 1, 26]`
rather than `[B, 26]`. With `N = 1`:

- `soft_vote` (sum over the node axis, then argmax) becomes the **identity**;
- `expanded_cross_entropy` (expand the event label to N nodes) becomes **ordinary
  cross-entropy**.

So FaultQuery-STGNN is a drop-in for `train.py`, `evaluate.py`,
`dropout/evaluator.py`, `run_dropout_study.py`, the targeted / persistent /
noise / correlated studies and the significance tests — **with no changes to any
of them**. It is therefore automatically paired with every existing model under
identical mask seeds, which is what makes the comparison fair rather than merely
similar.

One consequence to report honestly: the per-window node diagnostic degenerates to
the graph metric for this model, so `test_node_diagnostic.f1` is **N/A**, not a
node score.

---

## 4. The four attention terms — why each one exists

### 4.1 Content — `(W_Q q)·(W_K h_i)/√d`

Standard scaled dot-product. Gives each of the 26 classes a **learned embedding**,
so a class becomes an object the network can reason about rather than a logit
column. And because every query sees all 25 sensors in one step, the decoder's
receptive field is **global** — which is what addresses §2.2 without stacking
more hops (more hops would only add over-smoothing on a 25-node tree).

**Ablation:** none — removing this is removing the decoder, which is exactly the
RGSAGE-Max + soft-vote baseline we already have at 3 seeds.

### 4.2 Physics prior — `− softplus(α_h) · D[q,i]`

`D` is the `[26, 25]` matrix of hop distances from each candidate fault bus to
each μPMU bus, computed over the **full 123-bus topology** (range **0–23** hops,
mean 9.6). This is the geometry §2.3 says was thrown away, handed back to the
model as an attention bias — "prefer sensors near this candidate fault, unless
the content says otherwise."

Design details that matter:
- `softplus(α)` keeps the term a **penalty**, never a reward for being far away.
- `α` is learned **per head**, so different heads can settle on different
  locality scales instead of one global radius being imposed.
- Initialized small (0.1) so training starts near unbiased attention and the
  prior is something the model can *reject* if it is unhelpful.
- Row 0 (no-fault) is all zeros — a constant row cancels in the softmax, so the
  no-fault query gets no geometric prior rather than an arbitrary one.

**Ablation A2** (`α ≡ 0`): identical parameter count.

### 4.3 Reliability — `+ β · log r_i`, with `r_i = sigmoid(MLP(h_i))`

This targets the one stressor where dropout-aware training provably does nothing.
From `results/dropout_study_noise/`, standard RGSAGE-Max at k=0:

| noise σ (p.u.) | macro-F1 |
|---|---|
| 0 | 0.846 |
| 5e-4 | 0.786 |
| 2e-3 | 0.579 |
| 1e-2 (IEEE C37.118 1% TVE cap) | **0.345** |

Dropout-aware training is **not better at any σ** (−0.008 to −0.034, all CIs span
zero, p ≥ 0.107) — a clean null. The mechanism is well understood: **noise leaves
the availability mask at 1**, so the "(0,0,0) means dark" response that
dropout-aware training learns has nothing to fire on. A binary availability gate
has the same blind spot by construction.

A continuous `r_i ∈ (0,1)` can in principle express "present but not to be
trusted", which binary availability cannot.

**Ablation A4** (`β ≡ 0`): identical parameter count.

### 4.4 Availability — overwrite with `MASK_VALUE` if sensor i is dark

A dropped sensor gets **exactly zero** attention weight (asserted in the unit
tests, not assumed). `MASK_VALUE = -1e9` rather than `-inf`: it underflows to
exactly 0 in float32 whenever any sensor is live, but keeps the softmax defined
if *every* sensor is dropped, where the row degenerates to uniform instead of
producing NaN.

**The placement is the whole point.** This mask is applied **only in the decoder**;
the local spatial block stays ungated. SDR-STGNN gated at every hop, which
silenced a dropped node permanently and blocked multi-hop routing *through* it —
information that could have been reconstructed from its other neighbours never
arrived. Here a dropped node is rebuilt by message passing first, and the decoder
then decides how much to trust the result.

**Ablation A3** (no mask): identical parameter count.

---

## 5. What is new — stated precisely

Being exact here matters, because overclaiming is what the guide objected to.

**What is NOT new (and we say so):**
- Cross-attention decoders with learned queries — DETR-style object queries.
- Distance/relative-position bias in attention — standard in graph transformers.
- Availability gating — our own SDR-STGNN, and prior masked-GNN work.
- The encoder and the spatial block — unchanged from the base paper's RGSAGE.

**What is new:**

1. **Posing fault localization as query decoding over candidate fault buses.** To
   our knowledge no prior fault-localization work gives the *fault locations*
   their own representations. It matters here precisely because 19 of 25 are not
   nodes of the graph, so they otherwise have no representation at all.
2. **Using the discarded feeder geometry as the attention prior.** The bias is
   not a learned positional embedding — it is the **actual hop distance on the
   physical 123-bus feeder**, i.e. domain knowledge the graph construction had
   thrown away, reintroduced where it can act.
3. **Separating where availability is enforced from where messages are passed.**
   This is a direct, falsifiable response to a measured negative result of ours.
   We are not proposing a new gate; we are proposing the old gate was in the
   wrong place, and we have the failed experiment that motivates it.
4. **Removing soft voting.** Pooling 25 independent per-sensor softmaxes is
   inherited from the base paper and never justified. The decoder replaces it
   with attention that is *learned*, *availability-aware*, and *interpretable*.

**Bonus property:** the attention matrix `[B, 4, 26, 25]` is a direct statement of
which sensors the model used for which candidate fault. It can be plotted against
the feeder topology — the baselines offer no equivalent.

---

## 6. Comparison with the existing models

| | GRU | RGCN / RGSAGE / RGATv2 | SDR-STGNN | **FaultQuery-STGNN** |
|---|---|---|---|---|
| Temporal encoder | GRU-128 | GRU-128 | GRU-128 | GRU-128 (same) |
| Spatial block | none | message passing | **gated** message passing | message passing, **ungated** |
| Class representation | logit column | logit column | logit column | **learned query embedding** |
| Receptive field | none | 4 hops (diameter is 10) | 4 hops | 4 hops **+ global attention** |
| Feeder geometry used | no | edges only | edges only | **edges + hop distances** |
| Handles dropout by | nothing | nothing | gate at every hop | **mask in the decoder only** |
| Handles noise by | nothing | nothing | nothing (mask stays 1) | **learned reliability** (untested) |
| Decoder | per-node + soft vote | per-node + soft vote | per-node + soft vote | **26-query cross-attention** |
| Parameters | — | 95,258 | 95,258 | **118,279 (+24.2%)** |
| Inference cost | — | baseline | +18% | to be measured |

### The bar it has to clear

Not the untreated baseline — **the mitigation**:

| | clean F1 | Ret@10 | AUDC |
|---|---|---|---|
| standard RGSAGE-Max | 0.846 | 5.7% | 0.402 |
| + neighbour-mean imputation | — | 11.8% | 0.476 |
| SDR-STGNN | — | 75.2% | 0.892 |
| **dropout-aware RGSAGE-Max** | **0.839** | **80.3%** | **0.913** |

FaultQuery-STGNN will be trained in both arms (standard and dropout-aware). The
headline comparison is FaultQuery-DA vs RGSAGE-Max-DA.

---

## 7. How it is tested

**Unit tests** (`tests/test_fault_query.py`, 15 passing, CPU, ~3 s). The two
designed to catch a silent no-op:
- dropped sensors receive **exactly** 0 attention — *and* the paired test that
  with the mask disabled they receive non-zero attention, proving the mask is
  not merely decorative;
- with a constant input (so the content term is identical for every sensor) and a
  large α, **every one of the 25 fault classes attends most to its nearest
  sensor** — proving the physics prior actually steers attention.

Plus: `D` alignment (exactly the 6 instrumented fault buses may have a zero
entry), the `[B,1,26]` contract against the real `soft_vote` and
`expanded_cross_entropy`, an all-sensors-dropped NaN guard, and gradient flow to
`queries`, `alpha_raw`, `beta`.

**Ablation grid.** A2 (no physics), A3 (no mask), A4 (no reliability), A6 (no
local GNN). **A2, A3 and A4 have parameter counts identical to the full model
(118,279)** — so if any of them changes accuracy, that change cannot be explained
by capacity. Only A6 differs (91,463), as expected.

**Gate.** A seed-42 pilot on the frozen split, against RGSAGE-Max's 0.846 ± 0.007:
≥ 0.83 proceed; 0.75–0.83 one diagnostic retry; < 0.75 stop and report.

**Protocol.** Identical frozen corpus, splits and mask seeds as every other model,
plus the scenario-held-out split (`dataset_fullscale_50_scenario`) — a new
architecture is exactly the kind of thing that could exploit the frozen split's
825/825 scenario overlap.

---

## 8. Honest limits

- **It may only match.** Dropout-aware training retains 80.3% at k=10 at **zero**
  inference cost. SDR-STGNN is our own precedent for an architectural idea
  matching but not beating it.
- **Capacity confound at the top level.** +24.2% parameters. If FaultQuery wins
  narrowly, a parameter-matched RGSAGE-Max control is required before the claim
  stands. (The *ablations* are clean; the headline comparison is not.)
- **The reliability head may be untrainable as configured.** Nothing in standard
  or dropout-aware training teaches `r_i` to detect noise, because the labels
  never depend on it. If A4 shows no effect, the fix is noise augmentation during
  training — which is a **training** change and must be reported as one, not
  folded into the architecture claim.
- **25 sensors is small.** Global attention fixes a receptive-field problem that
  may not be the binding constraint. The honest possibility is that the ceiling
  here is data, not architecture. Supporting evidence: a plain logistic regression
  on 375 summary features reaches **0.713** at k=0 on 8% of the training data,
  against RGSAGE-Max's 0.846 — the deep-model margin is 0.133, not vast.

**Why it is still worth running.** Three outcomes, two of which do not require
winning:
1. It beats dropout-aware training → a genuine new architecture.
2. It ties, but A2 shows the physics prior carries a measurable share → "a
   topological prior can substitute for part of the augmentation" is a real
   finding.
3. It ties everywhere → two *structurally different* architectural interventions
   both fail to beat a training-time fix. That is a substantially stronger
   negative result than the single data point the paper has now.

---

## 9. Provenance

| Claim | Source |
|---|---|
| 0.846 / 0.048 / AUDC 0.402 | `results/dropout_study/degradation_curves.csv` |
| 0.839 / 0.674 / AUDC 0.913 / Ret 80.3% | `results/dropout_study_aware/degradation_curves.csv` |
| SDR 0.892 / 75.2% / p=0.098 / +18% | `paper/figures/sdr_robustness_summary.csv`, `results/runtime/`, `results/statistics/` |
| targeted 0.448 vs 0.736, p=0.010 | `results/dropout_study_targeted/targeted_summary.json` |
| noise 0.786 / 0.579 / 0.345, DA null p≥0.107 | `results/dropout_study_noise/` |
| imputation AUDC 0.476 / 11.8% | `results/dropout_study_impute/imputation_summary.json` |
| logreg 0.713 | `results/classical_baseline/` |
| trivial floor 0.0256 | `scripts/build_absolute_audc.py` |
| 6 of 25 fault buses instrumented | `fault_localization/config/constants.py`; asserted in `tests/test_fault_query.py` |
| hop distances 0–23 | `fault_localization/data/ieee123/config/fault_hop_distance.npy` |
| all parameter counts | measured from the instantiated model by `scripts/build_faultquery_figure.py` |

**No FaultQuery-STGNN accuracy appears anywhere in this document, because none
exists yet.**

---

## 10. Files

| Path | Role |
|---|---|
| `fault_localization/models/fault_query.py` | the model |
| `fault_localization/graph/fault_distance.py` | builds + caches the `[26,25]` hop-distance matrix |
| `fault_localization/data/ieee123/config/fault_hop_distance.npy` | the cached matrix (static feeder property) |
| `config/fault_query.yaml` | standard arm |
| `config/dropout_aware/fault_query.yaml` | dropout-aware arm (flattened, no `base:`) |
| `scripts/run_fault_query.py` | training orchestrator, resumable |
| `scripts/build_faultquery_figure.py` | the architecture slide |
| `tests/test_fault_query.py` | 15 invariants |
| `fault_localization/experiments/runner.py` | `_model` registers `"fault_query"` |
