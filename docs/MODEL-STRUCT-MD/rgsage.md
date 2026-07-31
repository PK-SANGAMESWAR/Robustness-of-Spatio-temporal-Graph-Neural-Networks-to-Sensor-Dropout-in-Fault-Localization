# Phase 6 — RGSAGE mean/max reproduction

Read [experiments-overview.md](../experiments-overview.md) first. Dataset fingerprint, splits, optimizer, loss, soft voting, F1, checkpointing, seeds, and artifacts are **see experiments-overview.md, reproduction contract**.

## Purpose

Reproduce [BASE] §§3.3.2 and 4.3’s recurrent GraphSAGE model on the 25-node measured-only graph. Mean and max differ only by aggregation.

## Implementation

1. Add `fault_localization/models/graphsage.py`: shared GRU embeddings `[B,25,128]` → GraphSAGE → logits `[B,25,26]`.
2. Use GNN hidden dimension **64** and **15 epochs**.
3. Use full neighborhoods, never sampled neighborhoods; [BASE] explicitly deems this feasible for the small measured graph.
4. Implement Eq. 5 with both self paths: concatenate each node’s own representation with the aggregate **and** include that node in its aggregation neighborhood.
5. Support only `mean` and `max`; common layer order is the shared protocol decision.

## Configurations

```yaml
# rgsage_mean.yaml; max differs only in name and aggregator
model:
  name: rgsage_mean
  temporal_hidden_dim: 128
  gnn_hidden_dim: 64
  aggregator: mean
  neighbor_sampling: false
training:
  epochs: 15
```

## Tests

- Correct input/output dimensions.
- Hand graph produces distinct expected mean/max output.
- Tests separately fail when aggregation self-loop or concatenated self path is removed.
- Batched graph isolates samples and preserves node mapping.
- Tiny overfit and CPU smoke run pass.

## Execution and investigation gate

Run mean and max for seeds `42`, `43`, `44` (then `45`, `46` if possible) using shared reporting only. [BASE] §5.1 reports about 94.7 F1 for both; treat this and its large GRU-to-GNN gain as qualitative references.

If performance is materially below GRU or behavior is anomalous, stop Phase 7 and audit: (1) shared voting/metric, (2) graph/node batch order and both Eq. 5 self paths, (3) labels/splits/normalization, (4) resolved layer settings, then (5) defects. Log findings; do not tune unreported hyperparameters. Continue to [extended baselines](extended-baselines.md) only after validation.
