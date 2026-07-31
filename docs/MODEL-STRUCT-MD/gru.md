# Phase 5 — GRU baseline reproduction

Read [experiments-overview.md](../experiments-overview.md) first. Its reproduction contract governs benchmark fingerprint, splits, optimizer, loss, voting, F1, artifacts, and seeds; this plan contains GRU-only work.

## Purpose

Reproduce [BASE] §4.3’s aggregated temporal-only GRU baseline.

## Implementation

1. Add `fault_localization/models/gru.py` using shared temporal/classifier interfaces.
2. Apply one parameter-shared GRU independently to each μPMU sequence.
3. Transform `[B,25,20,3]` to node embeddings `[B,25,128]`, then logits `[B,25,26]`.
4. Use hidden size **128** and **11 epochs**, as [BASE] §4.3 specifies.
5. Use the shared layer-order decision; common dropout/BN/ReLU values are **see experiments-overview.md, reproduction contract**, not independent settings here.

## Configuration

```yaml
model:
  name: gru
  temporal_hidden_dim: 128
  gnn: null
training:
  epochs: 11
```

Inherit shared fields from `config/experiment_base.yaml`; persist the fully resolved config.

## Tests

- `[B,25,20,3] → [B,25,26]` shape.
- No per-node GRU modules; weights are shared.
- Expanded node target is `[B,25]`; both encoder and classifier receive gradients.
- Use shared `soft_vote`, not a duplicate implementation.
- Tiny deterministic overfit and CPU artifact smoke test.

## Execution and gate

Run a complete-corpus smoke seed, then seeds `42`, `43`, `44` (and `45`, `46` when feasible). Save under `results/benchmark_v1.0/gru/seed_<seed>/` through the shared runner.

Report node F1 only as diagnostic next to canonical soft-voted graph F1. [BASE] §5.1 reports roughly +0.10 F1 from GRU voting; that is a diagnostic expectation, not pass/fail.

All required seeds must converge and have complete artifacts. A poor result first audits shared labels, normalization, voting, metric, and loader before changing GRU code or beginning RGSAGE.
