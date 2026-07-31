# Phase 7 — Conditional extended baselines: RGCN and RGATv2

Read [experiments-overview.md](../experiments-overview.md) first. Fingerprint, data, split, optimizer, loss, voting, evaluation, seeds, checkpointing, and artifacts are **see experiments-overview.md, reproduction contract**.

Run this phase only after GRU and both RGSAGE variants pass. If constrained, document omission in benchmark validation rather than using an improvised substitute.

## Terminology

In [BASE], **RGCN is recurrent GCN: GRU followed by graph convolution**. It is not R-GCN / relational graph convolution and needs no relation types. RGATv2 likewise means GRU followed by GATv2.

## RGCN

1. Add `fault_localization/models/gcn.py`: `[B,25,128]` GRU embeddings → GCN → `[B,25,26]` logits.
2. Use GNN hidden dimension **64** and **15 epochs**.
3. Implement normalized GCN adjacency; self-loop/normalization semantics are protocol decision 5 in the overview.
4. Common dropout/BN/ReLU settings are **see experiments-overview.md, reproduction contract**.

Tests: hand-computed normalized adjacency, selected self-loop policy, isolated graph batching, dimensions/gradients, tiny overfit, CPU smoke.

## RGATv2

1. Add `fault_localization/models/gatv2.py`: `[B,25,128]` GRU embeddings → GATv2 → `[B,25,26]` logits.
2. Use hidden dimension **64**, **four attention heads**, attention dropout **0.3**, and **15 epochs**, per [BASE] §4.3.
3. Other common settings are **see experiments-overview.md, reproduction contract**.

Tests: configured post-head dimension, attention only over valid graph edges/self-loop policy, attention dropout active only in train mode, isolated batching, gradients, tiny overfit, CPU smoke.

## Execution

Run each model for seeds `42`, `43`, `44` (then `45`, `46` if feasible) through the shared runner. [BASE] §5.1 references about 94.7 F1 (RGCN) and 94.9 (RGATv2); exact matching is not required. Proceed only to [benchmark approval](../benchmark-approval.md) after selected baselines have complete comparable artifacts.
