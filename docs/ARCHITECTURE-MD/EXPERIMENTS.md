# Experiments documentation index

The baseline-reproduction plan is split so shared rules have one source of truth and model plans contain only architecture-specific work.

| Document | Responsibility |
|---|---|
| [experiments-overview.md](experiments-overview.md) | Shared purpose, scope, contract, protocol decisions, layouts, and Phases 0–4. |
| [models/gru.md](models/gru.md) | Phase 5 — GRU baseline. |
| [models/rgsage.md](models/rgsage.md) | Phase 6 — RGSAGE mean/max. |
| [models/extended-baselines.md](models/extended-baselines.md) | Phase 7 — conditional RGCN and RGATv2. |
| [benchmark-approval.md](benchmark-approval.md) | Phase 8 — aggregation, error analysis, and approval. |

Read the overview before any architecture plan. It is authoritative for settings held constant across models: benchmark fingerprint, dataset/splits, optimizer, loss, soft voting, metric, artifacts, and seeds.

The base reference is [BasePaper.pdf](BasePaper.pdf): Karabulut, Manna, and Develder (2026). **RGCN means recurrent GCN (GRU + GCN), not relational GCN.**
