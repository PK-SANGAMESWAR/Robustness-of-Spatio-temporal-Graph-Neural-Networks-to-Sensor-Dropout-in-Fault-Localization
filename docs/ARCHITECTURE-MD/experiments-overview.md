# Experiments overview — shared benchmark infrastructure

## Purpose

Build a reproducible IEEE 123-bus **default feeder / measured-only graph** baseline benchmark from [BasePaper.pdf](BasePaper.pdf) ([BASE]). GRU and STGNN baselines must be trusted before any edge-feature architecture is developed.

This is the sole authority for shared experimental rules. Architecture files link here instead of independently redefining data, optimization, soft voting, evaluation, or artifacts.

## Scope

Included: dataset audit/freeze, lazy loading and graph batching, shared model/trainer/evaluator infrastructure, reproduced baselines, multi-seed analysis, and approval.

Deferred: edge-feature models and ablations, green feeder, full topology, hyperparameter search, and paper writing.

## Current repository status

The repository already contains simulation, processing, graph, and dataset-validation code, but no model, loader, trainer, evaluator, reporting package, or Torch/PyG dependencies. The committed corpus has seven pilot events (287 windows): it is a smoke-test fixture, not a benchmark.

| Pre-freeze issue | Evidence | Required action |
|---|---|---|
| Dataset root | Code uses `fault_localization/data/ieee123/dataset/`; older docs also name `dataset/` | Select one configuration-relative canonical root; retain current location unless migration is necessary. |
| Run count | `constants.py`: 74; `dataset_spec_v1.yaml`: 76 | Derive both from one resolved protocol; never hand-maintain duplicates. |
| Windowing | [BASE] §4.2: 40/event and 50/50 healthy/fault; current code: 41 (21/20) | Lock exact endpoint and label convention, then regenerate processing outputs if needed. |

## Reproduction contract

| Area | Shared contract |
|---|---|
| Benchmark | `benchmark_v1.0`, default feeder, measured-only 25-node graph; verify dataset fingerprint before every run. |
| Input | Float32 `[B,N=25,S=20,F=3]`; node order from `node_mapping.json`; Z-score statistics fit on training IDs only. |
| Labels/loss | 26 classes; graph/window target expanded per node only for node-level PyTorch cross-entropy. |
| Soft voting | At inference: `softmax(node_logits).sum(node)` then argmax, including no-fault. It is not part of the loss. |
| Canonical metric | Soft-voted graph/window F1 on fixed test IDs. Lock F1 averaging before experiments; node metrics are diagnostic only. |
| Optimizer | AdamW; learning rate `0.001`, weight decay `1e-4`, batch size `32`. See this contract rather than restating it in model files. |
| Common layers | ReLU, dropout `0.35`, batch normalization; exact ordering is an audited protocol decision. |
| Selection | Fixed-epoch training; choose best checkpoint by predeclared validation soft-voted F1; read test once afterwards. |
| Randomness | Seed Python, NumPy, Torch/CUDA, and DataLoader workers; record deterministic-mode state. 3 seeds required, 5 preferred. |
| Comparability | All models use identical fingerprint, splits, normalizer, batching, evaluator, and result schema. |

[BASE] §4.3 supplies the shared GRU hidden size 128, GNN hidden dimension 64, ReLU, dropout, batch normalization, AdamW, batch size, cross-entropy, and voting approach. Per-model epochs, aggregation, and attention settings belong only in the architecture plans.

## Five protocol decisions needing evidence

Log each in `docs/reproduction_log.md` before training:

1. **40-window and label rule:** exact dropped endpoint and 20 healthy / 20 fault convention.
2. **Layer order:** batch normalization, ReLU, and dropout placement.
3. **GNN depth:** number of GCN, GraphSAGE, and GATv2 spatial layers.
4. **F1 definition:** macro/weighted/other averaging, class order `0..25`, and zero-division policy.
5. **Recurrent GCN semantics:** normalized adjacency and self-loop policy. [BASE]’s RGCN is GRU + GCN, not relational GCN.

```text
Paper states:   <quote or paraphrase with section/page>
Implementation: <class/function/config value>
Reason:         <faithful choice or ambiguity>
Verification:   <test/code review/empirical check>
Status:         Verified | Assumed | Needs confirmation
```

## Target layout

```text
fault_localization/
├── data/{manifest.py,dataset.py}
├── models/{common.py,voting.py,gru.py,graphsage.py,gcn.py,gatv2.py}
├── experiments/{config.py,seed.py,train.py,evaluate.py,runner.py,reporting.py,cli.py}
└── analysis/errors.py
config/{experiment_base.yaml,gru.yaml,rgsage_mean.yaml,rgsage_max.yaml,rgcn.yaml,rgatv2.yaml}
```

Add compatible `torch` and `torch-geometric` dependencies and lock them. Record Python, Torch, PyG, OpenDSS, OS, CPU/GPU, CUDA, and device selection per run. The loader must not preload the full corpus.

## Result layout

```text
results/benchmark_v1.0/
├── manifest.json             # checksums/counts/graph/split fingerprint
├── protocol.md               # resolved window/label convention
├── <model>/seed_<seed>/
│   ├── config.yaml           # resolved immutable config
│   ├── environment.json
│   ├── dataset_manifest.json
│   ├── metrics.json
│   ├── metrics_by_epoch.csv
│   ├── checkpoint_{last,best}.pt
│   ├── confusion_matrix.{csv,png}
│   ├── learning_curve.png
│   ├── predictions_test.parquet
│   └── run.log
├── benchmark_comparison.csv
├── benchmark_validation.md
└── error_analysis.md
```

Run IDs derive from resolved config, seed, fingerprint, and code revision. Completed runs cannot be overwritten; resume requires all inputs to match.

## Phase 0 — Protocol lock and dependencies

1. Enter evidence from [BASE] §§3.1–3.3, 4.2–4.3, and 5.1 in the reproduction log.
2. Add/lock Torch and PyG; pass a CPU graph smoke test.
3. Create base configuration schema: dataset/graph, model fields, optimizer, epochs, metric, seed/device, output root, checkpoint rule.
4. Resolve the five decisions. Change processing only for a documented reproduction defect.
5. Generate dataset spec and constants from the same resolved protocol.

**Exit:** configuration complete; window rule tested; GRU/RGSAGE assumptions auditable.

## Phase 1 — Audit, complete, and freeze

1. Implement `data.manifest.create_manifest(data_root)` with checksums for raw events, windows, labels, metadata, splits, graph, mapping, normalizer, and spec.
2. Add strict validation: 825 scenarios/runs, per-event balance, all 26 labels, no split leakage, readable split IDs, graph/mapping consistency, train-only normalization.
3. Retain the seven-event corpus as fixture; generate the complete corpus only after protocol lock.
4. Store `results/benchmark_v1.0/manifest.json`; reject a run with a changed live fingerprint.

**Exit:** complete validated corpus and immutable benchmark inputs.

## Phase 2 — Lazy data and graph batching

1. Implement `WindowGraphDataset`, mapping IDs to event NPZ/intra-event index and using a bounded LRU event cache.
2. Return normalized `x [25,20,3]`, int64 target, window ID, metadata; never write derived training data into benchmark paths.
3. Load the static graph once; batch isolated graph copies per sample with PyG or tested equivalent.
4. Provide deterministic loaders; shuffle training only.

**Tests:** shape/dtype/node ordering, split coverage, train-only normalization, no cross-sample edges, cache bound.

**Exit:** pilot iterates deterministically; full corpus dry-run avoids full materialization.

## Phase 3 — Shared primitives

1. Implement and hand-test `soft_vote([B,N,C])` including no-fault.
2. Implement shared node-wise temporal encoding: `[B,N,S,F] → [B×N,S,F] → [B,N,128]`.
3. Implement 26-logit node classifier and target expansion for loss.
4. Implement evaluator: canonical graph metrics, node diagnostics, per-class values, fixed 26×26 confusion, per-window predictions.

**Tests:** shapes/gradients, shared GRU weights, manual voting/metric examples, tiny synthetic overfit.

**Exit:** node and graph-level metrics cannot be confused.

## Phase 4 — Runner and safeguards

1. Seed all sources and record deterministic state.
2. Implement train/validation loops with NaN checks, timings, mode correctness, epoch CSV.
3. Lifecycle: resolve config → verify manifest → fixed epochs → select validation-best checkpoint → test once → atomically save artifacts.
4. Resume only for matching config/seed/manifest/environment; fail safely for completed/mismatched runs.
5. Expose `validate-benchmark`, `train`, `evaluate`, and `report` CLI commands.

**Tests:** CPU artifact smoke test, resume/mismatch behavior, manifest rejection, proof test IDs are accessed only after checkpoint selection.

**Exit:** reproducible smoke run. Continue [GRU](models/gru.md) → [RGSAGE](models/rgsage.md) → [extended](models/extended-baselines.md) → [approval](benchmark-approval.md).
