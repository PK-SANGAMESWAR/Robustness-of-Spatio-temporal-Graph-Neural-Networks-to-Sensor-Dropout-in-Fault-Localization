# Experiments Implementation Plan — Baseline Reproduction and Benchmark Validation

## Goal and scope

Implement a reproducible experimental system that validates the frozen IEEE 123-bus **default feeder / measured-only graph** benchmark against Karabulut et al. (2026), hereafter **[BASE]**, before any edge-feature model is introduced.

The implementation order is intentionally:

```text
dataset audit and freeze → data access → shared evaluation contract → GRU
→ RGSAGE (mean/max) → optional RGCN/RGATv2 → multi-seed analysis
→ benchmark approval → proposed model work
```

This plan does not implement the edge-feature-augmented model, the green feeder, full-topology experiments, hyperparameter search, or paper writing.

## Current repository assessment

The data-generation and validation pipeline exists under `fault_localization/{simulation,processing,graph,validation}`. The repository currently has **seven pilot events** (287 windows) at `fault_localization/data/ieee123/dataset/`, plus graph artifacts and split files. It has no model, dataset-loader, trainer, evaluator, reporting package, or PyTorch/PyG dependency.

The pilot is useful for smoke tests only; it is not a valid benchmark. It lacks the 825-scenario coverage, class coverage, and scale required for a baseline result.

Two inconsistencies must be reconciled before freezing:

| Item | Current state | Required resolution |
|---|---|---|
| Dataset location | Code/pilot use `fault_localization/data/ieee123/dataset/`; `DATASET.md` describes `dataset/` | Choose one canonical root and make every path configuration-relative. The recommended canonical root is `fault_localization/data/ieee123/dataset/`, retaining the repository’s existing layout. |
| Run count | `constants.py` uses 74; `dataset_spec_v1.yaml` says 76 | Regenerate the manifest from a single source of truth after the 40-versus-41 window protocol is locked. Never hand-edit one without the other. |
| Window count / labels | [BASE] states 40 windows and a 50/50 healthy/fault distribution; current extractor yields 41, with 21 healthy and 20 fault-labelled windows | Resolve from the paper’s exact time-index convention, document it, and regenerate all processed artifacts if the current pilot convention is wrong. |

## Reproduction contract

The following settings are fixed for all reproduced models unless [BASE] is found to specify otherwise. Each is stored in the per-run resolved configuration and reproduction log.

| Area | Contract |
|---|---|
| Input sample | Float32 tensor `[B, N=25, S=20, F=3]`; nodes follow `node_mapping.json`; features use training-split Z-score statistics only. |
| Labels | 26 classes: no-fault = 0, then the sorted 25 fault buses. The graph/window label is expanded to each node only for the node-level training loss. |
| Evaluation unit | **Graph/window-level soft-voted prediction** is the canonical reported metric. Raw node-level metrics are diagnostic only and must never replace the canonical score. |
| Soft voting | `softmax(node_logits, dim=-1).sum(dim=node)` followed by `argmax`, including class 0. It is used only for evaluation/inference; loss remains node-level cross-entropy unless [BASE] evidence says otherwise. |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, cross-entropy loss. |
| Shared neural settings | GRU hidden size 128, GNN hidden size 64, ReLU, dropout 0.35, batch normalization. GRU = 11 epochs; GNN models = 15 epochs. |
| RGSAGE | Full-neighborhood aggregation; explicit self-loop included in the aggregation neighborhood as [BASE] Eq. 5 specifies; run both mean and max variants. |
| RGATv2 | Four attention heads and attention dropout 0.3 in addition to shared settings. |
| Randomness | Seeds are named in configuration, applied to Python/NumPy/PyTorch/CUDA, and deterministic-mode state is recorded. Three seeds are required; five are preferred. |
| Comparability | All models share the identical dataset fingerprint, graph, split IDs, normalization statistics, batch construction, metrics, hardware capture, and evaluation code. |

### Protocol decisions that require evidence before implementation

Create a decision row in `docs/reproduction_log.md` for each of the following before training:

1. Exact 40-window extraction and label rule. The current 41-window overlap rule cannot yield the published 50/50 healthy/fault balance, so the exact dropped endpoint and class assignment must be established first.
2. Exact placement/order of batch normalization, dropout, and ReLU in GRU and GNN branches when not explicit in [BASE].
3. Number of layers for each GNN and RGCN relation definition. [BASE] describes the model family and dimensions, but ambiguities must be recorded as `Assumed` until verified.
4. Whether the reported F1 is macro, weighted, or another scikit-learn averaging mode. The metric must not be selected after seeing results.
5. The source/reference used to create RGCN edge types. If the measured-only graph has no paper-defined relation labels, RGCN is an optional extension rather than a silently improvised reproduction.

Use this entry format:

```text
Paper states:   <quote or close paraphrase, section/page>
Implementation: <exact class/function/config value>
Reason:         <why this is faithful or the remaining ambiguity>
Verification:   <test, code review, or empirical check>
Status:         Verified | Assumed | Needs confirmation
```

## Target implementation layout

Add the following; existing simulation, processing, graph, and validation modules remain the data producer.

```text
fault_localization/
├── data/
│   ├── manifest.py              # benchmark fingerprint and artifact inventory
│   └── dataset.py               # lazy split-aware window dataset
├── models/
│   ├── common.py                # shared classifier, activation/BN/dropout blocks
│   ├── gru.py                   # node-wise GRU baseline
│   ├── graphsage.py             # RGSAGE mean/max
│   ├── rgcn.py                  # optional paper-faithful RGCN
│   ├── gatv2.py                 # optional RGATv2
│   └── voting.py                # tested soft-voting operation
├── experiments/
│   ├── config.py                # dataclass/schema and resolved-config validation
│   ├── seed.py                  # deterministic seed setup and state capture
│   ├── train.py                 # one-epoch train/evaluate loops
│   ├── evaluate.py              # canonical metrics and confusion outputs
│   ├── runner.py                # one run lifecycle and resume safeguards
│   ├── cli.py                   # `python -m ...` commands
│   └── reporting.py             # aggregate seeds and benchmark table
└── analysis/
    └── errors.py                # error slices and electrical-distance analysis

config/
├── experiment_base.yaml         # fields shared by all experiments
├── gru.yaml
├── rgsage_mean.yaml
├── rgsage_max.yaml
├── rgcn.yaml
└── rgatv2.yaml

tests/
├── test_benchmark_manifest.py
├── test_experiment_dataset.py
├── test_models.py
├── test_soft_voting.py
├── test_training_smoke.py
├── test_evaluation.py
└── test_reporting.py
```

`pyproject.toml` must add compatible CPU/GPU installation instructions for `torch`, `torch-geometric`, and their required PyG extensions. Pin the resolved versions in `uv.lock`; record `torch`, PyG, OpenDSS, Python, OS, CPU/GPU, and CUDA versions in every run’s `environment.json`. Do not make the full dataset a package dependency or load all windows into memory.

## Result layout and immutable run identity

Every run receives a deterministic ID derived from the resolved model config, seed, dataset fingerprint, and code revision. A completed run directory is immutable; rerunning it without `--resume` must fail rather than overwrite evidence.

```text
results/
├── benchmark_v1.0/
│   ├── manifest.json             # dataset/graph/split/norm checksums and counts
│   ├── protocol.md               # resolved window and label convention
│   ├── gru/seed_00042/
│   ├── rgsage_mean/seed_00042/
│   ├── rgsage_max/seed_00042/
│   └── ...
├── benchmark_comparison.csv
├── benchmark_validation.md
└── error_analysis.md
```

Each `model/seed` directory contains:

```text
config.yaml               # resolved config, never a mutable template
environment.json          # package/hardware/device information
dataset_manifest.json     # copied benchmark fingerprint
metrics.json              # best-validation and final-test metrics
metrics_by_epoch.csv      # loss, node-level and soft-voted validation scores
checkpoint_last.pt
checkpoint_best.pt        # selected only by validation metric
confusion_matrix.csv
confusion_matrix.png
learning_curve.png
predictions_test.parquet  # window ID, truth, vote prediction, node predictions
run.log
```

Test labels are read only once per run, after `checkpoint_best.pt` is selected. No test score may control epochs, architecture, or hyperparameters.

## Phase 0 — Resolve protocol and prepare dependencies

### Work

1. Read [BASE] §§3.1–3.3, 4.2–4.3, and 5.1 into `docs/reproduction_log.md`, with page/section references.
2. Add the training dependencies and a documented install path appropriate for the available hardware. Verify imports with a small PyG graph before writing models.
3. Create `config/experiment_base.yaml` with explicit fields for data root, graph files, node ordering, model parameters, optimizer, epochs, checkpoint selection metric, metrics averaging, seed, device, and output root.
4. Resolve the five protocol decisions above. Update the dataset implementation/spec only if this demonstrates a reproduction defect; otherwise preserve the dataset unchanged.
5. Add a `benchmark_version` and `dataset_root` to the machine-readable dataset spec. Generate it from code/config rather than maintaining divergent constants.

### Exit criteria

- All settings required to train and score a model are explicit in configuration.
- The 40/41-window decision is documented and tested.
- PyTorch/PyG imports and a CPU smoke graph work in the locked environment.
- No reproduction-log item needed by GRU/RGSAGE is `Needs confirmation` without an explicitly approved assumption.

## Phase 1 — Dataset audit, completion, and freeze

### Work

1. Implement `fault_localization.data.manifest.create_manifest(data_root)` to inventory raw events, processed event files, labels, metadata, split files, graph arrays, node mapping, normalization statistics, and dataset spec. Store SHA-256 checksums for non-window artifacts and a deterministic per-file checksum inventory for windows/raw files.
2. Extend validation with a **strict benchmark mode** that verifies:
   - all 825 `(fault_bus, fault_type, resistance)` scenarios exist;
   - each has the agreed `runs_per_scenario` count;
   - each event has the agreed number of windows and exactly the required class balance;
   - 26 labels are present with expected counts/tolerances;
   - every split has no event leakage and expected 70/15/15 counts;
   - every split ID maps to exactly one readable window;
   - `node_mapping.json`, `edge_index.npy`, and `edge_attr.npy` agree on 25 nodes and directed-edge ordering;
   - training normalization statistics were fit only from training IDs.
3. Treat the checked-in seven-event files as a fixture/pilot. Generate the full dataset only after the window convention is decided; then run the strict checks.
4. Produce `results/benchmark_v1.0/manifest.json`, tag the benchmark version in Git, and reject training if the live fingerprint differs from the recorded manifest.

### Important implementation rule

Freeze means immutable **inputs**, not merely a folder name. Training code compares the current manifest with the expected fingerprint before every run. A mismatch requires creating `benchmark_v1.1` with a written reason; it must not overwrite `benchmark_v1.0`.

### Exit criteria

- A complete validated corpus exists, or the project is explicitly stopped before experimental claims.
- The manifest and strict validation report pass.
- `benchmark_v1.0` is tagged and its data artifacts are protected from experiment writes.

## Phase 2 — Data access and batching

### Work

Implement `fault_localization.data.dataset.WindowGraphDataset`:

1. Read `labels.csv`, `metadata.csv`, a requested split text file, `node_mapping.json`, and `norm_stats.json` once at initialization.
2. Build an index from `window_id` to `(event_npz_path, intra_event_index, label, metadata)`. Lazily read event NPZs; use a bounded LRU cache so a 2.5M-window run does not exhaust RAM.
3. Return `x: float32[N,S,F]`, `y: int64`, `window_id`, and metadata. Normalize on read using the frozen train-only statistics. Do not write normalized arrays into the benchmark directory.
4. Load `edge_index` once, validate it, and make it available to graph model batches. Replicate/batch the same static graph for every item in a mini-batch using PyG’s documented batching utilities or a tested equivalent.
5. Provide a deterministic DataLoader factory. Training shuffle seed must derive from run seed and epoch; validation/test loaders must be unshuffled.

### Tests

- A returned sample has `[25,20,3]`, float32 features, valid class ID, and correct node order.
- Train normalization is applied identically to train/val/test without recalculation.
- Each split returns exactly its listed IDs; no ID is duplicated or missing.
- Batched graph edges do not connect samples to each other.
- LRU cache bounds are respected on a synthetic multi-event corpus.

### Exit criteria

The loader can iterate the pilot deterministically, and a full-corpus dry-run reports counts without materializing all windows.

## Phase 3 — Shared model and evaluation primitives

### Work

1. Implement `soft_vote(node_logits)` in `models/voting.py`. Its input is `[B,N,C]`; output is both class-probability sums `[B,C]` and predictions `[B]`. It must include no-fault and must not average probabilities, because averaging is argmax-equivalent but obscures the paper’s stated sum.
2. Implement the shared temporal encoder in `models/common.py`: reshape `[B,N,S,F]` to `[B*N,S,F]`, use a shared GRU, take its final hidden representation, and restore `[B,N,128]`.
3. Implement a node classifier that emits `[B,N,26]`. The training target `[B]` is expanded to `[B,N]`; calculate cross-entropy over all `B*N` logits/targets.
4. Implement `evaluate.py` to produce:
   - canonical graph-level accuracy, precision, recall, and predeclared F1;
   - raw node-level versions as diagnostic fields;
   - per-class precision/recall/F1/support;
   - fixed class-order confusion matrix (0–25);
   - per-window prediction records for later error analysis.
5. Use `sklearn.metrics` with explicit `labels=range(26)` and `zero_division=0`; persist metric definition/version in every result.

### Tests

- Soft-voting matches a hand-computed two-node example and can select class 0.
- The GRU accepts `[B,25,20,3]`, has shared weights across nodes, and emits `[B,25,26]`.
- Cross-entropy target expansion has correct shape and gradients reach the encoder/classifier.
- Evaluation hand examples distinguish node-level from soft-voted graph-level F1.

### Exit criteria

All unit tests pass and one overfit test can drive loss near zero on a tiny synthetic dataset.

## Phase 4 — Training runner and scientific safeguards

### Work

1. Implement `experiments.seed.seed_everything(seed, deterministic)` for Python, NumPy, Torch CPU/CUDA and DataLoader workers. Record whether deterministic algorithms were enabled and any unsupported-operation warning.
2. Implement `train.py` with a training epoch, validation epoch, gradient/NaN checks, timing, and epoch-level CSV logging. Set train/eval mode correctly so batch normalization and dropout behave correctly.
3. Implement `runner.py` lifecycle:
   - resolve and validate config;
   - verify benchmark manifest before loading data;
   - initialize model/optimizer;
   - train the specified fixed epoch count;
   - select and save `checkpoint_best.pt` by a predeclared **validation soft-voted F1** (ties use earliest epoch);
   - load the selected checkpoint once and evaluate test;
   - atomically write artifacts and final status.
4. Implement resume only from `checkpoint_last.pt` when config, manifest, package versions, and seed match. Otherwise fail with a clear mismatch report.
5. Implement CLI commands:

```bash
python -m fault_localization.experiments.cli validate-benchmark --config config/experiment_base.yaml
python -m fault_localization.experiments.cli train --config config/gru.yaml --seed 42
python -m fault_localization.experiments.cli evaluate --run results/benchmark_v1.0/gru/seed_00042
python -m fault_localization.experiments.cli report --benchmark benchmark_v1.0
```

### Tests

- CPU smoke training writes all required artifacts on the pilot fixture.
- Rerunning a completed ID fails safely; `--resume` continues an interrupted compatible run.
- Changing any manifest checksum prevents training.
- The test loader is not invoked until checkpoint selection is complete (mock/spy test).

### Exit criteria

One complete CPU smoke run is reproducible byte-for-byte where deterministic kernels allow, or its documented nondeterminism is bounded.

## Phase 5 — GRU baseline reproduction

### Implementation

Implement `models/gru.py` using the shared GRU encoder and node classifier only. Configure hidden size 128, dropout 0.35, batch normalization, ReLU, 26 logits, AdamW settings from the contract, batch size 32, and 11 epochs.

### Execution

1. Run a single-seed smoke result on the complete frozen benchmark; inspect curve, class supports, and confusion matrix.
2. Run seeds `42`, `43`, `44` (then `45`, `46` if budget permits).
3. Record both raw node-level F1 and graph-level soft-voted F1; confirm that the latter is the canonical value. [BASE] indicates roughly a +0.10 soft-voting lift for GRU, but it is a diagnostic comparison—not an acceptance threshold.
4. Save all artifacts under `results/benchmark_v1.0/gru/`.

### Gate

All seeds finish without NaN/divergence; every run has a fixed-test evaluation and complete artifacts. A low score triggers audit, not tuning: validate labels/window convention, normalization, soft voting, and graph-independent batching first.

## Phase 6 — RGSAGE mean and max reproduction

### Implementation

Implement `models/graphsage.py` as GRU embeddings `[B,N,128]` followed by the paper-faithful GraphSAGE spatial layer(s), then the shared node classifier:

1. Ensure the aggregation neighborhood contains each node itself **and** preserves the explicit self/aggregated concatenation in Eq. 5. Add a unit test that would fail if either mechanism is omitted.
2. Support only `aggregator: mean|max` through config; no neighborhood sampling is used on this 25-node graph.
3. Apply the recorded activation/BN/dropout ordering and output hidden dimension 64.
4. Use 15 epochs and the exact same loader, splits, normalization, optimizer, batch size, voting, and metrics as GRU.

### Execution and gate

Run mean and max for the required three seeds. Compare their confidence intervals and deltas against the GRU in the single aggregate table. [BASE] reports approximately 94.7 F1 for both and a large GRU-to-GNN gap; results need comparable behavior, not an exact match.

If GraphSAGE is materially below GRU or the expected qualitative range, stop later models and investigate in this order:

1. soft-voting implementation and metric unit;
2. node order / graph construction / self-loop semantics;
3. label and 40-window convention;
4. normalization and split leakage;
5. tensor shape/batching and model hyperparameters;
6. general implementation defects.

Record each finding in `docs/reproduction_log.md`; do not tune unspecified hyperparameters to conceal a mismatch.

## Phase 7 — Extended baselines (conditional)

These models are valuable but optional after GRU + RGSAGE are validated. Skip them only with a written scope decision in `benchmark_validation.md`.

### RGCN

Implement only after the relation schema is reproducibly defined from [BASE]. Add tests for relation IDs, reverse-edge semantics, and no missing relation. If no defensible relation definition exists for the measured-only graph, mark RGCN `Not reproduced—insufficient specification`, cite [BASE]’s number as external reference, and do not manufacture a comparable result.

### RGATv2

Implement GRU → GATv2 → classifier using hidden dimension 64, four heads, attention dropout 0.3, standard dropout 0.35, batch normalization, ReLU, and 15 epochs. Add tests that attention coefficients are edge-local, dimensions are correct after head combination, and graph batching isolates samples.

Run the same three seeds and artifact protocol. Never compare an extended-baseline result produced on a different benchmark fingerprint.

## Phase 8 — Aggregation, error analysis, and approval

### Reporting

`experiments.reporting` aggregates run metrics by model and computes mean, sample standard deviation, and two-sided 90% Student-t confidence intervals (`n`, critical value, and formula saved to output). Do not label a one-seed result with an interval.

Write `results/benchmark_comparison.csv` with one row per `(model, seed)` and a separate summary row/table:

| Model | Seeds | Graph F1 mean ± 90% CI | Node F1 mean ± 90% CI | Precision | Recall | Training time | Inference time | Dataset fingerprint |
|---|---:|---:|---:|---:|---:|---:|---:|---|

`results/benchmark_validation.md` must state the metric definition, dataset fingerprint, unresolved assumptions, expected [BASE] references, observed values, deviations, and disposition (`approved`, `investigate`, or `not comparable`).

### Error analysis

For the best validated RGSAGE variant, implement `analysis/errors.py` to emit:

1. full 26×26 confusion matrix and most-confused off-diagonal pairs;
2. per-fault-bus, fault-type, resistance, load-multiplier-bin, and window-position metrics;
3. error rate by graph shortest-path distance and by electrical impedance distance from the true fault bus to the predicted bus, with an explicit mapping from fault buses to graph nodes where possible;
4. a reproducible table of representative failures, keyed by window ID, rather than only screenshots;
5. a concise interpretation that distinguishes observation from causal claim.

Do not use edge attributes in the baseline model or error analysis to claim improvement. Error patterns may motivate the later edge-feature hypothesis but do not validate it.

### Benchmark approval criteria

Approve only when all conditions hold:

- [ ] full benchmark manifest and strict dataset validation pass;
- [ ] protocol decisions are logged and no material issue is unresolved;
- [ ] GRU and both RGSAGE variants converge across at least three seeds;
- [ ] evaluation uses verified soft-voted graph-level F1 and no test-set tuning occurred;
- [ ] GNN-vs-GRU and within-GNN behavior are qualitatively comparable to [BASE] (published reference: GRU 86.3 ± 4.2, RGCN 94.7 ± 0.3, RGSAGE mean/max 94.7 ± 0.1, RGATv2 94.9 ± 0.6 for the specified setting);
- [ ] complete results, environment, seeds, manifest, and analysis are saved;
- [ ] no unresolved implementation issue explains a major deviation.

If approval fails, retain all artifacts, mark the validation `investigate`, and return to the earliest implicated phase. Do not begin the proposed edge-feature architecture.

## Test and execution sequence

Run in this order; full-scale training begins only after prior gates pass:

```bash
uv sync
python -m pytest tests -q
python -m fault_localization.experiments.cli validate-benchmark --config config/experiment_base.yaml
python -m pytest tests/test_benchmark_manifest.py tests/test_experiment_dataset.py tests/test_models.py tests/test_soft_voting.py tests/test_evaluation.py -q
python -m fault_localization.experiments.cli train --config config/gru.yaml --seed 42
python -m fault_localization.experiments.cli train --config config/rgsage_mean.yaml --seed 42
python -m fault_localization.experiments.cli train --config config/rgsage_max.yaml --seed 42
# repeat required seeds, then optional extended baselines
python -m fault_localization.experiments.cli report --benchmark benchmark_v1.0
```

## Deliverables

| Deliverable | Location |
|---|---|
| Frozen, checksummed benchmark contract | `results/benchmark_v1.0/manifest.json`, `results/benchmark_v1.0/protocol.md` |
| Auditable paper-to-code decisions | `docs/reproduction_log.md` |
| Lazy data and static-graph batching layer | `fault_localization/data/` |
| Reproduced baseline models and runner | `fault_localization/models/`, `fault_localization/experiments/` |
| Model configs | `config/*.yaml` |
| Unit, integration, and training smoke coverage | `tests/test_{benchmark_manifest,experiment_dataset,models,soft_voting,training_smoke,evaluation,reporting}.py` |
| Per-seed artifacts and checkpoints | `results/benchmark_v1.0/<model>/seed_<seed>/` |
| Aggregate comparison and validation decision | `results/benchmark_comparison.csv`, `results/benchmark_validation.md` |
| Fine-grained failure analysis | `results/error_analysis.md` and associated CSV/plot files |

