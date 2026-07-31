# Dataset Specification — IEEE 123-Bus Fault Localization: Full Benchmark Reproduction

## Source Protocol

This dataset reproduces the simulation methodology described in:

Karabulut, B., Manna, C., Develder, C. (2026). *Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids.* arXiv:2604.20403.

Referred to below as **[BASE]**.

### Benchmark Reproduction Philosophy

This work follows a benchmark reproduction methodology. Rather than introducing a new dataset, the objective is to faithfully reproduce the benchmark proposed in [BASE] using the same simulation protocol.

The following components are reproduced:

- IEEE 123-bus feeder
- OpenDSS/PyDSS simulation environment
- 25 predefined fault locations
- 25 μPMU locations
- 11 fault types
- Fault resistances (0.1 Ω, 1 Ω, 10 Ω)
- Load variation strategy
- Sampling rate (1 ms)
- Window extraction procedure
- Measured-only graph construction
- Fault-location labels
- Train/validation/test splitting protocol

This reproduced benchmark is then used to evaluate the reproduced baseline models and the proposed architecture under identical experimental conditions.

## 1. Simulation Environment

- Simulator: OpenDSS via the PyDSS interface. *(reproduced)*
- Feeder: IEEE 123-bus feeder, 60 Hz, 4.16 kV nominal. *(reproduced)*
- Feeder configuration: The benchmark will reproduce the default feeder configuration first. The reconfigured ("green") feeder configuration described in [BASE] will be implemented subsequently to complete the full benchmark reproduction.

## 2. Fault-Type Taxonomy — Reproduced in Full

**All 11 fault types are reproduced**: AG, BG, CG (line-to-ground), AB, BC, CA (line-to-line), ABG, BCG, CAG (double line-to-ground), ABC, ABCG (three-phase). This preserves the LG/LL/LLG/three-phase category structure exactly as in [BASE]. Fault type is varied during simulation as an experimental factor (it changes the voltage signature shown to the model) but is **not** part of the prediction label — see §5.

## 3. Reproduced Benchmark Parameters

| Element | [BASE] | This dataset (reproduced) |
|---|---|---|
| Fault types | 11 | **11** |
| Fault resistance (Ω) | 0.1, 1.0, 10 | 0.1, 1.0, 10 |
| Fault positions (buses) | 25 fixed buses | **25 fixed buses** |
| μPMU locations | 25 fixed | 25 fixed |
| Simulation repetitions per scenario | 100 | **100 runs** per scenario (load multiplier redrawn each run) |
| Load multiplier | L ~ U(0.5, 1.3) | L ~ U(0.5, 1.3) |
| Fault duration | 20 ms | 20 ms |

Scenario count: 25 positions × 11 fault types × 3 resistances = **825 scenarios**, matching [BASE].

The exact number of generated windows will depend on the implemented simulation pipeline and will be verified after reproducing the benchmark. The train/validation/test split follows the same 70/15/15 protocol described in [BASE].

## 4. Sampling and Windowing — Reproduced Exactly

- Sampling resolution: 1 ms.
- Recorded window per fault event: 60 ms total, starting 40 ms before fault onset.
- Sliding window length: S = 20 timesteps (20 ms), stride 1 ms → 40 overlapping windows per fault event.
- Half of all windows = fault-free (pre-fault) condition; half = fault-active condition. Healthy (pre-fault) windows are retained to preserve the original benchmark structure and to enable future multi-task fault detection and localization without regenerating the dataset.
- Features per node per timestep: 3-phase RMS voltage magnitudes (V1, V2, V3).
- Normalization: Z-score per feature (mean 0, std 1), fit on the training set only.

The exact number of generated windows will be verified after implementing the simulation pipeline. The dataset is expected to approximately match the scale reported in [BASE].

## 5. Labels — Location Only, Not Location × Type

Prediction target is **fault location only**, exactly matching the protocol of [BASE]. Fault type and fault resistance are treated as simulation variables rather than prediction targets.

- 25 fault-location classes + 1 no-fault class = **26 classes total** (matching [BASE]'s 26-dimensional output logits): `No Fault, Bus_7, Bus_13, ..., Bus_108` (mapped to the 25 predefined physical bus IDs from §6).
- No location × type label space (would be 25 × 11 + 1 = 276 classes) — not used.
- Label applies to the full node feature vector; classification is performed at node level (per [BASE] §4.3), then aggregated via soft voting across nodes at inference.

## 6. Fault Position Selection — Reproduced from [BASE]

All 25 predefined fault locations from [BASE] are used:

**Bus IDs**: 7, 13, 18, 21, 25, 29, 35, 42, 47, 51, 53, 55, 57, 62, 65, 72, 80, 83, 86, 89, 93, 97, 99, 101, 108.

**μPMU Locations** (25 measurement nodes, per [BASE] Table 1):

**Bus IDs**: 1, 6, 13, 18, 25, 26, 33, 35, 36, 45, 47, 54, 60, 69, 75, 76, 81, 84, 91, 97, 103, 105, 109, 113, 300.

These locations are reproduced exactly from [BASE] and define the graph structure (see §8).

## 7. Train / Val / Test Split — Reproduced

- 70% / 15% / 15% random split.
- Sliding windows from the same time step are kept together across the split (no leakage across splits from the same fault event's overlapping windows).

## 8. Graph Construction

Two graph topologies, per [BASE] Algorithm 1 and §3.1. **Measured-only first** — build and validate completely before attempting full-topology.

1. **Measured-only graph** (primary): nodes = the 25 μPMU locations. Edges built via [BASE]'s Algorithm 1 — three-phase μPMUs connected along the backbone by electrical proximity (no closed loops); single/two-phase μPMUs connected to nearest three-phase μPMU, or to both flanking three-phase μPMUs if positioned between two in a loop.
2. **Full-topology graph**: all 123 buses + 5 auxiliary buses (128 nodes), unmeasured buses zero-valued and masked during loss computation.

## 9. Edge Attributes (Stored for Future Architectural Extensions)

Store the following per edge, alongside the measured-only graph:

- Resistance (R)
- Reactance (X)
- Impedance magnitude |Z|
- (Optional) Physical line length

These are static per graph topology, not per-timestep — store once per graph, not once per window. The subset of edge attributes used as model input (only R,X; only |Z|; or all) is an architecture-level decision to be made during model development. Naming these fields as "edge-feature-augmented GraphSAGE" input, not "physics-informed," until an actual electrical-equation-derived loss or constraint is added — see project.md for that distinction.

## 10. File / Folder Structure

```
dataset/
  raw_sim/                    # raw OpenDSS/PyDSS simulation outputs, one file per (fault_scenario, run)
  processed/
    windows/                  # extracted sliding-window tensors, node features [N, S, 3]
    labels.csv                 # window_id, fault_bus, label
    metadata.csv                # window_id, run_id, fault_bus, fault_type, resistance,
                                 # graph_type, seed
    edge_index.npy               # PyG-format edge connectivity for the measured-only graph
    edge_attr.npy                 # PyG-format edge attributes (R, X, |Z|, optional line length)
    graph_full_topology.json     # phase 2
  splits/
    train.txt val.txt test.txt  # window_id lists, generated once, fixed for all model runs
  config/
    dataset_spec_v1.yaml        # this spec, machine-readable, referenced by every training run
```

`edge_index.npy` / `edge_attr.npy` (rather than JSON) so the measured-only graph loads directly into PyTorch Geometric without a conversion step. `metadata.csv` as a flat table makes later slicing (by fault type, resistance, bus region) a simple filter rather than a JSON parse.

## 11. Open Items to Confirm Before Generation Starts

- [ ] Confirm OpenDSS/PyDSS environment runs the standard IEEE 123-bus model correctly (baseline power flow, no faults) before writing any fault-injection code.
- [ ] Confirm all 25 fault positions and 25 μPMU locations against the actual feeder one-line diagram ([BASE] Fig. 2).
- [ ] Confirm OpenDSS line objects expose per-line impedance/reactance directly.
- [ ] Confirm scenario/run count (825 scenarios × 100 runs) fits the available compute/time budget; adjust run count (not fault-type count) if it doesn't.
- [ ] Verify the exact sample counting methodology against [BASE] before hardcoding dataset size.

## 12. Baseline Validation

Before developing the proposed model, the reproduced benchmark will be validated.

### Validation Pipeline

```
Dataset (reproduced)
       ↓
   GRU baseline
       ↓
 GraphSAGE baseline
       ↓
   RGCN baseline
       ↓
  GATv2 baseline
       ↓
Benchmark Validation
       ↓
Develop proposed model
```

**GraphSAGE will be the primary reproduced baseline** because it forms the foundation of the proposed architecture. Additional baselines (GRU, RGCN, and GATv2) will also be reproduced where computational resources and project timeline permit.

The reproduced baseline performance is expected to be reasonably consistent with the published benchmark. Small differences are acceptable because of stochastic optimization, software versions, and implementation details. This is the standard expectation in reproduction studies.

This validation step ensures that:

1. The dataset generation pipeline is correct.
2. The graph construction matches [BASE].
3. The training and evaluation protocol is faithful.
4. Any performance differences observed with the proposed model are attributable to the architectural changes, not dataset or pipeline discrepancies.

## 13. Reproducibility

Every experiment will use:

- Fixed random seeds
- Identical train/validation/test splits
- Identical preprocessing
- Identical graph construction
- Identical evaluation metrics (F1 score on test set, per [BASE])
- Identical hyperparameter settings for reproduced baselines (where applicable)

Only the neural network architecture will vary between experiments. This ensures that all observed performance differences are directly attributable to architectural changes and not to experimental confounds.

## 14. Success Criteria

The benchmark reproduction will be considered successful if the following conditions are satisfied:

1. The IEEE 123-bus simulation reproduces the benchmark protocol described in [BASE].
2. The measured-only graph matches the graph construction algorithm described in [BASE].
3. Baseline architectures achieve performance reasonably consistent with the published benchmark.
4. Dataset statistics (fault distribution, train/validation/test split, graph structure, and feature dimensions) match the reproduced protocol.
5. The proposed architecture is evaluated under the identical benchmark without modifying the dataset generation pipeline.

Only after satisfying these conditions will architectural improvements be evaluated.