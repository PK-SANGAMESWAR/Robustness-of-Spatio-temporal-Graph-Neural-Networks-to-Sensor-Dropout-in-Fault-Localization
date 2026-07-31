# Dataset Specification — STGNN Fault Localization Benchmark (Reproduction + Sensor-Dropout Study)

**Status:** Baseline prototype under investigation — NOT yet an approved reproduction.
**Base paper:** Karabulut et al. (2026), *Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids.*
**Purpose of this document:** Single source of truth for dataset generation. Any deviation from this spec must be logged in `docs/reproduction_log.md` with justification.

---

## 0. Scope and Gating Rule

This dataset supports two things:
1. **Reproduction** of the paper's spatio-temporal baselines (GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2).
2. A **sensor-dropout robustness study** (μPMU dropout at inference + dropout-aware training).

**GATING RULE (do not skip):** No graph/model/dropout artifact is valid until **Section 9 — Gate 1 (Temporal Validity)** passes. Static or duplicated time series invalidate every downstream result.

---

## 1. Power Network

| Item | Value | Source |
|---|---|---|
| Feeder | IEEE 123-bus distribution test feeder | Paper §4.2 |
| Simulator | OpenDSS via `OpenDSSDirect.py` (NOT deprecated PyDSS) | Project decision |
| Feeder model file | Official `IEEETestCases/123Bus/IEEE123Master.dss` shipped with OpenDSS | Do NOT hand-rebuild |
| Nominal voltage | 4.16 kV | Paper §4.2 |
| Base frequency | 60 Hz | Paper §4.2 |

**Requirement:** Load the official `.dss` model. Do not reconstruct buses/lines manually.

---

## 2. Measurement Infrastructure (μPMUs)

| Item | Value | Source |
|---|---|---|
| Number of μPMUs | 25 | Paper §4.2 |
| μPMU bus locations | As listed in paper Table 1 — **copy exact bus IDs into `config/pmu_buses.yaml`** | Paper Table 1 |
| Measured quantity | Three-phase RMS voltage magnitudes (V1, V2, V3) per phase | Paper §4.2 |
| Sampling resolution | 1 ms | Paper §4.2 |
| Per-μPMU phase metadata | Explicit known phase type (1-phase / 2-phase / 3-phase) per device — **stored, not inferred** | Paper Algorithm 1 |

**CRITICAL:** μPMU phase count MUST come from device metadata (`config/pmu_phase_metadata.yaml`), NOT inferred from incident-line phase counts. Inference is a known defect.

**Missing/absent phases:** encode with an explicit **availability mask channel** (see §6), NOT silent 0.0.

---

## 3. Fault Scenarios

| Parameter | Values | Count | Source |
|---|---|---|---|
| Fault locations (buses) | Exact bus list from Table 1 | 25 | Paper Table 1 |
| Fault types | AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC, ABCG | 11 | Paper §4.1 |
| Fault resistances (Ω) | 0.1, 1, 10 | 3 | Paper §4.2 |
| Fault duration | 20 ms | Paper §4.2 |
| Load scaling per bus | L ~ Uniform(0.5, 1.3) p.u., independent per bus | — | Paper §4.2 |
| Runs per scenario | 100 (full) / **[ASSUMPTION: reduced to 20 for prototype — document]** | — | Paper §4.2 |

**Fault injection:** use OpenDSS `Fault` object with correct `phases`, `r`, and bus/phase-node mapping. Phase-node mappings MUST be physically validated (see §9 Gate 3), not left "to be validated."

---

## 4. Temporal Window Definition

| Item | Value | Source |
|---|---|---|
| Recording window per event | 60 ms total | Paper §4.2 |
| Pre-fault duration | 40 ms (40 samples @ 1 ms) | Paper §4.2 |
| Fault duration | 20 ms (20 samples @ 1 ms) | Paper §4.2 |
| Sliding window length S | 20 timesteps (20 ms) | Paper §4.2 |
| Number of sliding windows per event | 40 | Paper §4.2 |
| Window stride | 1 ms (overlapping) | Derived |
| Window endpoint convention | **[ASSUMPTION: drop start-40 window — document in reproduction_log.md]** | Paper silent |

**CRITICAL — Temporal validity:** Each 1 ms sample MUST be produced by genuine time-stepped simulation. Consecutive samples MUST be physically distinct. Copying one snapshot N times is a blocking defect (see §9 Gate 1).

---

## 5. Simulation Execution Requirements (Blocking)

The simulator MUST:
1. Run OpenDSS in **time-series / dynamic mode** (`Set Mode=...` with a 1 ms step), NOT snapshot mode with duplicated solves.
2. Advance simulation time by exactly 1 ms per recorded sample.
3. Apply the fault at t = 40 ms and record its transient evolution across the 20 fault samples.
4. Record via a validated extraction path (monitors actually registered in OpenDSS, or a verified per-step API read). Command strings that are built but never sent are a defect.

---

## 6. Sample / Tensor Format

Per node (μPMU), per timestep:
- Features: `[V1, V2, V3]` (RMS voltage per phase), z-normalized using **train-split-only** statistics.
- **Availability mask:** `[m1, m2, m3]` binary per phase (1 = measured/live, 0 = absent/dropped). This is a SEPARATE channel, never conflated with a zero value.

Per input window tensor:
- Shape: `[num_nodes=25, S=20, F]` where `F = 3 (voltage) + 3 (mask) = 6`.

Per window label:
- Integer class in **26-class space**: 25 fault locations + 1 no-fault class.

---

## 7. Graph Construction (Measured-Only)

| Item | Requirement | Source |
|---|---|---|
| Algorithm | Faithful implementation of paper **Algorithm 1** | Paper §3.x |
| Start point | Substation | Algorithm 1 |
| Traversal | Substation-oriented; three-phase backbone traversal handled separately | Algorithm 1 |
| Connection rule | μPMUs connected by electrical-path reachability; stop at three-phase PMU (not any PMU) | Algorithm 1 |
| Loop handling | Explicit cycle prevention (result must be acyclic / tree-like) | Algorithm 1 |
| Phase handling | Partial-phase μPMUs connected per Algorithm 1 partial-phase rules, using device phase metadata | Algorithm 1 |
| Node count | 25 | Paper |
| Edge validation | Verify edge list against paper Fig. 3 where possible | Paper Fig. 3 |

**Defect to avoid:** unordered BFS from every 3-phase PMU with only duplicate-edge prevention produces cycles (observed cycle rank 11). This is NOT Algorithm 1.

**Edge features (`edge_attr`):** electrical line attributes (R, X, Z) MAY be produced and stored, but the **reproduction baselines use node features only** — do not feed edge_attr to baseline models. Config/docs must not present edge_attr as active baseline input.

---

## 8. Full-Topology Graph (for topology-comparison reproduction)

| Item | Value | Source |
|---|---|---|
| Node count | 128 (full feeder topology) | Paper |
| Unmeasured-node handling | Mask unmeasured-node outputs in both loss and soft-voting | Paper |

Required to reproduce the paper's measured-only vs. full-topology comparison.

---

## 9. Feeder Configurations

| Config | Definition | Source |
|---|---|---|
| **Default** | Base IEEE 123-bus switch configuration | Paper |
| **Green (reconfigured)** | Open tie switch **60–160**, close switch **54–94** | Paper §4.3 |

Both configurations MUST be generated. The green config is the paper's most challenging scenario and is required for the dropout-recovery claim.

---

## 10. Dataset Scale and Splits

| Item | Value | Source |
|---|---|---|
| Target corpus (full) | ~2.5M windows | Paper |
| Prototype corpus | **[ASSUMPTION: reduced scale — document exact size]** | Project |
| Class balance | ~50% no-fault, remainder across 25 fault locations | Paper |
| Split | 70 / 15 / 15 (train/val/test) | Paper |
| Split level | **Event-level** (all windows of one event stay in one split — no window leakage) | Paper |
| Normalization | Train-split statistics only | Paper |

---

## 11. Sensor-Dropout Study Artifacts (Novelty — REQUIRED)

These do NOT change the dataset; they are runtime policies applied to the frozen dataset.

| Artifact | Requirement |
|---|---|
| Dropout selection | Randomly select k μPMU nodes to drop; k ∈ {0, 1, 2, 5, 10} |
| Missingness representation | Set dropped node's availability mask to 0 AND define graph-message policy (drop its edges / zero its outgoing messages) — do NOT merely zero features |
| Inference-time evaluation | Evaluate every trained baseline across all k; produce F1-vs-k **degradation curves** |
| Dropout-aware training | Randomly mask nodes during training; re-evaluate across all k |
| Noise (optional) | Additive Gaussian noise on voltages at test time |
| Seeds | ≥3 seeds for every reported model/condition; report mean ± std |

---

## 12. Acceptance Gates (IDE self-check — all must pass in order)

**Gate 1 — Temporal Validity (BLOCKING):**
- [ ] Consecutive 1 ms samples are physically distinct (not bit-identical copies).
- [ ] Pre-fault samples vary due to load dynamics; fault samples show real transient evolution (difference >> numerical noise, i.e. not ~1e-8).
- [ ] A baseline **GRU reaches credible macro-F1** (paper ballpark ≈0.8+, NOT ~0.026) on frozen regenerated data.

**Gate 2 — Graph Correctness:**
- [ ] Measured-only graph is acyclic (cycle rank 0) and matches Algorithm 1.
- [ ] μPMU phase types come from metadata, not inference.
- [ ] Edge list validated against Fig. 3 where possible.

**Gate 3 — Fault Physics:**
- [ ] Each fault type's OpenDSS phase-node mapping physically validated (correct phases faulted; expected voltage sag pattern per type).

**Gate 4 — Baseline Reproduction:**
- [ ] GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2 all trained on identical frozen data, ≥3 seeds, credible F1.
- [ ] RGATv2 matches paper Eq. (8) attention; attention dropout applied to normalized attention coefficients.

**Gate 5 — Configurations:**
- [ ] Both default and green feeder datasets generated and validated.

**Gate 6 — Dropout Study:**
- [ ] Missingness mask + message policy implemented.
- [ ] Degradation curves produced for all baselines.
- [ ] Dropout-aware training implemented and compared, ≥3 seeds.

**No dropout-robustness claim may be published until Gates 1–5 pass.**

---

## 13. Provenance and Versioning

- [ ] Graph and dataset are versioned **together**; regenerate all processed files after any simulation/graph fix.
- [ ] `build_dataset.py` builds/validates the graph rather than consuming whatever `edge_index.npy` happens to exist.
- [ ] Every `[ASSUMPTION]` in this doc is logged in `docs/reproduction_log.md` with justification.