# Review 1 — Slide Content

Structured to the rubric in `REVIEW.md`.
**Slides 1–12 = Guide review (8 marks). Slides 13–15 = Panel-only additions (12 marks).**

| Rubric item | Slides |
|---|---|
| Problem statement & research gap | 2, 3, 4 |
| Proposed methodology | 5, 6, 7, 8, 9 |
| Implementation | 10, 11, 12 |
| Regular meetings with guide | 13 |
| SDG mapping + justification | 14 *(panel)* |
| Sem 3 project plan | 15 *(panel)* |

> ⚠️ **Slide 13 needs your input** — I cannot invent meeting dates. Fill the table yourself.

---
---

# PART A — PROBLEM STATEMENT & RESEARCH GAP

## SLIDE 1

**Title:** Dropout-Aware Training for Sensor-Robust Fault Localization in Partially Observable Distribution Grids

```
P K Sangameswar
Guide: Dr. M Anbazhagan

Amrita Vishwa Vidyapeetham, Coimbatore
Review 1 · Semester 2
```

---

## SLIDE 2

**Title:** What Problem Am I Solving, and Why Does It Matter?

```
THE PROBLEM
•  Locate WHICH bus faulted on a distribution feeder
•  Not detection — localization. 26 classes (25 buses + no-fault)
•  Sensors are sparse: only 25 μPMUs on a 123-bus feeder

WHY IT IS A PROBLEM
•  Crews cannot patrol the whole feeder — restoration time is the cost
•  Only 6 of 25 fault buses carry a sensor — 19 are invisible
•  μPMUs are expensive, so full instrumentation is never deployed
•  Real sensors fail: comms loss, GPS-sync loss, power loss, maintenance
```

---

## SLIDE 3

**Title:** What Has Already Been Done

**Table:**

| Approach | What it does | Limitation |
|---|---|---|
| Impedance / travelling-wave | Physics from line parameters | Degrades on unbalanced feeders with DER |
| CNN / LSTM learning | Learns fault signatures | Ignores network topology |
| **STGNN** (Karabulut 2026) | GRU + GNN + soft voting on IEEE-123 | **Assumes all sensors always available** |
| Graph imputation (Zhang 2025) | Reconstructs missing node values | Not evaluated for fault localization |

```
•  STGNNs are the current state of the art for this task
•  Every one of them is evaluated with a fixed, fully available sensor set
```

---

## SLIDE 4

**Title:** Research Gap and Objectives

**Box (highlighted):**
```
RESEARCH GAP
No published work measures what sensor dropout does to STGNN fault
localization, and none proposes a mitigation for it.
```

```
OBJECTIVES
1.  Adopt the STGNN methodology and report our own multi-seed benchmark
2.  Quantify accuracy degradation as μPMUs are removed at inference
3.  Develop a mitigation that is architecture-agnostic and cheap
4.  Test an architectural alternative and report the outcome honestly
5.  Validate every claim statistically, not anecdotally
```

---
---

# PART B — PROPOSED METHODOLOGY

## SLIDE 5

**Title:** Architecture

**Diagram (centred, monospace):**
```
[25 nodes × 20 steps × 6 features]
        ↓   shared GRU-128 per node           (temporal encoder)
        ↓   4 × GNN layers  ← only variable   (spatial reasoning)
        ↓   BatchNorm → ReLU → Dropout → Linear(26)
        ↓   soft vote over 25 nodes → fault location
```

```
•  Five architectures share this template, differing only in the GNN block
•  GRU (no graph) · RGCN · RGSAGE-Mean · RGSAGE-Max · RGATv2
•  6 features per node = 3 phase voltages + 3 availability-mask channels
•  Changing one block at a time isolates the effect of aggregation
```

---

## SLIDE 6

**Title:** Workflow

**Diagram (centred):**
```
OpenDSS simulation  →  windowing + labelling  →  measured-only graph
        ↓
event-level 70/15/15 split  →  train 5 baselines (3 seeds each)
        ↓
evaluate under sensor dropout  →  degradation curves
        ↓
dropout-aware retraining  →  re-evaluate  →  statistical validation
```

```
•  One frozen dataset and one frozen split are reused by every model
•  Dropout masks are paired — every model sees the identical dropped nodes
```

---

## SLIDE 7

**Title:** Algorithms and Mathematical Model

**Left column — Algorithms:**
```
ALGORITHM 1 — Measured-only graph
Substation-rooted spanning tree over the 25 μPMU buses;
two sensors are joined if no third sensor lies between them.

ALGORITHM 2 — Dropout-aware training
for each batch:
    k ~ Uniform{0, 1, 2, 5}
    zero all 6 channels of k random nodes
    forward, backward, clip gradient to 1.0
```

**Right column — Mathematical model:**
```
Temporal encoding
    h_v = GRU(x_v) ∈ ℝ¹²⁸

Soft-vote decision rule
    ŷ = argmax_c  Σ_v  softmax(z_v)_c

Training objective
    L = (1/N) Σ_v  CE(z_v , y)

Availability (used by the gated variant)
    a_i = 1 { max_s max_p  m_i,p(s) > 0 }
```

**Bottom note:**
```
Trained on k ∈ {0,1,2,5} — but evaluated up to k = 10, a level never seen in training
```

---

## SLIDE 8

**Title:** Dataset

**Figure (left half):** `measured_only_graph.png`
Caption: *25 μPMU nodes · red ring = also a fault location (6 of 25)*

**Bullets (right half):**
```
•  IEEE 123-bus feeder simulated in OpenDSS at 1 ms
•  825 scenarios = 25 buses × 11 fault types × 3 resistances
•  50 runs per scenario, load ~ U(0.5, 1.3)
•  41,250 events → 1,650,000 windows
•  Exactly 50 % no-fault by construction
•  Event-level 70/15/15 split — zero leakage verified
•  One sample = 25 nodes × 20 steps × 6 features
```

---

## SLIDE 9

**Title:** Evaluation Strategy and Technical Justification

```
EVALUATION STRATEGY
•  Metric: soft-voted macro-F1 (26 classes), plus per-resistance breakdown
•  Robustness: drop k ∈ {0,1,2,5,10} μPMUs at inference, k=10 is 40 % of the fleet
•  Summary measures: AUDC (area under retention curve) and retention at k=10
•  Every point = 3 training seeds × 3 mask seeds
•  Paired t-tests with Holm correction + cluster bootstrap over 6,175 events

TECHNICAL JUSTIFICATION
•  Graph, not CNN: 19 of 25 fault buses are unmeasured — information must propagate
•  Event-level split: window-level splitting would leak and inflate every number
•  Macro-F1, not accuracy: 26 classes, and rare classes must not be hidden
•  3 seeds: a single-seed robustness claim cannot be separated from seed noise
```

---
---

# PART C — IMPLEMENTATION

## SLIDE 10

**Title:** Implementation — Code and Dataset Preparation

```
CODE  (Python 3.12 · PyTorch · PyTorch Geometric · OpenDSSDirect)
•  fault_localization/  — simulation, processing, graph, models, dropout, experiments
•  scripts/             — 30+ runnable sweeps, baselines and figure builders
•  config/              — YAML per model, one frozen experiment base
•  tests/               — unit and regression tests
•  Git-versioned; every run writes metrics, checkpoint and config

DATASET PREPARATION  (complete)
•  OpenDSS time-stepped simulation, 1 ms, per-event fault injection
•  Windowing → labelling → event-level split → train-only normalization
•  Leakage check: no window of an event appears in two splits — verified 0
•  Frozen corpus reused unchanged by every experiment
```

---

## SLIDE 11

**Title:** Implementation — Baselines and Prototype Modules

```
BASELINE IMPLEMENTATION  (complete)
•  All 5 architectures trained at seeds 42 / 43 / 44
•  Frozen config: depth 4, 30 epochs, grad-clip 1.0, 3 warm-up epochs
•  RGSAGE-Max reaches macro-F1 0.846 ± 0.007

PROTOTYPE MODULES  (complete)
•  dropout/masking.py + evaluator.py   — degradation curves
•  dropout/training.py                 — dropout-aware training (the contribution)
•  models/sdr_stgnn.py                 — availability-gated ablation
•  dropout/targeted.py, persistent.py  — worst-case and per-event stress tests
•  run_imputation_baseline.py          — competing-method comparison
•  run_significance_tests.py           — statistical validation
```

**Box (highlighted):**
```
Review-1 target was "implementation has started" — implementation is COMPLETE.
```

---

## SLIDE 12

**Title:** Results So Far

**Figure (full width, top):** `sdr_degradation_all.png`
Caption: *Baselines collapse as sensors are removed; the mitigation stays flat*

**Table:**

| RGSAGE-Max | k=0 | k=10 | Retained |
|---|---|---|---|
| Standard training | 0.846 | 0.048 | 5.7 % |
| Neighbour-mean imputation | 0.846 | 0.099 | 11.8 % |
| **Dropout-aware training** | 0.839 | **0.674** | **80.3 %** |

```
•  Standard STGNNs lose 94 % of their accuracy at 40 % sensor loss
•  The most accurate architecture is also the most fragile
•  Dropout-aware training: 5.7 % → 80.3 % retention, no inference cost
•  Works on all five architectures (p ≤ 0.027)
•  Our gated architecture matches but does not beat it — reported honestly
```

---

## SLIDE 13

**Title:** Regular Meetings With Guide

> ⚠️ **FILL THIS IN YOURSELF — I do not have your meeting record.**

```
Meeting log
```

| Date | Discussion | Outcome / action |
|---|---|---|
| | Problem finalization and base paper selection | |
| | Dataset scale and simulation protocol | |
| | Architecture depth and stabilization | |
| | Dropout contribution scope | |
| | Results review and paper drafting | |

```
•  Frequency:  ______ (e.g. weekly / fortnightly)
•  Mode:       ______ (in person / online)
•  Key guide inputs that changed the work:
      – ______________________
      – ______________________
```

---
---

# PART D — PANEL-ONLY SLIDES

## SLIDE 14 *(panel)*

**Title:** SDG Mapping and Justification

**Table:**

| SDG | Relevance | Justification |
|---|---|---|
| **7 — Affordable and Clean Energy** | Direct | Faster fault localization shortens outages and reduces energy not supplied; robust monitoring supports DER and rooftop-solar integration on unbalanced feeders |
| **9 — Industry, Innovation and Infrastructure** | Direct | Builds resilient grid infrastructure — the method keeps working when 40 % of monitoring hardware fails, rather than requiring perfect instrumentation |
| **11 — Sustainable Cities and Communities** | Direct | Reliable distribution supply is basic urban infrastructure; faster restoration reduces disruption to households and services |
| 13 — Climate Action | Indirect | Enables higher renewable penetration by making DER-heavy feeders observable |

**Box (highlighted):**
```
Strongest justification: our contribution reduces the sensor coverage needed for
reliable localization — lowering the cost barrier for utilities that cannot
afford dense μPMU deployment.
```

---

## SLIDE 15 *(panel)*

**Title:** Semester 3 Project Plan

**Table:**

| Phase | Work | Output |
|---|---|---|
| 1 | Conference paper submission and revision | ICMLDE 2026 paper |
| 2 | Spatially correlated failures — a comms hub taking out a whole region | New degradation study |
| 3 | Extend worst-case and persistent stress tests to all five architectures | Completed robustness matrix |
| 4 | Full-scale run on the 3.30 M-window corpus | Scale validation |
| 5 | Cross-topology generalization — reconfigured feeder | Transfer results |
| 6 | Thesis writing and journal extension | Thesis + journal draft |

```
•  Immediate next step: spatially correlated dropout — the one failure
   geometry not yet tested
•  Code, dataset and evaluation harness are already in place, so each phase
   reuses the existing pipeline
```
> Add your actual month/date ranges to the first column before presenting.
> **[Correction, 2026-10-15]** Spatially correlated (contiguous) dropout is no longer future
> work — it was run 2026-08-17, 3 seeds, both standard and dropout-aware arms
> (`results/dropout_study_correlated/`; see also `PPT-EVAL-STRATEGY.md` SLIDE E5). Update this
> slide's "immediate next step" line before presenting; row 2 of the plan table above is done,
> not planned.

---
---

# FIGURES USED

| Slide | File |
|---|---|
| 8 | `paper/figures/measured_only_graph.png` |
| 12 | `paper/figures/sdr_degradation_all.png` |

**Optional extras if you have room:**
`sdr_heatmap.png` (model × k grid) · `sdr_mechanism.png` (the gate) ·
`fault_signature.png` (why 10 Ω is hard) · `stgnn_dataset_size_breakdown.png` (dataset flow)

Never use any `*.prebugfix.png` file.

---

# THINGS NOT TO SAY

1. Don't say **"reproduce"** — we adopt the methodology and report our own numbers.
2. Don't call the gated architecture a **win** — it is a tie (p = 0.098).
3. Don't call the accuracy/robustness inversion **significant** — p = 0.076, a trend.
4. Don't promise the **green-feeder** experiment — it was cut from scope.

> Speaker scripts and PowerPoint setup steps: **PPT-SPEAKER-NOTES.md**
> (its slide numbering is the older conference-talk order, not this rubric order).
