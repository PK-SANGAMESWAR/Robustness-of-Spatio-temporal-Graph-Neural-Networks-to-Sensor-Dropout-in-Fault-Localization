# Review — Phase 2 Slide Content: Physics-Consistent STGNN (PC-STGNN)

Slide order follows `REVIEW-new.md` (13 items). **Only completed and verified work is stated as a
result.** Anything still running or planned is labelled **[IN PROGRESS]** or **[PLANNED]**.
Every number below comes from a file in `results/` or a closed gate in `PC-STGNN/`.
**Last updated 2026-10-05** (S4.3ᶜ result, test-output quarantine, S4.3ᵈ auxiliary loss, S4.3ᵉ started).
**[Correction, 2026-10-15]** Several "in progress"/"running" status markers below predate
S4.3ᵉ's completion (2026-10-08, selected λ₁=0, λ₂=0.1) and the start of S4.4/Gate 4 training
(2026-10-09, now running). Fixed inline at Slides 5, 6a, 8f, 12, and "Next decision points"
below; the rest of the doc's content is unaffected.

> **Terminology.** The project's own rule: call it **physics-consistent**, never "PINN". There
> is no PDE residual and no KCL/KVL or power-flow loss. The physics is a **fixed operator**, a
> table of fault signatures solved offline in OpenDSS. If the panel asks "is this a PINN?", say:
> *"No. A PINN penalises a differential-equation residual. Here we compare measurements against
> simulator-solved fault signatures. The physics is fixed, never learned, and it adds one scalar
> parameter to the network."*

> **⚠ Two slides need your input:** Slide 10 (Domain Expertise course) and Slide 11 (Technical
> Enrichment Activities). Neither is recorded anywhere in the repo, so I have left placeholders
> rather than invent them.

---

## SLIDE 1 — Title

**Physics-Consistent Spatio-Temporal Graph Neural Networks for Sensor-Robust Fault Localization in
Partially Observable Distribution Grids**

```
P K Sangameswar
Guide: Dr. M Anbazhagan
Amrita Vishwa Vidyapeetham, Coimbatore
Review · Semester 3 (Phase 2: continuation of the Phase-1 dropout-robust STGNN)
```

---

## SLIDE 2 — Introduction and Motivation

**Where Phase 1 ended (purely data-driven, no physics):**
- Task: locate **which bus** faulted on the IEEE 123-bus feeder. 26 classes: 25 fault locations + no-fault
- Only **25 µPMUs** on 123 buses. Just **6 of the 25** fault buses carry a sensor
- Phase-1 contribution: **dropout-aware training** for losing sensors
  - Robustness AUDC rose from 0.402 to **0.913**
  - Retention at 10 lost sensors rose from 5.7 % to **80.3 %**
  - Zero inference cost: same 95,258 parameters, same latency

**Why Phase 2 is needed. Three failure modes Phase 1 measured but could not fix:**

| Stressor | Phase-1 measured result |
|---|---|
| **Measurement noise** | RGSAGE-Max falls 0.846 → 0.345 at σ = 1e-2 p.u. (the IEEE C37.118 1 % TVE cap). Dropout-aware training gives **no benefit** (−0.008 to −0.034, all CIs span 0) |
| **Unseen fault resistance** | Trained on R ∈ {0.1, 1, 10} Ω. On R ∈ {5, 20} Ω the network collapses **0.839 → 0.089** F1 |
| **Worst-case sensor loss** | Losing the μPMU nearest the fault drops F1 to **0.448** (vs 0.736 under random loss at k = 1) |

**Motivation in one line:** the network and the physics fail under *different* stressors. Phase 2
tests whether a fixed physics operator covers the cases where learning alone breaks.

---

## SLIDE 3 — Problem Statement

> Given voltage-magnitude windows from 25 µPMUs, some missing, noisy or out-of-distribution,
> localize the faulted bus. Inject electrical physics in a way that is **(a) provably neutral when
> switched off**, **(b) cannot leak labels**, and **(c) adds no meaningful capacity**, so that any
> gain is attributable to physics.

**Scope:**
- **Input:** a 20 ms window `[25 nodes × 20 samples × 6 channels]`. Channels are 3 RMS voltage magnitudes (one per phase) plus 3 availability masks
- **Output:** one of 26 classes, soft-voted across nodes
- **Not claimed:** KCL/KVL or power-flow residuals, physics-informed temporal dynamics, or a new GNN layer. The corpus is quasi-static and magnitude-only, so it cannot support these claims

**Research questions:**
1. Does a fixed physics operator carry information the dropout-aware network lacks?
2. Can that information be fused into training and inference **without** confounding capacity, auxiliary tasks or noise augmentation?
3. Where does the gain appear? Under noise, unseen resistance, heavy dropout?

---

## SLIDE 4 — Proposed Methodology: PC-STGNN Architecture (1 slide)

**Core idea:** keep the Phase-1 backbone exactly as it is. Add a fixed physics residual ρ and use
it in two places:
1. **At inference:** reweight each node's class distribution (a product of experts per node)
2. **In training:** two physics-consistency loss terms. **λ₁** scores the available sensors; **λ₂** scores the held-out sensors (leave-k-out)
3. **In training (added 2026-10-05, S4.3ᵈ):** an **auxiliary cross-entropy on the network-only
   logits**, weight **γ = 1**, fixed and not tuned. It forces the network branch to localize on its
   own instead of handing localization to the physics term (see Slide 8f)

### Figure A — Full architecture (academic level of detail)

**Use this rendered figure on the slide.**
> **⚠ The rendered PNG/SVG predates the γ term.** Its loss box shows CE + λ₁ + λ₂ only. Either
> regenerate it after adding `+ γ·CE(z, y)` to `scripts/build_pcstgnn_arch_figure.py`, or add
> the γ term as a text box on the slide. The Mermaid and plain-text versions below are updated.
- Files: [paper/figures/pcstgnn_architecture.png](paper/figures/pcstgnn_architecture.png) (3840 × 2344) and [paper/figures/pcstgnn_architecture.svg](paper/figures/pcstgnn_architecture.svg) (vector, scales cleanly in PowerPoint)
- Regenerate with `uv run python scripts/build_pcstgnn_arch_figure.py`

![PC-STGNN architecture](paper/figures/pcstgnn_architecture.png)

The Mermaid source below is the same diagram in editable form:

```mermaid
flowchart TB
    %% ===================== OFFLINE =====================
    subgraph OFF["① OFFLINE — Physics operator construction (fixed, never trained)"]
        direction TB
        FEED["IEEE 123-bus feeder<br/>OpenDSS model"]
        GRID["Hypothesis grid<br/>τ: 11 fault types<br/>R: 18 resistances, 0.05–30 Ω (log + {0.1,1,10})<br/>L: 17 load levels, 0.5–1.3, step 0.05"]
        SOLVE["Snapshot solves<br/>84,167 solves · 0 non-converged · 0 NaN<br/>load frozen at L × 0.99953"]
        TAB["Signature table μ[f, τ, R, L, i, p]<br/>25 loc × 11 × 18 × 17 × 25 nodes × 3 phases<br/>+ no-fault μ₀[L, i, p]<br/>flattened: 26 classes × 3366 hyp × 75 channels"]
        FEED --> SOLVE
        GRID --> SOLVE
        SOLVE --> TAB
    end

    %% ===================== INPUT =====================
    subgraph INP["② INPUT"]
        direction TB
        X["Window X ∈ ℝ^(B×25×20×6)<br/>ch 0–2: RMS |V| per phase (z-scored)<br/>ch 3–5: availability mask m ∈ {0,1}"]
        AUG["Dropout augmentation (training only)<br/>k ~ U{0,1,2,5} per batch<br/>zero all 6 channels of k random nodes"]
        XD["X̃ (masked input)"]
        XC["X (clean copy) + drop mask D ∈ {0,1}^(B×25×3)<br/>(training only)"]
        X --> AUG
        AUG --> XD
        AUG --> XC
    end

    %% ===================== LEARNED BACKBONE =====================
    subgraph NET["③ LEARNED BACKBONE — identical to Phase-1 dropout-aware RGSAGE-Max"]
        direction TB
        GRU["Temporal encoder<br/>shared GRU, hidden 128, per node over S = 20<br/>H⁰ ∈ ℝ^(B×25×128)"]
        SAGE["Spatial block: 4 × RGSAGE-Max layers, hidden 64<br/>h ← W·[h ‖ max_{j∈N(i)} h_j], ReLU between layers<br/>measured-only graph: 25 nodes, 24 edges (spanning tree)<br/>H⁴ ∈ ℝ^(B×25×64)"]
        HEAD["Node classifier<br/>BN → ReLU → Dropout(0.35) → Linear(64→26)<br/>z ∈ ℝ^(B×25×26)"]
        GRU --> SAGE --> HEAD
    end

    %% ===================== PHYSICS BRANCH =====================
    subgraph PHY["④ PHYSICS BRANCH — function of the input only, no gradient"]
        direction TB
        OBS["Observation<br/>s = last timestep, de-normalised to p.u. ∈ ℝ^(B×25×3)<br/>A = {(i,p): m_ip = 1}"]
        RES["Profiled residual, per class f = 0…25<br/>r_f(A) = min over (τ,R,L) of Σ over (i,p)∈A (s_ip − μ_ip)² / (σ_ε² + σ₀²)<br/>σ_ε = 7.2555e-4 p.u. (calibrated on train)"]
        RHO["Shifted, halved residual<br/>ρ_f(A) = (r_f − min_f' r_f') / 2   ∈ ℝ^(B×26)<br/>hyp*_f = argmin hypothesis per class"]
        RHOD["Held-out residual (training only)<br/>ρ_f(D): score hyp*_f (fitted on A only)<br/>on the dropped channels D, using clean s"]
        OBS --> RES --> RHO
        RHO --> RHOD
    end

    %% ===================== FUSION =====================
    subgraph FUS["⑤ FUSION + DECISION"]
        direction TB
        PROD["Per-node product of experts<br/>z'_(n,f) = z_(n,f) − β · ρ_f(A)<br/>same ρ broadcast to all 25 nodes · β learned scalar, init 0"]
        VOTE["Soft vote over nodes<br/>P(f|X) = Σ_n softmax(z'_n)_f, normalised<br/>ŷ = argmax_f P(f|X)"]
        PROD --> VOTE
    end

    %% ===================== LOSS =====================
    subgraph LOSS["⑥ TRAINING OBJECTIVE"]
        direction TB
        L["ℒ = CE(z', y)<br/>+ λ₁ · Σ_f P(f|X) · ρ_f(A)    (physics consistency on available sensors)<br/>+ λ₂ · Σ_f P(f|X) · ρ_f(D)    (leave-k-out consistency on dropped sensors)<br/>+ γ · CE(z, y)    (auxiliary loss on network-only logits, γ = 1)<br/>λ terms masked off on w = 20 windows (no post-fault sample)"]
    end

    XD --> GRU
    XD -.-> OBS
    XC -.-> RHOD
    TAB -.-> RES
    TAB -.-> RHOD
    HEAD --> PROD
    RHO -.-> PROD
    VOTE --> L
    HEAD -.->|"network-only logits z"| L
    RHO -.-> L
    RHOD -.-> L
    VOTE --> OUT["Predicted fault location ŷ ∈ {0 = no fault, 1…25}"]

    classDef fixed fill:#eef3fb,stroke:#3b6aa0,stroke-width:1px;
    classDef learned fill:#fdf1e6,stroke:#b5651d,stroke-width:1px;
    classDef train fill:#f2f2f2,stroke:#777,stroke-dasharray: 4 3;
    class FEED,GRID,SOLVE,TAB,OBS,RES,RHO,RHOD fixed;
    class GRU,SAGE,HEAD,PROD learned;
    class AUG,XC train;
```

**Legend:**
- **Solid arrows:** gradient flows
- **Dashed arrows:** constants. No gradient reaches ρ, μ, μ₀ or σ_ε
- **Blue:** fixed physics. **Orange:** learned. **Grey dashed:** training-time only

### Figure A′ — Same architecture as a plain-text block diagram (to redraw in PowerPoint)

```
 OFFLINE (fixed)                                    ONLINE
 ───────────────                                    ──────
 IEEE-123 OpenDSS ─┐                 X [B,25,20,6] ──► Dropout aug (train: k∈{0,1,2,5}) ──► X̃
 Grid τ×R×L ───────┼─► 84,167 solves                                 │                        │
 (11×18×17)        │   ─► μ [26 × 3366 × 75]          ┌──────────────┘                        │
                   │          │                       ▼                                       ▼
                   │          │       ┌──────── LEARNED (Phase-1 backbone) ────────┐   ┌─ PHYSICS (no grad) ─┐
                   │          │       │ GRU-128 (per node, 20 steps)               │   │ s = last step (p.u.)│
                   │          │       │   ▼                                        │   │ A = mask = 1        │
                   │          │       │ 4 × RGSAGE-Max (hidden 64, 25-node tree)   │   │ r_f = min_{τ,R,L}   │
                   │          └──────►│   ▼                                        │   │  Σ_A (s−μ)²/σ²      │
                   │                  │ BN→ReLU→Drop(0.35)→Linear 26  ⇒ z [B,25,26]│   │ ρ_f = (r_f−min r)/2 │
                   │                  └───────────────────────┬────────────────────┘   └─────────┬───────────┘
                   │                                          ▼                                  │
                   │                    z'_(n,f) = z_(n,f) − β·ρ_f(A)   ◄────────────────────────┘
                   │                                          ▼
                   │                    Soft vote: P(f|X) = Σ_n softmax(z'_n)  ─►  ŷ = argmax
                   │                                          ▼
                   └───────────► ρ(D) on dropped sensors (train only, uses clean X + hyp* from A)
                                                              ▼
                     ℒ = CE(z′) + λ₁·Σ_f P·ρ_f(A) + λ₂·Σ_f P·ρ_f(D) + γ·CE(z)   (γ = 1; λ terms skip w = 20)
```

**Design guarantees (all verified, see Slide 7):**
- **Null reduction.** λ₁ = λ₂ = β = 0 reproduces the Phase-1 model **byte for byte**
- **Auxiliary loss is neutral when off.** With γ = 0 the new code gives the same loss and the same gradients as the old code (difference exactly 0 on all 17 parameter tensors)
- **One extra parameter.** The γ term adds no parameters. β takes the model from 95,258 to 95,259 parameters, so there is no capacity confound
- **No leakage.** The physics reads the input tensor only. No label, window index, R, type or load ever enters ρ
- **Neutral at initialisation.** β starts at 0, so the model starts as the baseline

---

## SLIDE 5 — Implementation Progress

**Built as a stage-gated pipeline.** No GPU training happens before the cheap, training-free
checks pass.

| Stage | Gate | Status | What was delivered |
|---|---|---|---|
| A — Audit of Phase-1 assets | G0 | ✅ Done | Phase-1 results reproduced **exactly at k = 0**. Evaluation deterministic (repeat run differs by 0.00e+00). Whole-corpus window census and quasi-static check (settled level flat to 1.1e-7 p.u.) |
| B — Physics operator | G1 | ✅ **Closed, 5/5 criteria** | 84,167-solve signature table, 0 failures. Fidelity checks passed (Slide 7) |
| C — Training-free go/no-go | G2 | ✅ **Closed: GO** (2026-09-29) | Identifiability, confusability, B1/PSM baselines, OOD contrast, inference-only fusion |
| D — Infrastructure | G3 | ✅ **Closed, 5/5** | 6 failure injectors, incl. a new **silent-failure** injector. Residual layer. Physics-imputation baseline. ρ cache |
| E — PC-STGNN training | G4 | 🔶 **In progress** | S4.1 null test ✅ · S4.2 gradient audit ✅ · S4.3 λ tuning ✅ (6 runs; provisional, network collapsed) · S4.3ᶜ noise diagnostic ✅ (unresolved) · S4.3ᵈ auxiliary loss ✅ (**fixed**) · S4.3ᵉ noise-augmented re-tuning ✅ **done 2026-10-08** (selected λ₁=0, λ₂=0.1; see Slide 8f) · **S4.4 core ablation / Gate 4 🔶 running** (started 2026-10-09, amended to 30 epochs 2026-10-12; 3 of 6 runs complete as of 2026-10-15) |
| F, G, H | G5 | ⏳ Planned | Trust gate, full evaluation, writing |

**Code written this phase** (all in `fault_localization/`, tested):
- `physics/`: signature solver, table builder, observation map, residual (direct and Gram forms), ρ cache
- `models/pc_stgnn.py`: the PC-STGNN model, with ablation switches for every planned arm (fusion, shuffled physics, type head, learned decoder)
- `models/common.py::pc_loss`: CE + λ₁ + λ₂ objective, with the w = 20 loss mask
- `dropout/silent.py`: stuck, bias, Gaussian and phase-swap corruptions that leave the mask at 1
- `dropout/training.py::PhysicsDropoutLoader`: exposes the clean batch and drop mask that λ₂ needs
- `experiments/`: λ₁/λ₂ training path wired into the Phase-1 runner. Pre-existing configs are untouched
- **Noise-augmented training:**
  - σ is drawn per window from {0, 5e-4, 2e-3, 1e-2} p.u.
  - The physics term uses the same σ that was injected into each window
  - The dropout masks are unchanged by adding noise; a test proves this
- **Collapse monitor:** logs the network branch's F1 alone, with the physics term removed, every epoch
- **Auxiliary network-only loss** (`pc_loss.aux_net_loss_weight` = γ). Default 0; a null test proves γ = 0 changes nothing
- **Collapse early stop:** a fusion run whose network branch is still collapsed at epoch 3 is aborted, not completed
- **Test-split lock for tuning:** `training.skip_test_eval` stops validation-only runs from ever loading the test split (see Slide 7)
- **Two latent CUDA bugs found and fixed** in the physics path (`np.asarray` called on GPU buffers). They were invisible until the first GPU training run
- **73 / 73 tests pass**, including 7 new noise-augmentation tests

---

## SLIDE 6 — Standard Algorithms: Algorithms Used and Compared, and Evaluation Metrics

Split over two slides: **6a** algorithms, **6b** metrics. Phase 1 = the dropout-robustness paper
(`paper/overleaf.tex`); Phase 2 = PC-STGNN. Phase-1 numbers are test split, mean over training
seeds 42/43/44. Phase-2 physics numbers are validation split (the test split is sealed).

### SLIDE 6a — Algorithms used and compared

**Common template (all deep models):** per-node GRU temporal encoder (hidden 128) → spatial GNN
(4 layers, hidden 64) → node classifier → **soft vote** across the 25 µPMU nodes.

| Family | Algorithm | Phase | Role | Clean macro-F1 (k = 0) |
|---|---|---|---|---|
| Temporal only | **GRU** | 1 | no-graph floor | 0.286 |
| Spectral GNN | **RGCN** (GCN, Kipf & Welling) | 1 | baseline | 0.712 |
| Attention GNN | **RGATv2** [Brody 2022] | 1 | baseline | 0.522 |
| Inductive GNN | **RGSAGE-Mean** [Hamilton 2017] | 1 | baseline | 0.786 |
| Inductive GNN | **RGSAGE-Max** | 1 | best baseline; backbone of every later model | **0.846** |
| Classical ML | **Logistic regression**, **Random forest** (375 summary features) | 1 | non-deep, non-graph check (trained on 8 % of the training data) | 0.713 / 0.633 |
| Imputation | **Neighbour-mean fill** (mean of available 1-hop neighbours) | 1 | impute-then-infer baseline | — |
| Robust training | **Dropout-aware training** (mask k ∈ {0,1,2,5} nodes during training) | 1 | **Phase-1 contribution** | 0.839 |
| Robust architecture | **SDR-STGNN** (availability-gated aggregation) | 1 | ablation: matches, does not beat it | 0.839 |
| Physics, no learning | **B1ᵃ** rank-1 matched filter | 2 | direction-only floor | 0.162 (val) |
| | **B1** nearest signature at nominal load | 2 | no profiling | 0.649 (val) |
| | **B2 / PSM** profiled signature matching, argmin over (τ, R, L) | 2 | physics-only localizer | 0.772 (val) |
| Imputation | **Physics fill** (signature μ at the PSM hypothesis) | 2 | physics-consistent imputation | — |
| Proposed | **PC-STGNN** (backbone + fixed physics residual ρ, β fusion, λ₁/λ₂ losses, γ aux loss) | 2 | **Phase-2 contribution** | λ selected (S4.3ᵉ, 2026-10-08); Gate 4 (S4.4) training **[Correction, 2026-10-15]** |

**Stress conditions every model is compared under:**
- **Random sensor dropout:** k ∈ {0, 1, 2, 5, 10} of 25 µPMUs (0–40 %), redrawn per window
- **Targeted (worst-case) dropout:** the k µPMUs nearest the fault
- **Persistent dropout:** one dropped set held for all 40 windows of an event
- **Contiguous outage:** a connected feeder segment goes dark
- **Measurement noise:** σ ∈ {5e-4, 2e-3, 1e-2} p.u. (1e-2 = the IEEE C37.118 1 % TVE limit)
- **Phase 2 adds:** unseen fault resistance (R ∈ {5, 20} Ω) and silent failures (stuck, bias, phase swap with the mask still 1)

**Headline comparison (Phase 1, RGSAGE-Max backbone, random dropout):**

| Method | F1 at k = 10 | Ret@10 | AUDC |
|---|---|---|---|
| Standard training | 0.048 | 5.7 % | 0.402 |
| + Neighbour-mean imputation | 0.099 | 11.8 % | 0.476 |
| SDR-STGNN | 0.631 | 75.2 % | 0.892 |
| **Dropout-aware training** | **0.674** | **80.3 %** | **0.913** |

### SLIDE 6b — Evaluation metrics

| Metric | Definition | Why |
|---|---|---|
| **Macro-F1 (primary)** | 26 classes (25 fault buses + no fault), one prediction per 20 ms window, soft-voted over nodes: P(f) = Σₙ softmax(zₙ)_f | balanced across classes; same as Karabulut et al. |
| Per-window node F1 | F1 of each node's own prediction, before voting | diagnostic only |
| **Degradation curve F1(k)** | macro-F1 with k ∈ {0,1,2,5,10} sensors dropped, 3 mask seeds, identical masks for every model | robustness to sensor loss |
| **Retention R(k)** | F1(k) / F1(0). **Ret@10** = R(10) | share of clean accuracy kept |
| **AUDC** (relative) | (1/10) ∫₀¹⁰ R(k) dk, trapezoidal over k ∈ {0,1,2,5,10}; range 0–1 | one-number robustness score |
| **AUDC_abs** | the same integral on raw F1(k) | the relative form flatters weak models (RGATv2: 0.626 relative vs 0.324 absolute) |
| Localization / detection split | mean F1 over the 25 fault classes vs F1 of the no-fault class | checks the macro-average is not inflated by the easy class |
| F1 by fault resistance | separate F1 for R = 0.1, 1, 10 Ω | 10 Ω is the hardest slice |
| No-fault precision / recall | class-0 metrics | false-alarm behaviour |
| **Reference bounds** | always-"no fault" classifier = **0.0256**; ceiling **0.9744** (2.5 % of fault windows have no post-fault sample) | how bad is "bad", how good can it get |
| Cost | parameters, inference latency (ms/event), throughput, activation memory, training s/epoch | "no inference cost" is measured, not assumed |
| **Statistics** | paired two-sided t-test over 3 training seeds, 95 % CI, Holm–Bonferroni per family, Cohen's d_z; cluster bootstrap over **events**, not windows | windows of one event are correlated |
| *Phase 2:* physics diagnostics | ρ top-2 margin, identifiability D(f₁, f₂), fault-type τ\* and load L\* recovery | is the physics informative? |
| *Phase 2:* **collapse guard** | network-only macro-F1 (physics term removed) and its "no fault" share. Ineligible if F1 ≤ 0.10 or > 95 % "no fault" | catches a model that hands all localization to physics |
| *Phase 2:* fusion decomposition | accuracy of fused / network-only / PSM, and agreement fused = PSM, fused = network, on fault and no-fault windows | how the model actually decides |

**Selection rules, fixed in advance:**
- **Checkpoint within a run:** best validation F1 at k = 0 (Phase-1 convention)
- **λ across runs (Phase 2):** best validation F1 averaged over k ∈ {0, 1, 2, 5, 10}, among guard-valid arms
- The test split is never used for selection; Phase-2 tuning runs never load it

---

## SLIDE 7 — Tools, Methods and Demonstrated Expertise

**Tools:**
- **Simulation:** OpenDSS via OpenDSSDirect.py. Daily-mode 1 ms event simulation; snapshot solves for the operator
- **Learning:** Python 3.12, PyTorch 2.5.1 (CUDA 12.1), PyTorch Geometric
- **Numerics and analysis:** NumPy, SciPy, pandas, scikit-learn, NetworkX, matplotlib, seaborn
- **Engineering:** uv (locked environment), pytest, Git, YAML configs, LaTeX/Overleaf
- **Hardware:** a single NVIDIA T1000 (8 GB)

**Methods that demonstrate rigour (each was carried out, not just planned):**
- **Byte-identical null test (S4.1).** The new model with the physics switched off reproduced all 30 epochs of the Phase-1 run exactly. Max |Δ| = 0.0 on loss and validation F1
- **Finite-difference gradient audit (S4.2), 12/12 checks.** Run on 3 deliberately chosen batches: a w = 20 window, k = 10, and k = 0 with |D| = 0. It proves the λ₁ and λ₂ terms carry non-zero gradient to the weights and that the physics table carries none
- **Pre-registration.** Hypotheses and analysis were committed *before* any test-split result was viewed (`14_PREREGISTRATION.md`, commit `9fb50b8`)
- **Full-population checks, not samples.** Example: a 400-event sample missed 3 control-driven settling outliers that the full 41,250-event scan found
- **Metric-consistency check.** A suspicious PSM-vs-network gap was decomposed and shown to be a metric mismatch. The claim was retracted rather than published
- **Cache verification on real training batches.** The k = 0 ρ cache matches the live computation to **1.6e-5 relative error** with 100 % hypothesis agreement. Lookup is by window ID, because the cache order differs from the split order
- **Checking how the model actually decides, not just its score.** The tuning winner's F1 looked good, so we split its predictions into network-only, physics-only and fused. That check found the collapse in Slide 8f; a headline F1 would have hidden it
- **Rules fixed in advance of results.** The noise diagnostic's decision rule and the collapse guard were written into the stage document **before** any noise-augmented run started. The S4.3ᵉ re-tuning was committed (`272ba01`) before its first run
- **Null test for every new loss term.** Before the auxiliary loss was used, γ = 0 was shown to reproduce the old loss and all 17 parameter gradients exactly on a fixed batch
- **Test-split leak found and quarantined.** The training runner always ran a test evaluation at the end, so the 8 validation-only tuning/diagnostic runs wrote test files they should not have. All 24 files were moved to a quarantine folder (nothing deleted), an audit showed no selection or verdict used them, and the runner now has a switch that never loads the test split. Reported openly in the stage document

---

## SLIDE 8 — Preliminary Verification and Results

### 8a. Physics operator fidelity: Gate 1 ✅ (5/5)

| Check | Criterion | Measured |
|---|---|---|
| Table vs corpus bias | at discretisation scale | median \|e\| = **1.31e-3 p.u.** |
| Fault-type recovery τ\* | > 90 % | **90.9 %** |
| Load recovery L\* (within 1 step) | > 90 % | **97.3 %** (was 38.4 % before an R-grid fix we found) |
| Residual floor r̄ at the true class | ∈ [0.3, 3] | **1.000** |
| Discretisation loss | < 0.01 F1 | **−0.0013** |

### 8b. Is the physics informative? Identifiability (S2.1, S2.2)
- Only **1.5 % → 1.8 %** of location pairs are indistinguishable from k = 0 to k = 10
- R = 0.1 Ω and R = 1 Ω: **fully separable** at every k. The residual limit sits at **R = 10 Ω**
- The table agrees with empirical nearest-neighbour confusion: **Spearman ρ = −0.655, p = 4e-38**. The "table wrong" case occurred **0 / 300** times

### 8c. Physics-only baseline ladder (validation, 26-class, no learning)

| Method | F1 at k = 0 | F1 at k = 10 | AUDC (abs) |
|---|---|---|---|
| B1ᵃ rank-1 matched filter | 0.162 | 0.162 | — |
| B1 nearest signature | 0.649 | 0.601 | 0.626 |
| **B2 PSM (profiled)** | **0.772** | **0.749** | **0.762** |
| Dropout-aware RGSAGE-Max (Phase 1) | **0.855** | — | — |

- Profiling over (τ, R, L) adds **+0.123 F1** over B1
- **The network still beats physics in-distribution** (0.855 vs 0.772). Physics does not win on its own turf

### 8d. The decisive Gate-2 evidence: they fail under different stressors

| Stressor | Network | Physics |
|---|---|---|
| **Unseen resistance** (OOD pilot R ∈ {5, 20} Ω, 3 seeds) | **0.839 → 0.089** (collapses) | PSM **0.772 → 0.539** (holds) |
| **Noise, via inference-only fusion** z − βρ on the existing checkpoint* | — | **+0.213 F1** at σ = 1e-2 · **+0.153** at σ = 2e-3 |
| **Heavy dropout, k = 10**, inference-only fusion* | — | **+0.166 F1** |
| **10 Ω faults**, inference-only fusion* | — | **+0.086 F1** |

\* Seed 42 only, 30,000-window validation subsample. Fusion was applied to a checkpoint that was
**never trained** for it. Treat these as preliminary; the 3-seed trained version is Stage E.
→ **Gate 2: GO.**

### 8e. Physics-consistent imputation (3 seeds, full test split)

| k | Neighbour-mean fill | **Physics fill** |
|---|---|---|
| 1 | 0.739 | **0.775** |
| 2 | 0.637 | **0.701** |
| 5 | 0.336 | **0.479** |
| 10 | 0.099 | **0.225** |
| AUDC (abs) / Ret@10 | 0.403 / 11.8 % | **0.508 / 26.6 %** |

- The physics fill wins at every k > 0
- Its seed variance is larger at high k, which we report as is

### 8f. PC-STGNN training: verification passed, tuning complete, a degenerate solution found

**Verification:**
- **S4.1 null test ✅.** λ₁ = λ₂ = β = 0 reproduces Phase-1 exactly (best validation F1 0.8554909744946672 in both runs)
- **S4.2 gradient audit ✅ 12/12.** β, λ₁ and λ₂ all receive non-zero gradient; μ and σ_ε receive none

**S4.3 λ tuning ✅ complete.** Seed 42, 15 epochs, β learned, dropout augmentation on, noise-free.
Validation split only. 6 runs; the pre-set boundary rule added λ₁ = 0.0333.

| λ₁ | λ₂ | Val F1, avg over k (λ rule) | Val F1 at k = 0 (checkpoint rule) | β learned |
|---|---|---|---|---|
| **0.0333** | **0** | **0.8501** | 0.8533 | 0.022 |
| 0.1 | 0 | 0.8277 | 0.8323 | 0.038 |
| 0.3 | 0 | 0.8026 | 0.8049 | 0.051 |
| 1.0 | 0 | 0.8218 | 0.8293 | 0.079 |
| 0.0333 | 0.1 | 0.8421 | 0.8452 | 0.029 |
| 0.0333 | 1.0 | 0.8200 | 0.8176 | 0.048 |

Figure F-E1: `paper/figures/fe1_lambda1_sweep.png`.

**The key finding: on noise-free data the network learned to defer entirely to physics.**
Measured on the full validation split (248,000 windows), all six runs:

| What we measured | Result |
|---|---|
| Network branch alone, physics term removed | votes "no fault" on **99.9–100 %** of windows. F1 = **0.0256** = the trivial always-"no fault" score |
| Fused model on **fault** windows (best run) | agrees with physics-only PSM **95.9 %** of the time. Accuracy 85.6 % vs PSM 85.8 % |
| Fused model on **no-fault** windows (best run) | **87.0 %** correct vs PSM's **49.8 %**. This is where the whole gain over PSM comes from |

- Trained noise-free, PC-STGNN became **PSM plus a learned "no fault" bias**. Our stage-E plan predicted exactly this degenerate case
- **So the λ values above are marked provisional and are not used for the ablation**
- Every arm with fusion now has a **collapse guard** (Slide 6)

**S4.3ᶜ noise diagnostic ✅: verdict UNRESOLVED.** λ₁ = 0 (control) and λ₁ = 0.0333, 5 epochs,
noise augmentation on. Full validation split, k = 0, epoch 5:

| Arm | Fused F1 | Network-only F1 | Network-only "no fault" |
|---|---|---|---|
| λ₁ = 0 | 0.879 | 0.078 | 54.5 % |
| λ₁ = 0.0333 | 0.824 | 0.061 | 55.9 % |

- **Noise broke the hard collapse.** The network's "no fault" share fell from ≈ 100 % to ≈ 55 %, close to the true 50 %
- **But the network still did not localize.** Network-only F1 0.06–0.08, below the 0.10 bar. Localization was still left to physics
- λ₁ > 0 was worse than λ₁ = 0 on both measures (single seed, 5 epochs; an observation only)

**S4.3ᵈ auxiliary network-only loss ✅: verdict FIXED.** Added γ·CE on the network-only logits,
γ = 1. Null test first (γ = 0 identical to old code). Then λ₁ = λ₂ = 0, noise on, 5 epochs, seed 42.
Validation only; the test split was not loaded.

| Epoch | Fused F1 | Network-only F1 | Network-only F1 at σ = 1e-2 | Network "no fault" | β |
|---|---|---|---|---|---|
| 1 | 0.855 | 0.157 | 0.153 | 55.8 % | 0.027 |
| 3 | 0.868 | 0.386 | 0.353 | 54.1 % | 0.031 |
| **5** | **0.879** | **0.571** | **0.512** | **53.5 %** | **0.025** |
| S4.3ᶜ λ₁ = 0, epoch 5 | 0.879 | 0.078 | 0.064 | 54.5 % | 0.035 |

- **The network now localizes on its own:** network-only F1 is **7.3×** the S4.3ᶜ value and was still rising at epoch 5
- **Fused F1 did not change** (0.879), and β stays small (0.025–0.036)
- Network-only F1 is still below the standalone Phase-1 model at epoch 5 (0.641)
- Cost: 4.32 GPU-h. Single seed, 5 epochs: a diagnostic, not a tuning result
- **γ = 1 is now a fixed part of the design** and is not tuned

**S4.3ᵉ noise-augmented λ re-tuning — [Result, 2026-10-08] DONE: SELECTED λ₁ = 0, λ₂ = 0.1.**
*(Section below is the original 2026-10-05 pre-registration text, kept as the record of what was
committed before the run; see the result note at the end.)*
- Pre-registered and committed before the first run (`272ba01`)
- γ = 1, noise and dropout augmentation on, 15 epochs, seed 42, β learned, validation only
- Grid λ₁ ∈ {0, 0.01, 0.0333, 0.1}, λ₂ = 0. Then one λ₂ = 0.1 arm at the best λ₁ (and λ₁ = 0.3 first if 0.1 wins)
- Collapse guard applies; a run still collapsed at epoch 3 is stopped
- If no arm is valid the result is "no selection" and S4.4 does not start
- Estimated ≈ 65–80 GPU-h. **Actual: 72.67 GPU-h, 5 arms run (2026-10-05 07:32 UTC to 2026-10-08
  08:13 UTC). λ₁ hurt monotonically (0.8778 → 0.8180 as λ₁ rises from 0 to 0.1); λ₂ = 0.1 at
  λ₁ = 0 was best (0.8797 avg over k, +0.0019 over λ₁=λ₂=0), guard-valid, single seed. S4.4
  (Gate 4) started 2026-10-09 on the selected checkpoint, amended to 30 epochs 2026-10-12, and is
  running (3 of 6 runs complete as of 2026-10-15). [Correction, 2026-10-15]**

---

## SLIDE 9 — Datasets

| Dataset | Details | Use |
|---|---|---|
| **Fault corpus** (`dataset_fullscale_50`, from Phase 1) | IEEE 123-bus · 825 scenarios (25 buses × 11 fault types × 3 R) × 50 runs = **41,250 events** → **1.65 M windows**. Load factor uniform in 0.5–1.3. Exactly 50 % no-fault. **Event-level 70/15/15 split, 0 leakage** | train / val / test |
| **Physics signature table** (new) | 25 loc × 11 types × 18 R × 17 L × 25 nodes × 3 phases, plus no-fault μ₀. **84,167 OpenDSS solves**, 0 failures | the operator |
| **OOD-pilot** (new) | R ∈ {5, 20} Ω, never in training. ≈ 1,100 events, all 25 locations × 11 types | Gate-2 decision only, then discarded |
| **ρ cache** (new) | ρ for every window × k ∈ {0, 1, 2, 5, 10} × 3 mask seeds. Train 1,155,000 · val 248,000 · test 247,000 windows | fast evaluation and the k = 0 training lookup |
| **Scenario-held-out split** (built) | 575 / 125 / 125 scenarios, 0 overlap, all 26 classes per split | optional generalisation axis |

**Each window:**
- Shape `[25 nodes × 20 samples × 6]` at 1 ms (20 ms window)
- Fault inception at step 40; windows 0–19 are no-fault, 20–39 are fault
- Only **14 of 25** µPMUs are three-phase. The mask channels encode per-node phase instrumentation

---

## SLIDE 10 — Domain Expertise Course: Progress

> **⚠ TO FILL.** Not recorded in the repo. Add course name, platform, modules completed, % done,
> certificate status.

| Course | Platform | Progress | Relevance to project |
|---|---|---|---|
| _[fill]_ | _[fill]_ | _[fill] %_ | _[fill]_ |

---

## SLIDE 11 — Technical Enrichment Activities (Update)

> **⚠ TO FILL.** The three selected activities are not recorded in the repo.

| # | Activity | Status / Outcome |
|---|---|---|
| 1 | _[fill]_ | _[fill]_ |
| 2 | _[fill]_ | _[fill]_ |
| 3 | _[fill]_ | _[fill]_ |

---

## SLIDE 12 — Project Plan for Semester 3 (with completion to date)

```mermaid
flowchart LR
    A["A · Audit<br/>G0 ✅"] --> B["B · Operator<br/>G1 ✅ 5/5"] --> C["C · Go/No-go<br/>G2 ✅ GO"] --> D["D · Infrastructure<br/>G3 ✅ 5/5"] --> E["E · PC-STGNN<br/>G4 🔶 in progress"] --> F["F · Trust gate<br/>G5 ⏳"] --> G["G · Full eval +<br/>model mismatch ⏳"] --> H["H · Writing ⏳"]
```

| Stage (planned weeks) | Work | Status |
|---|---|---|
| A (wk 1) | Phase-1 audit, reproduction, census | ✅ 100 % |
| B (wk 2–4) | Signature table, fidelity → Gate 1 | ✅ 100 % |
| C (wk 4–6) | Identifiability, PSM, OOD contrast, fusion → Gate 2 · pre-registration | ✅ 100 % |
| D (wk 5–9) | Injectors, silent failures, physics imputation, residual layer → Gate 3 | ✅ 100 % |
| E (wk 8–13) | S4.1 null ✅ · S4.2 gradient audit ✅ · S4.3 λ tuning ✅ (6 runs, degenerate solution found) · S4.3ᶜ noise diagnostic ✅ (unresolved) · S4.3ᵈ auxiliary loss ✅ (fixed) · S4.3ᵉ noise-augmented re-tuning ✅ **done 2026-10-08** (λ₁=0, λ₂=0.1 selected) · S4.4 core ablation / **Gate 4 🔶 running** (started 2026-10-09, 3/6 runs complete as of 2026-10-15) | 🔶 in progress |
| F (wk 12–16) | PCRG-STGNN trust gate for silent failures | ⏳ planned |
| G (wk 15–19) | Sealed OOD test, noise, model-mismatch study | ⏳ planned |
| H (wk 19–26) | Analysis, figures, paper | ⏳ planned |

- **Overall:** 4 of 8 stages (and 4 of 6 gates) are closed
- **Compute:**
  - The budget was re-measured once real physics training ran. It is now ≈ 431–441 GPU-h, up from 255–270, because the live residual costs 44.7 ms per batch
  - S4.3 actually used ≈ 74 GPU-h (≈ 10.8 h per run plus ≈ 1.5 h of scoring each). A host reboot cost another ≈ 8.6 h
  - S4.3ᵈ used 4.32 GPU-h. S4.3ᵉ is estimated at ≈ 65–80 GPU-h (≈ 13 h per 15-epoch run with noise on, because the ρ cache cannot be used)
- **Next decision points:**
  1. ~~S4.3ᵉ selects λ₁ and λ₂ among arms that pass the collapse guard, or returns "no selection"~~
     **[Resolved 2026-10-08]** λ₁ = 0, λ₂ = 0.1 selected.
  2. Gate 4: the full model must beat the noise-augmented reference R0ⁿ on validation, with the
     same sign across 3 seeds. **[Correction, 2026-10-15]** This is no longer "next" — R0ⁿ and R5
     training is running now (started 2026-10-09, amended to 30 epochs 2026-10-12; 3 of 6 runs
     complete). The comparison itself has not yet been read.

---

## SLIDE 13 — Conclusion

**What Phase 2 has established so far (all measured):**
1. **A high-fidelity, fixed physics operator.** Built from 84,167 OpenDSS solves, it recovers fault type (90.9 %) and load (97.3 %) from 25 magnitude-only µPMUs
2. **The physics is informative and complementary, not redundant:**
   - The network wins in-distribution (0.855 vs 0.772)
   - The network collapses on unseen resistance (0.089), while physics holds (0.539)
   - Physics recovers accuracy under noise (up to +0.213 F1), which dropout-aware training could not do
3. **Physics-consistent imputation beats graph-mean imputation** at every dropout level (Ret@10 26.6 % vs 11.8 %)
4. **PC-STGNN is implemented and verified:**
   - Byte-identical to Phase 1 when switched off
   - Gradients audited
   - One extra parameter
5. **Tuning found a degenerate solution, which we identified and fixed.** Trained noise-free, the
   network branch learned to always say "no fault" and let physics do all the localization. We
   detected this by splitting each prediction into its network and physics parts. Noise
   augmentation alone removed the "no fault" bias but not the deferral (network-only F1 0.078).
   An auxiliary loss on the network-only logits (γ = 1) fixed it: network-only F1 **0.571** with
   the fused F1 unchanged. The λ re-tuning under these settings is running now

**Not claimed yet:** that trained PC-STGNN beats dropout-aware training. That is the Gate-4
question, answered by the S4.4 ablation (R0ⁿ, R2, R5, R6 at 3 seeds). It runs only after the
S4.3ᵉ re-tuning gives a valid selection.

---

## SLIDE 14 — References

1. B. Karabulut, C. Manna, C. Develder, "Robustness of spatio-temporal graph neural networks for fault location in partially observable distribution grids," *arXiv:2604.20403*, 2026.
2. B. Karabulut, C. Manna, C. Develder, "Generalization of graph neural network models for distribution grid fault detection," *arXiv:2510.03571*, 2025.
3. W. Hamilton, Z. Ying, J. Leskovec, "Inductive representation learning on large graphs," *NeurIPS* 30, 2017.
4. S. Brody, U. Alon, E. Yahav, "How attentive are graph attention networks?" *ICLR*, 2022.
5. K. Chen, J. Hu, Y. Zhang, Z. Yu, J. He, "Fault location in power distribution systems via deep graph convolutional networks," *IEEE JSAC* 38(1), 2020.
6. L. Zhang, D. Hua, T. Ji, T. Qian, J. Wang, "Fault location in power distribution networks with massive missing data: a graph-based imputation and contrastive learning approach," *IEEE T-ASE* 22, 2025.
7. E. Rossi et al., "On the unreasonable effectiveness of feature propagation in learning on graphs with missing node features," *LoG, PMLR* 198, 2022.
8. R. Krishnathevar, E. E. Ngu, "Generalized impedance-based fault location for distribution systems," *IEEE Trans. Power Delivery* 27(1), 2012.
9. Q.-H. Ngo et al., "Deep graph neural network for fault detection and identification in distribution systems," *EPSR* 247, 2025.
10. M. Fan, J. Xia, H. Zhang, X. Zhang, "Fault location method of distribution network based on VGAE-GraphSAGE," *Processes* 12(10), 2024.
11. Y. Rong, W. Huang, T. Xu, J. Huang, "DropEdge: Towards deep graph convolutional networks on node classification," *ICLR*, 2020.
12. IEEE Std C37.118.1 — Synchrophasor measurements for power systems (1 % TVE accuracy limit).

---

## Appendix — Likely panel questions (answers are all measured)

**Q: Isn't PSM just a lookup of the simulator that made the data?**
Yes, on clean data. That is why clean k = 0 is reported only as a fidelity check, never as a
"physics wins" result. Even so, the network beats PSM there (0.855 vs 0.772). The decision
was read off noise and unseen resistance only.

**Q: How do you know the gain isn't extra capacity?**
Physics adds exactly one parameter, β. A shuffled-physics arm (R6) and a size-matched wider
backbone (R7) are part of the S4.4 plan. With physics off, the model is byte-identical to
Phase 1.

**Q: Could the physics leak the label?**
No. ρ is a function of the input tensor only. The w = 20 windows have no post-fault sample, and
every Phase-1 model scores at chance on them (0.0003–0.0418 against a chance rate of 0.0385). That
is a clean no-leakage check.

**Q: Why is training slower than Phase 1?**
The live residual is a minimum over 3,366 hypotheses × 26 classes for every batch: 44.7 ms,
compute-bound. In noise-free training, batches with no dropped sensors read a verified cache.
That cache was built noise-free, so it is switched off for noise-augmented runs.

**Q: Your tuned model scores 0.85. Doesn't that mean PC-STGNN works?**
Not yet. In the noise-free tuning, with the physics term removed, the network branch voted
"no fault" on every window. The 0.85 was physics-only PSM plus a "no fault" bias, so it wasn't
evidence that the fusion learned anything. That's why those tuned values are held back and why
every fusion arm now carries a collapse guard.

**Q: Why train with noise?**
On noise-free data the physics residual at the true fault is exactly zero, so trusting physics
completely is the easiest solution and the network contributes nothing. Real µPMUs are noisy:
IEEE C37.118 allows up to 1 % error. Noise gives the physics term a realistic, imperfect
confidence, which leaves the network something to learn. The diagnostic showed noise was
necessary but not sufficient: it removed the "no fault" bias, but the network still left
localization to physics (network-only F1 0.078) until the auxiliary loss was added.

**Q: Doesn't the auxiliary loss confound the comparison?**
It uses the same labels and the same unweighted cross-entropy as the main loss and adds no
parameters. γ is fixed at 1, not tuned. Every PC-STGNN arm in S4.4 uses γ = 1, and the paired
reference R0ⁿ is the same backbone trained with plain cross-entropy on its own logits, which is
the loss the auxiliary term applies to the network branch.

**Q: Did any test-split result influence your choices?**
No. The runner wrote test files for 8 validation-only runs by default. We found this, moved all
24 files to a quarantine folder (reading key names only, never values), and audited code,
logs and session transcripts: no selection or verdict read a test field. An automated process
(probably the editor's indexer) touched some of the small files; that cannot be fully
attributed, so we do not claim more than this. Tuning runs now never load the test
split.
