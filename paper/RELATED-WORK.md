# Related Work

> **How this positions your contribution.** Your novelty is *not* "GNNs for fault
> localization" (the base paper already does that) and *not even* "partial observability"
> (the base paper's measured-only graph handles the *static* sparse-sensor case). Your
> differentiator is **robustness to sensor DROP-OUT at inference time** (μPMUs that fail /
> lose comms after deployment) **and a mitigation for it — dropout-aware training (random
> node masking)**. So the related-work table is built around that axis: for every prior
> method we ask *does it study sensors that disappear at test time, and does it do anything
> to recover the lost accuracy?* The answer is "no" for essentially the entire literature —
> that empty column is your paper's gap.

## 1. Reading the landscape

Fault localization in distribution grids splits into three eras:

- **Model-based / physics methods** — impedance-based, voltage-sag, and traveling-wave
  techniques. Accurate but assume dense, always-on measurement infrastructure and a
  static, fully known topology.
- **Classical & non-graph deep learning** — kNN/SVM/random forests on hand-crafted
  features, then CNNs and RNNs/LSTMs. More flexible, but *topology-agnostic*: they cannot
  represent the non-uniform electrical connectivity of the feeder.
- **Graph neural networks (GNNs / STGNNs)** — inject topology via message passing;
  spatio-temporal variants (RNN + GNN) add the temporal dimension. This is the base
  paper's family. Almost all of this work assumes **near-complete, always-available
  3-phase observability**.

Your work extends the third era along a new axis: **operational robustness when the sparse
sensor set is itself unreliable.**

## 2. Main positioning table

Columns are chosen to spotlight the gap you fill. "Partial obs." = designed for a *sparse*
(not full) sensor set. "Sensor drop-out @ test" = evaluates accuracy when sensors that were
present at training **disappear at inference**. "Mitigation" = does anything to *recover*
accuracy under that loss.

| # | Work (year) | Approach / model | Task¹ | Test system / data | Input | Multi-arch? | Partial obs. | Noise robustness | **Sensor drop-out @ test** | **Mitigation for sensor loss** |
|---|-------------|------------------|-------|--------------------|-------|-------------|--------------|------------------|----------------------------|-------------------------------|
| 1 | Chen et al. (2016) | kNN / RF / SVM + hand-crafted features | L | Distribution feeder | Feature vectors | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2 | Alzubaidi et al. (2021) | RNN / LSTM | D+C | Distribution | Time series | ✗ | ✗ | partial | ✗ | ✗ |
| 3 | Nguyen et al. (2023) | CNN (+ hybrid CNN–GCN) | L+C | IEEE 123-bus | 3-phase signals | ✗ | ✗ | ✓ | ✗ | ✗ |
| 4 | Chen et al. (2020) / DGCN (arXiv 1812.09464) | Spectral Graph Conv. Net | L | IEEE 123-bus | Dense 3-phase V | ✗ | ✗ | partial | ✗ | ✗ |
| 5 | Kipf & Welling (2017); Ma et al. (2024) | Spatial GCN | L | IEEE 123-bus | Dense bus meas. | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 | de Freitas & Coelho (2021) | Gated Graph NN (GGNN) | D+L | Distribution nets | Node features | ✗ | ✗ | ✗ | ✗ | ✗ |
| 7 | Wang et al. (2024); Ngo et al. (2025) | Graph Attention (GAT) | L+C | IEEE 123-bus | Dense 3-phase | ✗ | ✗ | ✓ (some) | ✗ | ✗ |
| 8 | Fan et al. (2024) | GraphSAGE + VGAE | L | Distribution nets | Node features | ✗ | ✗ | ✗ | ✗ | ✗ |
| 9 | Zhang et al. (2025) — GDIA-GCL | Graph contrastive learning + imputation | L | Sparse-sensor grid | Partial meas. | ✗ | ✓ (missing data) | ✓ | ~ (static missing) | imputation (not retraining) |
| 10 | Physics-Informed Graph Learning (2023) | Physics-informed GNN, 2-stage | L | Distribution | μPMU | ✗ | ✓ | ✓ | ✗ | ✗ |
| 11 | Rimkus et al. (2025) | GNN, partially-observed system | L | Regional transmission | Sparse meas. | ✗ | ✓ | ✗ | ✗ | ✗ |
| 12 | Deep GAT (ScienceDirect, 2025) | Graph Attention Net | D+C+L | Distribution | μPMU | ✗ | ✗ | ✓ (3–10% noise) | ✗ | ✗ |
| **B** | **Karabulut et al. (2026) — base paper** | **STGNN: RGCN, RGSAGE-mean/max, RGATv2 + GRU** | **L** | **IEEE 123-bus (OpenDSS), default + green** | **3-phase RMS V, 25 μPMU** | **✓** | **✓ (measured-only graph)** | **partial** | **✗** | **✗** |
| **★** | **This work (2026)** | **Same STGNN family + inference-time μPMU drop-out study + dropout-aware training** | **L** | **IEEE 123-bus (OpenDSS), default + green** | **3-phase RMS V, 25 μPMU** | **✓** | **✓** | **✓** | **✓ (k ∈ {0,1,2,5,10} masked μPMUs, degradation curves)** | **✓ dropout-aware training (random node masking)** |

¹ Task: **D** = detection, **C** = type classification, **L** = localization.
✓ = yes · ✗ = no / not studied · ~ = partially · "partial" = touched but not systematic.

**The story the table tells:** the two right-hand columns are empty for the entire prior
literature — including the base paper. Method #9 (GDIA-GCL) is the closest neighbor, but it
handles *statically* missing sensors via imputation, not sensors that fail *after* a model
is trained and deployed, and it does not retrain the model to be robust. **This work is the
first to (a) quantify accuracy degradation of STGNNs under inference-time μPMU loss and
(b) show dropout-aware training recovers much of it, especially in the harder green feeder.**

## 3. Compact taxonomy table (optional, for the intro)

If you want a shorter one-glance table for the introduction:

| Era | Representative methods | Topology-aware | Handles sparse sensors | Handles sensors that *fail at runtime* |
|-----|------------------------|:--------------:|:----------------------:|:--------------------------------------:|
| Model-based | Impedance, voltage-sag, traveling-wave | ✗ | ✗ | ✗ |
| Non-graph DL | kNN/SVM/RF, CNN, RNN/LSTM | ✗ | ✗ | ✗ |
| GNN / STGNN | GCN, GAT/GATv2, GraphSAGE, RGCN | ✓ | some (base paper) | ✗ |
| **This work** | STGNN + dropout-aware training | ✓ | ✓ | **✓** |

## 4. One-paragraph prose version (for the paper body)

> Early fault-localization methods are model-based — impedance, voltage-sag, and
> traveling-wave techniques [Chandran 2024; Turizo 2022; Liu 2023] — which are accurate but
> assume dense, always-available measurements and a static, fully known topology. Data-driven
> methods relax these assumptions: classical ML on hand-crafted features [Chen 2016] and,
> later, CNNs and RNNs [Nguyen 2023; Alzubaidi 2021] add flexibility but remain
> topology-agnostic. Graph neural networks close this gap by encoding feeder connectivity via
> message passing; spatio-temporal variants combine RNNs with GCN, GAT, or GraphSAGE layers
> [Chen 2020; Kipf & Welling 2017; Wang 2024; Ngo 2025; Fan 2024]. Most of this work, however,
> assumes near-complete 3-phase observability. Karabulut et al. [2026] take an important step
> toward realism with a *measured-only* graph construction that reflects sparse μPMU placement
> and benchmark multiple STGNN architectures on the IEEE 123-bus feeder. Yet even they assume
> the sparse sensor set is *fully functional at inference*. In practice μPMUs fail, lose
> communication, or are decommissioned. Our work is the first to quantify how STGNN
> localization accuracy degrades as μPMUs drop out at test time, and to introduce
> *dropout-aware training* — random node masking during training — as a low-cost method that
> recovers a substantial share of the lost accuracy, most notably in the challenging
> reconfigured ("green") feeder scenario.

## 5. References (verify before submission)

> ⚠️ Author/year/venue below are reconstructed from the arXiv HTML version and web searches
> because the local `BasePaper.pdf` is corrupted and its reference list could not be read.
> **Confirm every citation against the real paper's bibliography.**

- **[Base]** B. Karabulut, C. Manna, C. Develder, *Robustness of Spatio-temporal Graph Neural
  Networks for Fault Location in Partially Observable Distribution Grids*, arXiv:2604.20403, 2026.
- Chen et al., ML with hand-crafted features for fault location, 2016.
- Alzubaidi et al., RNN/LSTM fault detection, 2021.
- Nguyen et al., CNN / hybrid CNN–GCN fault location on IEEE 123-bus, 2022/2023.
- Chen et al., *Fault Location in Power Distribution Systems via Deep Graph Convolutional
  Networks*, arXiv:1812.09464, 2018/2020.
- T. N. Kipf, M. Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017.
- Ma et al., spatial GCN for fault location, 2024.
- de Freitas & Coelho, Gated Graph NN for fault detection/location, 2021.
- Wang et al. (2024); Ngo et al. (2025), GAT-based fault location, IEEE 123-bus.
- Fan et al., GraphSAGE + VGAE, 2024.
- Zhang et al., *GDIA-GCL* graph contrastive learning with imputation for sparse
  infrastructure, 2025.
- *Physics-Informed Graph Learning for Robust Fault Location in Distribution Systems*, 2023.
- A. Rimkus et al., *Graph neural networks for the localization of faults in a partially
  observed regional transmission system*, Scand. J. Statistics, 2025.
- *Deep graph neural network / GAT for fault detection and identification in distribution
  systems*, ScienceDirect, 2025.
