# Related Work

<!--
DRAFTING NOTES (delete before submission):
- Citations here are now VERIFIED against the base paper's own bibliography (BasePaper.pdf
  re-downloaded and readable; ref numbers in the comment after each citation, e.g. [BASE ref 7],
  map to Karabulut et al.'s reference list so you can cross-check quickly). Full entries are in
  §References at the bottom of this file — fold them into your global .bib.
- [BASE] = Karabulut, Manna, Develder 2026 (the base paper), consistent with 01/02.
- This section is deliberately built around ONE axis — robustness to sensors that FAIL at
  inference time — because that, not "GNNs for fault location", is this paper's gap. Table 1's
  two rightmost columns are empty for the entire prior literature (including [BASE]); that empty
  column IS the contribution. Keep that framing even if you trim prose.
- Page budget: for a 4-6 page venue this section should land ~0.5-0.75 page. If tight, keep the
  table + §2.4 and compress §2.1-2.3 to one paragraph each.
- Table 2 (one-glance taxonomy) is commented out at the end — ship EITHER Table 1 OR Table 2.
- The prose here intentionally parallels [BASE]'s own Related Work structure (model-based ->
  non-graph DL -> spectral/spatial GNN -> temporal GNN -> GAT -> GraphSAGE) so a reviewer who
  knows the base paper sees continuity, then lands on the new robustness axis we add.
-->

## 2. Related Work

Work on fault location in distribution grids divides into model-based methods, non-graph
data-driven methods, and graph neural networks (GNNs). We review each in turn, following the
taxonomy of prior surveys [Chen et al. 2016; Rezapour et al. 2023], and then identify the gap
this paper targets: none of this literature characterizes — or defends against — μPMUs that
drop out *after* a model has been trained and deployed.

### 2.1 Model-based methods

Classical fault-location techniques are physics-based: impedance-based estimators [Chandran
et al. 2024; Krishnathevar & Ngu 2012], voltage-sag analysis [Buzo et al. 2021; Turizo et
al. 2022], smart-meter/digital-twin outage mapping [Numair et al. 2023], and traveling-wave
methods [Liu et al. 2023; Thomas et al. 2003]. These are effective in traditional radial
settings but rely on extensive measurement infrastructure for sufficient observability and on
static assumptions about feeder topology and operating conditions [Ojha et al. 2018].
Traveling-wave methods are further burdened by the capital cost of high-speed sensors, while
impedance- and sag-based methods are sensitive to the topological assumptions that modern,
reconfigurable feeders routinely violate.

### 2.2 Non-graph data-driven methods

Data-driven methods relax these modeling assumptions. Classical machine learning — kNN,
random forests, decision trees, and SVMs on hand-crafted features — has been applied to
faulty-line identification and localization [Livani et al. 2013; Hosseini et al. 2018].
Deep learning then removed the feature-engineering step: CNNs capture spatial relationships
and RNNs (GRU, LSTM) exploit temporal patterns in the measurement stream [Turizo et al. 2022].
These methods are flexible but *topology-agnostic* — a flat CNN or RNN cannot represent the
non-uniform electrical connectivity between buses [Chen et al. 2020], which limits localization
accuracy.

### 2.3 Graph neural networks for fault location

GNNs [Gori et al. 2005; Scarselli et al. 2009] encode the feeder directly — buses as nodes,
lines as edges — and aggregate neighbor information via message passing [Liao et al. 2022],
which is a natural fit for a task that depends on network structure. Within fault location,
**spectral GCNs** convolve in the graph-Fourier domain [Chen et al. 2020] but are tied to a
fixed global basis and adapt poorly to reconfiguration; **spatial GCNs** operate directly on
local node/edge features [Kipf & Welling 2017; Ma et al. 2024; MansourLakouraj et al. 2021]
and are better suited to the locally-structured fault-location task; and **gated graph
networks** iteratively refine node states [de Freitas & Coelho 2021]. To add the temporal
dimension that pure GNNs miss, **spatio-temporal** hybrids combine a temporal encoder with a
GNN: a 1D-CNN + GCN, later extended with LSTM [Nguyen et al. 2022; Nguyen et al. 2023], and
patch-based temporal GCNs [Wang et al. 2026]. On the architecture side, **graph-attention
networks** [Veličković et al. 2018] learn node-specific neighbor weights, reducing reliance on
a fixed graph — used for fault location with cost-sensitive weighting [Wang et al. 2024] and
with a 1D-CNN temporal front end [Ngo et al. 2025] — while **GraphSAGE** learns from sampled
local neighborhoods for scalability, e.g. GSAGE-GAT [Wei et al. 2024] and a VGAE-GraphSAGE
model targeting measurement noise [Fan et al. 2024].

A smaller line of work confronts sparse or missing measurements directly: Zhang et al. [2025]
(GDIA-GCL) impute missing bus measurements and train a GCN with graph contrastive learning for
localization under extreme data scarcity. Critically, however, this treats missing sensors as a
*static* condition fixed before training, not as sensors that fail at inference time.

### 2.4 Positioning of this work

The base paper [BASE] is the most direct precedent: it introduces a *measured-only* graph
construction that reflects sparse μPMU placement — rather than the usual 1-to-1 map between the
GNN graph and the full physical feeder — and is the first to systematically benchmark multiple
STGNN architectures (RGCN, GraphSAGE mean/max, and GATv2) against a GRU-only baseline on the
IEEE 123-bus feeder, including a reconfigured ("green") topology, reporting soft-voted graph
macro-F1 in the 0.86–0.95 range for its best configurations. As [BASE] itself notes, prior GNN
work assumes near-complete 3-phase visibility, keeps a 1-to-1 graph–feeder mapping, and lacks a
rigorous cross-architecture comparison — gaps it closes. But [BASE], too, assumes the sparse
sensor set is *fully functional at inference*.

To our knowledge, no prior work characterizes how STGNN fault-localization accuracy degrades
when μPMUs drop out at inference time, nor adapts training to anticipate such loss. Table 1
places the methods above along the axes that matter for this gap — topology awareness, sparse
observability, and the two columns that separate this paper from all prior work: whether a
method is evaluated under sensors that *fail at test time*, and whether it does anything to
*recover* the resulting loss. Our contribution fills those two columns: we quantify STGNN
degradation under inference-time μPMU dropout ($k \in \{0,1,2,5,10\}$ masked sensors) and
introduce dropout-aware training (random node masking) as a low-cost, architecture-agnostic
mitigation.

**Table 1. Positioning of representative fault-localization methods.**
Task: D = detection, C = type classification, L = localization.
✓ = yes · ✗ = not studied · ~ = partial. The two rightmost columns — the focus of this paper —
are empty across the prior literature, including the base paper.

| Work | Approach / model | Task | Test system | Multi-arch | Partial obs. | Noise robust | Drop-out @ test | Loss mitigation |
|------|------------------|:----:|-------------|:----------:|:------------:|:------------:|:---------------:|:---------------:|
| Livani et al. (2013) | ML (SVM / decision trees) + features | L | Distribution | ✗ | ✗ | ✗ | ✗ | ✗ |
| Turizo et al. (2022) | Deep CNN (voltage sags) | L+C | Distribution | ✗ | ✗ | ~ | ✗ | ✗ |
| Chen et al. (2020) | Spectral GCN (DGCN) | L | IEEE 123-bus | ✗ | ✗ | ~ (data aug.) | ✗ | ✗ |
| Ma et al. (2024) | Spatial GCN | L | Distribution (high DG) | ✗ | ✗ | ✗ | ✗ | ✗ |
| de Freitas & Coelho (2021) | Gated graph NN (GGNN) | L | Distribution | ✗ | ✗ | ✗ | ✗ | ✗ |
| Nguyen et al. (2023) | Spatio-temporal: 1D-CNN + GCN + LSTM | D+C+L | Distribution | ✗ | ✗ | ✓ | ✗ | ✗ |
| Wang et al. (2024) | Cost-sensitive GAT | L | Distribution | ✗ | ✗ | ✗ | ✗ | ✗ |
| Ngo et al. (2025) | GAT + 1D-CNN | D+C+L | Distribution | ✗ | ✗ (dense 3-φ) | ✓ | ✗ | ✗ |
| Fan et al. (2024) | GraphSAGE + VGAE | L | Distribution | ✗ | ✗ | ✓ (noise) | ✗ | ✗ |
| Wei et al. (2024) | GSAGE-GAT | L | Distribution | ✗ | ✗ | ~ | ✗ | ✗ |
| Zhang et al. (2025) | GDIA-GCL (imputation + contrastive) | L | Distribution, massive missing data | ✗ | ✓ | ✓ | ~ (static missing) | imputation only |
| [BASE] Karabulut et al. (2026) | STGNN: RGCN, RGSAGE-mean/max, RGATv2 (+GRU) | L | IEEE 123-bus (default + green) | ✓ | ✓ (measured-only graph) | ~ | ✗ | ✗ |
| **This work** | **Same STGNN family + inference-time μPMU drop-out study + dropout-aware training** | **L** | **IEEE 123-bus (default + green)** | **✓** | **✓** | **✓** | **✓ ($k\in\{0,1,2,5,10\}$)** | **✓ dropout-aware training** |

## References cited in this section

<!-- Verified against BasePaper.pdf's bibliography; bracketed [BASE ref N] = number in
Karabulut et al.'s reference list. Reformat to your venue's citation style. -->

- **[BASE]** B. Karabulut, C. Manna, C. Develder. *Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids.* arXiv:2604.20403, 2026.
- R. F. Buzo, H. M. Barradas, F. B. Leão. *A New Method for Fault Location in Distribution Networks Based on Voltage Sag Measurements.* IEEE Trans. Power Del. 36(2):651–662, 2021. [BASE ref 3]
- S. Chandran, R. Gokaraju, K. Narendra. *An extended impedance-based fault location algorithm … using synchrophasors.* IET Gener. Transm. Distrib. 18(3):479–490, 2024. [BASE ref 6]
- K. Chen, J. Hu, Y. Zhang, Z. Yu, J. He. *Fault Location in Power Distribution Systems via Deep Graph Convolutional Networks.* IEEE J. Sel. Areas Commun. 38(1):119, 2020. [BASE ref 7]
- K. Chen, C. Huang, J. He. *Fault detection, classification and location … a review on the methods.* High Volt. 1(1):25–33, 2016. [BASE ref 8]
- J. T. de Freitas, F. G. F. Coelho. *Fault localization method for power distribution systems based on gated graph neural networks.* Electr. Eng. 103:2259–2266, 2021. [BASE ref 9]
- M. Fan, J. Xia, H. Zhang, X. Zhang. *Fault Location Method of Distribution Network Based on VGAE-GraphSAGE.* Processes 12(10):2179, 2024. [BASE ref 11]
- M. Gori, G. Monfardini, F. Scarselli. *A New Model for Learning in Graph Domains.* IEEE Trans. Neural Netw., 2005. [BASE ref 13]
- W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. [BASE ref 16]
- Z. S. Hosseini, M. Mahoor, A. Khodaei. *AMI-Enabled Distribution Network Line Outage Identification via Multi-Label SVM.* IEEE Trans. Smart Grid 9(5):5470–5472, 2018. [BASE ref 19]
- T. N. Kipf, M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017. [BASE ref 23]
- R. Krishnathevar, E. E. Ngu. *Generalized Impedance-Based Fault Location for Distribution Systems.* IEEE Trans. Power Del. 27(1):449–451, 2012. [BASE ref 24]
- W. Liao, B. Bak-Jensen, J. R. Pillai, Y. Wang. *A review of graph neural networks and their applications in power systems.* J. Mod. Power Syst. Clean Energy 10(2):345–360, 2022. [BASE ref 25]
- F. Liu, L. Xie, K. Yu, et al. *A novel fault location method based on traveling wave for multi-branch distribution network.* Electr. Power Syst. Res. 224:109753, 2023. [BASE ref 26]
- H. Livani, C. Y. Evrenosoğlu, V. A. Centeno. *A machine learning based faulty line identification for smart distribution network.* NAPS 2013. [BASE ref 28]
- X. Ma, W. Zhen, H. Ren, et al. *A Method for Fault Localization in Distribution Networks with High Proportions of Distributed Generation Based on Graph Convolutional Networks.* Energies 17(22):5758, 2024. [BASE ref 30]
- M. MansourLakouraj, R. Hossain, H. Livani, M. Ben-Idris. *Application of graph neural network for fault location in PV penetrated distribution grids.* NAPS 2021. [BASE ref 31]
- Q.-H. Ngo, B. L. H. Nguyen, J. Zhang, et al. *Deep graph neural network for fault detection and identification in distribution systems.* Electr. Power Syst. Res. 247:111721, 2025. [BASE ref 34]
- B. L. H. Nguyen, T. Vu, T.-T. Nguyen, et al. *1-D Convolutional Graph Convolutional Networks for Fault Detection in Distributed Energy Systems.* ONCON 2022. [BASE ref 35]
- B. L. H. Nguyen, T. V. Vu, T.-T. Nguyen, et al. *Spatial-Temporal Recurrent Graph Neural Networks for Fault Diagnostics in Power Distribution Systems.* IEEE Access 11, 2023. [BASE ref 36]
- M. Numair, A. A. Aboushady, F. Arraño-Vargas. *Fault detection and localisation in LV distribution networks using a smart meter data-driven digital twin.* Energies 16(23):7850, 2023. [BASE ref 37]
- A. Ojha, V. Panwar, P. Kant. *Optimization techniques for network reconfiguration in distribution systems: An overview.* Natl. J. Multidiscip. Res. 3(3):68–71, 2018. [BASE ref 38]
- H. Rezapour, S. Jamali, A. Bahmanyar. *Review on Artificial Intelligence-Based Fault Location Methods in Power Distribution Networks.* Energies 16(12):4636, 2023. [BASE ref 40]
- F. Scarselli, M. Gori, A. Tsoi, et al. *The Graph Neural Network Model.* IEEE Trans. Neural Netw. 20(1):61–80, 2009. [BASE ref 42]
- D. W. P. Thomas, R. J. O. Carvalho, E. T. Pereira. *Fault location in distribution systems based on traveling waves.* IEEE Bologna PowerTech, 2003. [BASE ref 44]
- S. Turizo, G. Ramos, D. Celeita. *Voltage Sags Characterization Using Fault Analysis and Deep Convolutional Neural Networks.* IEEE Trans. Ind. Appl. 58(3):3367–3376, 2022. [BASE ref 45]
- P. Veličković, G. Cucurull, A. Casanova, et al. *Graph Attention Networks.* ICLR 2018. [BASE ref 46]
- S. Wang, Q. Fang, N. Li, M. Fan, Y. Duan. *Fault Location Method for Distribution Network Based on Temporal Graph Convolutional Network.* CIAC 2025 (LNEE 1501), 2026. [BASE ref 47]
- Z. Wang, B. Huang, B. Zhou, J. Chen, Y. Wang. *An Enhanced Fault Localization Technique for Distribution Networks Utilizing Cost-Sensitive Graph Neural Networks.* Processes 12(11):2312, 2024. [BASE ref 48]
- T. Wei, H. Zhang, M. Fan, J. Sun, X. Wang. *Fault Location Method of Distribution Network Based on Improved Graph Neural Network.* ICPRE 2024. [BASE ref 49]
- L. Zhang, D. Hua, T. Ji, T. Qian, J. Wang. *Fault Location in Power Distribution Networks With Massive Missing Data: A Graph-Based Imputation and Contrastive Learning Approach.* IEEE Trans. Autom. Sci. Eng. 22:16825–16837, 2025. [BASE ref 51]

<!--
OPTIONAL — one-glance taxonomy for the Introduction or a very tight page budget.
Ship EITHER Table 1 (above) OR Table 2, not both.

**Table 2. Fault-localization eras and what each handles.**

| Era | Representative methods | Topology-aware | Sparse sensors | Sensors that fail at runtime |
|-----|------------------------|:--------------:|:--------------:|:----------------------------:|
| Model-based | Impedance, voltage-sag, traveling-wave | ✗ | ✗ | ✗ |
| Non-graph DL | kNN/SVM/RF, CNN, RNN/LSTM | ✗ | ✗ | ✗ |
| GNN / STGNN | GCN, GAT/GATv2, GraphSAGE, RGCN ([BASE]) | ✓ | some | ✗ |
| This work | STGNN + dropout-aware training | ✓ | ✓ | ✓ |
-->
