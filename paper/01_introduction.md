# Introduction

<!--
DRAFTING NOTES (delete before submission):
- Venue not yet chosen (APPROACH.md lists ISGT / PowerTech / SmartGridComm / IEEE PES GM as
  candidates) — this draft is venue-agnostic prose, not formatted to any template yet. Page
  budget for these venues is typically 4-6 pages two-column, so the final Introduction should
  land around 0.75-1 page (~500-700 words); this draft is a slightly fuller first pass to cut
  down from, not a final-length target.
- Citations beyond the base paper are marked [CITE: ...] — I have verified bibliographic
  details only for Karabulut et al.; I'm not inventing other references. Fill these in with
  your own literature search.
- The GNN-depth/receptive-field finding (Section on "what we found along the way") is from an
  ongoing diagnostic investigation, NOT the paper's stated novelty (that's the sensor-dropout
  study, per PROBLEM.md). I've included it as an optional short paragraph since it's a genuine,
  interesting finding, but flagged it clearly — cut it if it doesn't end up supporting the
  final story, or move it to Discussion/Limitations instead of Introduction.
- Exact final reproduction numbers (macro-F1 achieved vs. paper's 0.86-0.95) are placeholders
  ([X]) pending the still-running investigation — fill in once settled.
-->

Reliable fault localization is a prerequisite for automated restoration in modern power
distribution grids. As feeders absorb growing shares of distributed energy resources (DERs)
and adopt self-healing reconfiguration schemes, the speed and precision with which an outage
management system can identify *where* a fault occurred — not merely *that* one occurred —
directly determines how much of the network can be re-energized before crews are dispatched.
Traditional fault-location methods rely on dense, synchronized measurement infrastructure
[CITE: classical impedance-based / traveling-wave fault location surveys], an assumption that
rarely holds in distribution networks, where micro-phasor measurement units (μPMUs) are
deployed sparsely due to cost and communication constraints. A distribution feeder with
hundreds of buses may carry only a handful of μPMUs, so any practical fault-localization method
must operate under **partial observability**: it must infer the fault location for buses it
cannot directly instrument, using only the temporal signatures visible at a sparse subset of
monitored nodes.

Graph neural networks (GNNs) are a natural fit for this setting, since the feeder's known
topology can be encoded directly as the message-passing graph, letting information from a fault
propagate to nearby sensors even when the fault bus itself is unmeasured. Karabulut et
al. (2026) [CITE: Karabulut, Manna, Develder — Robustness of Spatio-temporal Graph Neural
Networks for Fault Location in Partially Observable Distribution Grids] establish this direction
concretely: they combine a recurrent (GRU) temporal encoder with a spatial GNN layer — evaluating
RGCN, GraphSAGE (mean/max), and GATv2 variants against a GRU-only baseline — on the IEEE 123-bus
feeder with 25 sparsely placed μPMUs, 25 candidate fault locations, 11 fault types, and three
fault resistances, and report soft-voted graph-level macro-F1 in the 0.86–0.95 range for their
best spatio-temporal GNN (STGNN) configurations, substantially above the GRU-only baseline.

This result, however, rests on an assumption that does not hold in deployed systems: that every
μPMU is always available and functioning. Real sensors drop out — through communication loss,
maintenance, hardware failure, or deliberate decommissioning — and a fault-localization model
that was only ever trained and evaluated with a complete sensor set has no guarantee of
degrading gracefully, or at all, once some of its inputs go dark. To our knowledge, the
robustness of STGNN-based fault localization to *sensor dropout at inference time* has not been
characterized, nor has training been adapted to anticipate it.

**This paper makes the following contributions.** First, we reproduce the Karabulut et al. STGNN
benchmark protocol — the IEEE 123-bus feeder, μPMU placement, fault taxonomy, and windowing
scheme — training GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, and RGATv2 on identical data, splits, and
loss under a class-imbalance-corrected (inverse-frequency weighted) objective, and report our own
reproduced baseline numbers as the reference point for everything that follows [X]. Second, and
centrally, we quantify how far fault-localization accuracy degrades when $k \in \{0, 1, 2, 5,
10\}$ μPMUs are dropped at inference time, across all five model architectures, producing
macro-F1-vs-$k$ degradation curves. Third, we introduce **dropout-aware training** — randomly
masking a variable number of measurement nodes during training, rather than always presenting the
full sensor set — as a simple, architecture-agnostic mitigation, and show that it recovers a
substantial fraction of the accuracy lost to sensor dropout relative to standard training,
including under a reconfigured ("green") feeder topology that stresses the model's reliance on a
fixed graph structure. All comparisons — baselines against each other, and standard training
against dropout-aware training — are conducted on identical data, splits, and evaluation
protocol, so that reported effects are internally consistent even where they depart from the
original paper's absolute scale (see Reproduction Notes, Section [X]).

<!--
OPTIONAL — only keep if the depth/receptive-field finding survives the ongoing investigation
and you decide to foreground it (otherwise move to Discussion/Limitations, or drop):

Along the way, we found that the GNN's spatial receptive field — the number of stacked
message-passing hops — has an outsized effect on localization accuracy under partial
observability: because many candidate fault buses sit more than one hop from the nearest μPMU
in the measured-only graph, a single spatial layer can only resolve faults within its immediate
neighborhood. In our reduced-scale reproduction, increasing depth from 1 to 4 layers raised
soft-voted graph macro-F1 from [X] to [X] on RGSAGE-Max, a considerably larger effect than
additional training epochs alone. We report this as [a reproduction finding / a design choice
informing our dropout-aware architecture], since sensor dropout — the focus of this paper —
only compounds the reach problem multi-hop depth addresses. [Note: verify this framing doesn't
overstate an initial finding — cross-check against the paper's own stated architecture before
claiming a discrepancy.]
-->

The remainder of this paper is organized as follows. Section [X] reviews related work in
distribution-grid fault location and robustness to missing/corrupted sensor data. Section [X]
describes the simulation setup, feeder model, and dataset. Section [X] details the model
architectures and training protocol, including our reproduction notes and dropout-aware training
scheme. Section [X] presents results: reproduced baselines, dropout degradation curves, and the
effect of dropout-aware training on both the default and green feeder topologies. Section [X]
discusses limitations and concludes.
