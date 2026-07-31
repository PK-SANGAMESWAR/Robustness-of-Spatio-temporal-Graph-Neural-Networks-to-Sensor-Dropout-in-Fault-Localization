# Experimental Setup and Methods

<!--
DRAFTING NOTES (delete before submission):
- Numbers/parameters below are pulled directly from the codebase
  (fault_localization/config/constants.py, config/*.yaml) and verified against
  docs/reproduction_log.md — not guessed. [CITE] markers point back to [BASE]
  §-numbers already used consistently in this codebase's own reproduction log.
- Reproduction Notes subsection is the honest-disclosure paragraph KNOWN_ISSUES.md
  recommends: state what matches, what changed (scale), and why. Fill in [X] once
  the final scale/seed count is locked in.
- This assumes ICMLDE's 10-page budget; trim the per-model equation restatements
  if space is tight and instead just cite [BASE] for the exact forms, keeping only
  the deltas we introduced (e.g. the vectorized GATv2 attention, if worth a line).
-->

## 3. Experimental Setup

**Feeder and simulation.** We simulate the IEEE 123-bus distribution feeder in
OpenDSS/OpenDSSDirect.py at its nominal 4.16 kV line-to-line voltage and 60 Hz frequency,
following the protocol in [BASE §4.2]. Each simulation run draws a load-scaling multiplier
$L \sim U(0.5, 1.3)$ independently per run, injects a fault at 1 ms resolution, and records
40 ms of pre-fault baseline followed by a 20 ms fault duration (60 ms total per event).

**Fault taxonomy and scenario space.** We inject faults at 25 predefined buses, covering 11
fault types — three single-line-to-ground (AG/BG/CG), three line-to-line (AB/BC/CA), three
double-line-to-ground (ABG/BCG/CAG), and two three-phase (ABC/ABCG) — at three fault
resistances (0.1, 1, 10 Ω), giving $25 \times 11 \times 3 = 825$ (location, type, resistance)
scenarios. 50 independently seeded load draws are simulated per scenario (reduced from the
paper's 100; see Reproduction Notes below), for 41,250 total fault events.

**μPMU placement and partial observability.** 25 μPMUs are placed at a fixed, sparse subset
of feeder buses (disjoint from the fault-injection buses in the general case), following
[BASE Table 1 / Fig. 2]. Each μPMU reports per-unit RMS voltage on all three phases; a binary
per-phase availability mask is carried alongside the voltage channel so that a model can
distinguish "phase absent from this μPMU's wiring" from "phase present but reads near zero."
This is the central constraint of the problem: fault buses are largely *not* directly
instrumented, so localization must be inferred from the sparse μPMUs' response to a fault
elsewhere on the feeder.

**Windowing and labels.** Each 60 ms recording is split into 40 overlapping 20 ms windows at
1 ms stride (starts 0–39); the first 20 windows (wholly pre-fault) are labeled the no-fault
class, the remaining 20 (each containing at least one post-onset sample) are labeled with the
injected fault bus's class, giving a fixed 20/20 per-event balance and a 26-class label space
(25 fault-location classes + 1 no-fault class). Each window is a $[N{=}25, S{=}20, F{=}6]$
tensor: 25 μPMU nodes, 20 timesteps, 6 features (3 per-unit voltage phases + 3 availability-mask
channels). We resolve one genuine ambiguity in the base paper's windowing description here: a
60 ms record with 20 ms windows at 1 ms stride admits 41, not 40, valid window starts; we retain
starts 0–39 and drop the endpoint at start 40 to preserve the paper's reported window count and
exact class balance (see `docs/reproduction_log.md` for the full derivation).

**Graph construction.** The message-passing graph is built over the 25 μPMU nodes only (a
"measured-only" graph), rooted at the substation, using Algorithm 1 from [BASE §4.1]: edges
follow the feeder's electrical topology restricted to instrumented buses, producing a connected,
acyclic graph (24 edges, verified via `nx.is_forest()` at build time). Edge attributes carry
line resistance, reactance, and impedance magnitude ($R$, $X$, $|Z|$) between adjacent measured
buses.

**Splits and normalization.** Events are split 70/15/15 into train/validation/test at the
*event* level (all windows from one event stay in one split, preventing window-level leakage
across a shared fault instant). Normalization statistics (mean/std per voltage channel) are
computed on the training split only and applied to all three splits; availability-mask channels
are left unnormalized (binary by construction).

### Reproduction Notes

We reproduce the feeder model, fault taxonomy, μPMU placement, windowing protocol, and graph
construction exactly as specified (or, where the source text was ambiguous, resolved via the
most literal reading consistent with the paper's other reported quantities — documented in full
in `docs/reproduction_log.md`). We depart from the paper in one respect: **runs per scenario**.
The paper reports 100 runs/scenario; because our focus in this paper is the *relative* effect
of sensor dropout and dropout-aware training, not exact replication of the paper's absolute
accuracy, we evaluate at 50 runs/scenario (41,250 total events, 1,650,000 windows). All five model
architectures — baselines and, later, our dropout-aware variant — are trained and evaluated on
this identical data, split, and protocol, so comparisons *among our own results* remain valid
regardless of how our absolute numbers compare to the paper's; we report our own reproduced
baselines as the reference point for all robustness claims that follow, per the framing above.

---

## 4. Model Architectures and Training Protocol

All five models share a common temporal front end: a GRU with hidden size 128 processes each
node's 20-timestep, 6-feature window independently (weights shared across nodes), producing a
$[N, 128]$ per-node embedding. The **GRU baseline** feeds this embedding directly to a shared
node classifier (batch normalization → ReLU → dropout(0.35) → linear, 26-way) with no spatial
term — every node's prediction is a function of its own signal alone, providing a purely
temporal reference point with zero spatial reasoning ability.

The four **spatio-temporal GNN (STGNN)** variants add one spatial message-passing hop between
the GRU encoder and the classifier, each with a 64-dimensional GNN hidden size:

- **RGCN** aggregates neighbor embeddings with symmetric-normalized adjacency weights
  ($w_{ij} = (\deg_i \deg_j)^{-1/2}$), self-loops included.
- **RGSAGE-Mean / RGSAGE-Max** concatenate each node's own embedding with a mean- or
  max-aggregated neighbor embedding (self-loop included in the aggregate), then project
  ([BASE Eq. 5]).
- **RGATv2** computes attention coefficients per directed edge via a shared transform of the
  concatenated node pair, LeakyReLU, and a learned attention vector, normalized per destination
  node via softmax over incoming edges, with 4 attention heads and attention dropout 0.3
  ([BASE Eq. 8], following the corrected GATv2 formulation of Brody et al. — shared $W$ applied
  to the *concatenated* pair before the nonlinearity, not to separate projections).

**Loss.** All models are trained with inverse-frequency-weighted cross-entropy over node-level
predictions (each event's single ground-truth label is expanded to every node, since the
graph-level prediction is a soft vote over per-node logits at test time). This choice is frozen
across the entire study: an earlier diagnostic isolated that fixing the roughly 50%/50% class
imbalance (no-fault vs. 25 fault classes) preserves detection but does *not* by itself fix
localization — confirming the GRU baseline's shortfall is architectural (lack of spatial
reasoning), not a training-imbalance artifact.

**Optimization.** AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 32. Epoch
budgets follow the paper's protocol: 11 epochs for the GRU baseline, 15 for each STGNN variant.
Training is deterministic per seed (fixed RNG state, deterministic CUDA kernels where available);
model selection uses the checkpoint with the highest validation soft-voted graph macro-F1, which
is then evaluated once on the held-out test split.

**Metrics.** We report soft-voted graph-level macro-F1 (one prediction per event, via majority/
confidence vote over the event's node-level logits) as the primary metric, alongside the
per-node ("diagnostic") macro-F1, a per-resistance breakdown (0.1/1/10 Ω), and no-fault
precision/recall as a detection-preserved sanity check. All labels 0–25 are included in the
macro average with `zero_division=0`, even for classes absent from a given evaluation split.
