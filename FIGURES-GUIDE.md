# FIGURES-GUIDE.md — what every figure shows and how to explain it

A walkthrough of every figure in the project, **ranked most important → least** for
explaining the work. For each: the file, what is plotted (axes), the takeaway, the
key numbers, and a one-line talking point. Numbers are the final multi-seed values
(50-run corpus; RGSAGE-Max / dropout-aware / SDR-STGNN at seeds 42/43/44).

**Legend for "In paper?"** — ✅ already inserted in `overleaf.tex`; ⏳ generated,
belongs in the (still-empty) Results section; ➖ supporting/diagnostic, may not make
the page limit.

---

## TIER 1 — The contribution (these ARE the paper). Lead with these.

### 1. `paper/figures/sdr_ablation_backbone.png` — the headline figure ⏳
- **Plots:** soft-voted graph macro-F1 (y) vs number of μPMUs dropped at inference,
  k ∈ {0,1,2,5,10} (x). **Three curves on the *same* RGSAGE-Max backbone:**
  standard training (dashed), + dropout-aware training (solid), SDR-STGNN gated (bold).
- **Shows:** the entire thesis in one plot. All three start at the same clean accuracy
  (~0.84–0.85 at k=0), then the standard model **collapses** as sensors drop while the
  two robust variants stay high.
- **Numbers:** at k=10 (40% of sensors gone) — standard **0.06**, dropout-aware **0.68**,
  SDR-STGNN **0.64**.
- **Say to sir:** *"Same architecture, same data — the only change is how we train / gate.
  Standard training falls off a cliff; dropout-aware training keeps it up. This one figure
  is the paper."*

### 2. `paper/figures/sdr_retention.png` — robustness, normalised ⏳
- **Plots:** **retention** = F1(k)/F1(0) as a percentage (y) vs k (x), same three variants.
  A dotted line at 100% marks "no loss."
- **Shows:** the *shape* of degradation with the starting-accuracy advantage removed, so
  "flatter = more robust" is unmistakable. Standard drops toward 0%; the robust variants
  stay near the top.
- **Numbers:** retention at k=10 — standard **7%**, dropout-aware **81%**, SDR-STGNN **77%**.
- **Say to sir:** *"Normalised so it's purely about robustness, not raw accuracy. Dropout-aware
  training retains 81% of accuracy after losing 40% of sensors; the untrained model retains 7%."*

### 3. `paper/figures/sdr_audc_bar.png` — the single robustness number ⏳
- **Plots:** one bar per variant = **normalised AUDC** (area under the retention curve, 0–1).
  Higher = degrades more gracefully. Value printed on each bar.
- **Shows:** compresses each whole curve into one comparable scalar.
- **Numbers:** standard **0.414**, dropout-aware **0.916**, SDR-STGNN **0.895**.
- **Say to sir:** *"If you want one number: robustness score goes 0.41 → 0.92 with dropout-aware
  training — more than doubles. The gated architecture (0.89) matches it but does not beat it."*
- ⚠️ **Honesty point (important):** SDR-STGNN (0.895) is **slightly below** dropout-aware (0.916).
  We report this faithfully — the gated architecture is an honest ablation, **not** a winner.
  The gate silences dropped nodes at *every* hop, which also blocks useful multi-hop routing
  through recovered neighbours. **Do not claim the gate wins.**

### 4. `paper/figures/sdr_degradation_all.png` — all five baselines + SDR-STGNN ⏳
- **Plots:** macro-F1 (y) vs k (x) for **all 5 standard baselines** (GRU, RGCN, RGSAGE-Mean,
  RGSAGE-Max, RGATv2) with **SDR-STGNN overlaid in bold**. Error bars = std over mask seeds.
- **Shows:** none of the standard architectures is robust — the problem is universal, not a
  quirk of one model — which is what *motivates* the contribution.
- **Say to sir:** *"Every off-the-shelf model degrades badly under sensor loss; that's the gap
  we close. This figure sets up why the fix is needed."*

### 5. `paper/figures/sdr_heatmap.png` — everything at a glance ⏳
- **Plots:** a model × k grid coloured by macro-F1 (green = high), every cell annotated.
- **Shows:** a compact table-as-picture of all models across all dropout levels — good backup
  slide / appendix, and a quick way to read off any single value.
- **Say to sir:** *"The same data as the curves, but as a grid if you want to point at one cell."*

---

## TIER 2 — Mechanism & method (explains HOW / WHY). Use when sir asks "how does it work?"

### 6. `paper/figures/sdr_mechanism.png` — why a dead sensor corrupts neighbours ⏳
- **Plots:** a small schematic graph, twice — **standard** message passing vs
  **availability-gated** message passing — with a live sensor (green) and a dropped one (red).
- **Shows:** in the standard case the dead node still sends a (zeroed, meaningless) message into
  its live neighbours' aggregation; the gate removes exactly that edge so the corruption can't happen.
- **Say to sir:** *"This is the intuition for the architecture: a dropped sensor shouldn't get a
  vote. The gate enforces that structurally."* (No data — safe to show any time.)

### 7. `paper/figures/sdr_architecture.png` — the SDR-STGNN pipeline ⏳
- **Plots:** the forward pipeline — GRU temporal encoder → gated graph message passing (the novel
  block highlighted in red) → per-node head → soft vote — with the availability gate called out.
- **Shows:** where the one new component sits inside an otherwise-standard STGNN.
- **Say to sir:** *"Everything is the RGSAGE-Max backbone except the red block — that's the only
  architectural change we introduce."*

### 8. `paper/figures/sdr_protocol.png` — how we measure robustness ⏳
- **Plots:** the sensor-dropout evaluation protocol — k of 25 μPMUs dropped at inference, drawn
  from paired/fixed seeds shared across models, swept over k.
- **Shows:** the experimental protocol behind every Tier-1 curve; explains what "drop k sensors" means.
- **Say to sir:** *"We freeze a trained model and progressively black out sensors at test time,
  identically for every model, so the comparison is fair."*

---

## TIER 3 — Baseline reproduction & problem setup (supports "we reproduced the benchmark").

### 9. `results/fullscale_final/figures/graph_macro_f1_by_model.png` — baseline accuracy ⏳
- **Plots:** clean-test (k=0) soft-voted macro-F1, one bar per baseline model.
- **Shows:** the reproduced benchmark and **why RGSAGE-Max is the backbone** — it's the best.
- **Numbers:** GRU **0.30**, RGATv2 **0.61**, RGCN **0.73**, RGSAGE-Mean **0.81**,
  **RGSAGE-Max 0.846 ± 0.006** (best).
- **Say to sir:** *"We reproduced all five architectures; RGSAGE-Max is strongest, so we build
  the contribution on it."*

### 10. `paper/figures/measured_only_graph.png` — the sensing graph ✅ (in paper)
- **Plots:** the 25 μPMU buses (coloured by phase count) and the 24 topology-derived edges forming
  a connected tree; red rings mark the 6 μPMUs that are also fault locations.
- **Shows:** the partial-observability setup — only 6 of 25 fault locations are directly measured;
  the other 19 must be inferred through the graph.
- **Say to sir:** *"This is the hard part of the problem: most faults happen where there's no sensor."*

### 11. `paper/figures/fault_signature.png` — why high-impedance faults are hard ✅ (in paper)
- **Plots:** a real fault's voltage sag (deepest-sagging μPMU channel, normalised to pre-fault mean)
  for the three fault resistances.
- **Shows:** sag depth shrinks sharply with resistance — a 0.1 Ω fault sags ~0.83 p.u., a 10 Ω fault
  under 0.02 p.u. — foreshadowing why 10 Ω faults are the hardest to localise.
- **Say to sir:** *"High-resistance faults barely move the voltage, so they're the hardest — this
  motivates the per-resistance analysis."*

### 12. `results/fullscale_final/figures/per_resistance_f1.png` — accuracy vs fault resistance ➖
- **Plots:** macro-F1 at 0.1 / 1 / 10 Ω (blue ramp: light = low R / easy, dark = high R / hard),
  grouped by model.
- **Shows:** confirms the story of Fig. 11 in the results — accuracy is lowest at 10 Ω.
- **Say to sir:** *"The models are accurate on bolted faults and struggle on high-impedance ones,
  exactly as the signal predicts."*

### 13. `results/fullscale_final/figures/nofault_precision_recall.png` — detection preserved ➖
- **Plots:** precision and recall of the no-fault class (class 0), per model.
- **Shows:** the models don't achieve localisation accuracy by sacrificing basic fault detection —
  no-fault precision/recall stay high.
- **Say to sir:** *"They still reliably tell fault from no-fault; localisation isn't bought by
  false alarms."*

### 14. `results/fullscale_final/figures/confusion_matrix_rgsage_max.png` — where errors go ➖
- **Plots:** row-normalised confusion matrix for the best model (RGSAGE-Max) over the 26 classes.
- **Shows:** which fault locations get confused with which — errors cluster among electrically
  nearby, unmeasured buses, not at random.
- **Say to sir:** *"When it's wrong, it confuses neighbouring unmeasured locations — a sensible,
  physically-explainable error pattern."*

### 15. `results/fullscale_final/figures/training_curves_val_f1.png` — convergence audit ➖
- **Plots:** validation graph macro-F1 vs epoch, per model.
- **Shows:** training is stable (grad-clip 1.0 + 3-epoch warmup) and converged within 30 epochs —
  a sanity/reproducibility check, not a result.
- **Say to sir:** *"Evidence the runs converged cleanly and nothing diverged — a methods-appendix figure."*

---

## Orphans / not for the paper (mention only if asked)
- `paper/figures/stgnn_dataset_size_breakdown.png` — no generating script found; likely a
  hand-made dataset-size explainer. **Verify or regenerate before using.**
- `paper/figures/*112448.png` (screenshot-style name) — orphan; not a paper asset.

---

## Quick "which figures for the talk" cheat-sheet
- **3-slide version:** #10 (problem) → #1 (headline ablation) → #3 (AUDC number).
- **Full story:** #10, #11 (setup) → #9 (baselines reproduced) → #8, #6, #7 (protocol +
  mechanism + architecture) → #4, #1, #2, #3 (results) → #5 (backup grid).
- **The one honesty slide to rehearse:** #3 — be ready to say the gated architecture *matches
  but does not beat* dropout-aware training, and why (gate blocks multi-hop routing through
  recovered nodes). Sir will respect the honest framing.
