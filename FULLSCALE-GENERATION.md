# FULLSCALE_GENERATION_SPEC.md — Definitive Reproduction Dataset (50→100 Extensible)

## Purpose
Generate the paper-reproduction dataset and run the definitive 5×3 baseline sweep.
Target 50 runs/scenario FIRST, built as an extensible prefix of 100, so runs 51–100
can be appended later WITHOUT regenerating runs 1–50.

## Preconditions (MUST be true before starting — do not skip)
- [ ] GPU confirmed working: `torch.cuda.is_available() == True`, and `nvidia-smi`
      shows activity during a real training step (NVML warning resolved).
- [ ] RGATv2 attention loop vectorized AND verified (numerical-equivalence check vs.
      old loop on a fixed input; gradients flow; attention semantics preserved).
- [ ] 1-epoch timing probe re-run on the fixed setup; full-scale sweep cost
      re-extrapolated with the NEW number. Confirm the 5×3 sweep is overnight-feasible.
- [ ] Disk check: ~1.65M windows (50 runs) fits with headroom; ~3.3M (100 runs) fits
      if extension is likely.

---

## 1. Extensible Generation Scheme (the key requirement)

Generate runs so that 1–50 is a stable PREFIX of the eventual 1–100.

- Per (scenario, run_index) pair, derive the RNG seed DETERMINISTICALLY, e.g.:
      seed(scenario_id, run_index) = base_seed + scenario_id * 1000 + run_index
  so run_index 0..49 are identical whether you later generate 0..99.
- Load scaling L~U(0.5,1.3), fault resistance selection, and any per-run randomness
  MUST draw from this per-(scenario,run) seed — NOT from a single global RNG whose
  stream would shift when you change the run count.
- Raw sim goes into the shared checkpointed pool (already built): appending runs
  51–100 simulates ONLY the new runs; 1–50 are reused untouched.
- Verify prefix-stability: regenerating run_index 0..49 after a 0..99 generation must
  produce byte-identical (or numerically identical) raw events.

### Scale parameters
| Param | Phase 1 (now) | Phase 2 (only if needed) |
|---|---|---|
| runs/scenario | 50 | 100 |
| scenarios | 825 (25 loc × 11 types × 3 R) | 825 |
| events | 41,250 | 82,500 |
| windows (×40) | ~1.65M | ~3.3M |
| feeder | default | default |

All other params fixed to paper protocol: 11 fault types, R∈{0.1,1,10}Ω, 1 ms sampling,
20 ms windows (40/event), F=6 features (3 voltage + 3 mask), 26-class labels.

---

## 2. Dataset Build (per scale)
- Build measured-only graph FIRST; assert `nx.is_forest()` before windowing.
- 70/15/15 EVENT-LEVEL split (no window leakage). Freeze split; store split files.
- Recompute train-ONLY normalization stats for the 50-run set (and again if extended
  to 100 — normalization must match the data actually used).
- Preserve full 26-class coverage; confirm ~50% no-fault balance.
- Pre-flight manifest check BEFORE training: assert event count = 41,250, window count
  ≈ 1.65M, all 26 classes present. Abort if mismatch (prevents a wasted overnight run).

---

## 3. Definitive Sweep (per scale)
- Models: gru, rgcn, rgsage_mean, rgsage_max, rgatv2
- Seeds: 42, 43, 44
- Loss: unweighted CE (FROZEN from Step 1)
- Epochs: paper values (GRU 11 / GNN 15) unless early-stopping triggers
- Resumable; one crash must not kill the sweep.
- Report per model, mean ± std over seeds:
  - soft-voted graph macro-F1 (PRIMARY)
  - per-window macro-F1
  - per-resistance macro-F1 (0.1 / 1 / 10 Ω)
  - no-fault precision + recall (detection preserved check)
- Output: results/fullscale_50/benchmark_summary.csv (+ _comparison.csv)

---

## 4. BENCHMARK GATE — stop at 50 vs. extend to 100

Read the BEST GNN soft-voted macro-F1 (mean over 3 seeds) at 50 runs.
Paper reference range: 0.86–0.95.

### ✅ STOP AT 50 — reproduction accepted at reduced scale
Criterion (any of):
- best-GNN macro-F1 ≥ 0.80 (within striking distance of paper), OR
- best-GNN macro-F1 ≥ 0.75 AND clearly still rising vs. the 20-run medium-A point
  (0.343) — i.e., a strong monotonic data-scaling trend (20→50 shows a large jump).
Action: Stop at 50. Document as a deliberate reduced-scale reproduction per
KNOWN_ISSUES (internal consistency > exact scale match). Proceed to green feeder.

### 🟡 EXTEND TO 100 — likely to reach paper range with more data
Criterion:
- best-GNN macro-F1 in ~0.60–0.80 AND still rising from the 20→50 trend.
Action: Append runs 51–100 (cheap raw sim), rebuild, re-run the 5×3 sweep at 100.
Re-apply this gate.

### ❌ INVESTIGATE — not a data problem
Criterion:
- best-GNN macro-F1 well below expectation (< ~0.55) despite 50 runs, OR the 20→50
  trend is flat/weak.
Action: STOP. Do NOT spend compute extending to 100. Investigate before scaling:
  - hyperparameters vs. paper (LR, epochs, hidden dims, GAT heads)
  - graph correctness at scale (edges actually used; edges-on/off gap still present)
  - label alignment / normalization applied to the correct split
  - per-resistance: is the shortfall concentrated in 10 Ω (physical) or across the board (bug)?

---

## 5. Reporting back
After the 50-run sweep, report:
1. Full results table (5 models × [soft-voted F1, per-window F1, per-R F1] × mean±std).
2. Best-GNN macro-F1 and the 20→50 trend (medium-A 0.343 → 50-run value).
3. GATE verdict: STOP-AT-50 / EXTEND-TO-100 / INVESTIGATE.
4. Confirmation that no-fault recall stayed >0.99 (detection preserved).

---

## 6. Hard rules
- Extensible seeding so 1–50 is a stable prefix of 1–100 (verify byte/numeric identity).
- All models on identical frozen data/splits at a given scale.
- ≥3 seeds, mean ± std everywhere.
- Pre-flight manifest check before any overnight run.
- Judge against paper by BOTH absolute value AND the data-scaling trend, not absolute alone.
Why it's built this way
50-first, extensible prefix = you never regenerate. Your checkpointed pool already supports this; the only new requirement is per-(scenario, run) deterministic seeding so the RNG stream for runs 1–50 doesn't shift when you add 51–100. That's the one subtle thing to get right, and the spec calls it out explicitly.
The gate uses trend + absolute, not just absolute. Your 20-run point (0.343) plus a 50-run point gives you a two-point scaling curve. If 50 runs is at, say, 0.78 and clearly climbing from 0.343, that's strong evidence 100 would clear 0.86 — so you can either extend or argue reproduction validity from the trend without necessarily running full 100.
Pre-flight manifest + preconditions stop you from burning an overnight run on a broken setup — the recurring lesson of this whole project.
Two honest caveats:

Window/event counts are approximate (runs × 825 × 40) — the spec makes the pipeline assert them, so a miscount aborts early instead of wasting the run.
The gate thresholds (0.80 / 0.75 / 0.55) are reasonable defaults, not sacred — if your supervisor or the paper suggests a specific bar, adjust. They're there to force a decision, not to be exact.
Do the preconditions first (GPU + RGATv2 + re-timing), then generate the 50-run set.