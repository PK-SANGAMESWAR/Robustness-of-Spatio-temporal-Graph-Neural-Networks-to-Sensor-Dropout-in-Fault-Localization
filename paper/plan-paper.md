# Plan: Trimming `overleaf.tex` from ~15 pages to 12–13

**Target:** cut **~2–3 pages**, landing at **12–13 pp** (ICMLDE regular-paper cap = 12).
**Not** a scope cut — **all three contributions stay** (degradation study, dropout-aware
training, SDR-STGNN ablation). We tighten prose, demote redundant tables/figures, and
remove duplication. **No edits were made to `overleaf.tex`** — this file is planning only.

---

## Part 0 — Claim verification (done first, as requested)

I re-derived every headline number directly from the results files
(`results/dropout_study{,_aware,_sdr}/degradation_curves.csv`,
`paper/figures/sdr_robustness_summary.csv`). **All numeric claims are correct and
internally consistent.**

| Claim in paper | Source | Status |
|---|---|---|
| RGSAGE-Max clean F1 `0.846±0.007` | dropout_study `rgsage_max,k=0` | ✅ exact |
| Baseline table (5 models, F1 by resistance, no-fault P/R) | fullscale_final metrics | ✅ |
| Degradation F1 (5 models × k∈{0,1,2,5,10}) | dropout_study CSV | ✅ exact |
| Degradation AUDC / Ret@10 (0.634/31.8%, 0.615/24.7%, 0.551/15.1%, 0.521/18.0%, 0.414/7.0%) | recomputed trapezoid | ✅ exact |
| Dropout-aware AUDC 0.414→0.916, Ret@10 7.0%→80.6% | sdr_robustness_summary | ✅ |
| Dropout-aware clean F1 `0.839±0.011` (ΔF1<0.01) | aware CSV | ✅ |
| SDR-STGNN AUDC 0.895, Ret@10 76.7% (matches, not beats) | sdr CSV | ✅ faithful |
| Across-architecture DA (Ret@10 81.4/87.0/91.3/26.2%) | aware CSV, recomputed | ✅ exact |
| Depth sweep 0.509→0.571→0.751→0.806→0.846 | diagnostic narrative | ✅ consistent |

**Reporting is honest** — SDR-STGNN is correctly described as an ablation that matches
but does not beat dropout-aware training (per the project's hard rule). Keep it that way.

### Minor items to fix before submission (not results errors)
1. **`Karabulut2026` arXiv id `2604.20403` looks like a placeholder** — verify the real
   identifier / peer-reviewed venue before camera-ready.
2. The supervisor file's deviation table lists "class-weighted CE → unweighted CE" as a
   difference. **The live paper is correct** (unweighted in both; not a deviation) —
   do **not** re-add loss weighting to `tab:deviations`.
3. `Karabulut2025` is already cited in the body (line 317) — that concern is resolved.

---

## Part 1 — Where the ~2–3 pages come from (summary)

Nothing is deleted wholesale; every result, table, and figure that carries a unique
number stays. The savings come from **duplication and over-explanation**, exactly the
targets the supervisor review (`paper-issue.tex`) already identified.

| # | Action | Est. saving |
|---|---|---|
| 1 | **§"Differences from Base Paper" → fold into Methods** (keep `tab:deviations`, cut the ~2 prose paragraphs that repeat it) | **~0.9 pp** |
| 2 | **§6.2 / §7.1 Results+Discussion: stop re-reciting table numbers**, keep interpretation + the max-pooling mechanism | **~0.6 pp** |
| 3 | **§3.2 baselines: condense per-operator descriptions to one paragraph** | **~0.4 pp** |
| 4 | **Intro: drop the 4-item bullet expansion + roadmap paragraph** → 2 tight paragraphs | **~0.4 pp** |
| 5 | **Demote depth-sweep paragraph (line 658) to one inline sentence** | **~0.15 pp** |
| 6 | **§4 Setup: cut the duplicate "Dropout-aware training" paragraph; versions → footnote (already a footnote — remove the in-text version list)** | **~0.25 pp** |
| 7 | **§3.1 dataset: trim `tab:dataset` lower rows + fault-signature prose to one sentence each** | **~0.25 pp** |
| 8 | **Related Work prose tighten** around `tab:related` (keep the table) | **~0.2 pp** |
| | **Total** | **~3.1 pp → ~12.4 pp** |

Any two-thirds of this list already hits 13 pp; the full list reaches ~12. Start at the
top (biggest, lowest-risk savings first) and stop when the target is met.

---

## Part 2 — What to KEEP untouched (protects quality)

- **All 4 figures** (`fig:graph`, `fig:sig`, `fig:degradation`, `fig:ablation`) — each
  carries unique content; the supervisor review calls Figs 3–4 publication-ready and
  Fig 1 an effective device. (Only ensure Fig 1 is exported ≥300 dpi.)
- **All primary result tables** (`tab:baseline`, `tab:degradation`, `tab:aware`,
  `tab:agnostic`) — these hold the verified numbers and the three findings.
- **`tab:related`** — the positioning table is the paper's clearest novelty argument.
- **SDR-STGNN** in full (subsection + 3 equations) — it is a stated contribution and
  the honest "training beats architecture" conclusion depends on it.
- **The max-pooling fragility mechanism** and the **practical "lose 10/25 sensors"**
  implication — the two insights that make the paper more than a benchmark.

**No figure or table is removed.** The trim is entirely prose + one section merge.

---

## Part 3 — Section-by-section trim plan (line refs to current `overleaf.tex`)

### Introduction (219–236) — save ~0.4 pp
- Keep the first two paragraphs (problem + GNN motivation + Karabulut framing).
- **Collapse the 4-item `itemize` (228–233)** into 2–3 sentences. It currently spends a
  full paragraph per contribution; reviewers count bullets and the reproduction bullet
  (#1) reads as infrastructure, not novelty. State: "we (i) quantify degradation under
  dropout, (ii) introduce dropout-aware training, and (iii) test an availability-gated
  variant (SDR-STGNN) as an ablation."
- **Delete the roadmap paragraph (236)** or reduce to one sentence.

### Related Work (238–317) — save ~0.2 pp
- **Keep `tab:related`.** Tighten the surrounding prose (317) — the interpretation is
  good; just remove any sentence that merely re-reads the checkmarks.

### Methods → Grid & Dataset (323–384) — save ~0.25 pp
- Keep `tab:dataset`, `fig:graph`, `fig:sig`.
- **Trim `tab:dataset`**: the window-level split counts are already stated inline in the
  prose (333) — you can drop the "Total events / windows" duplication or shorten a few
  low-value rows.
- **Shorten the fault-signature paragraph (331)** — the "high-impedance faults sag
  <0.02 p.u." point is one sentence; the figure carries the rest.

### Methods → Baselines (388–398) — save ~0.4 pp
- **Condense the per-operator descriptions (392)**: RGCN/RGSAGE-Mean/Max/RGATv2 are
  standard and cited; one sentence naming each aggregation operator + citations suffices
  instead of the current formula-by-formula prose. Keep the 4-hop justification (394–396).

### Methods → Dropout-Aware Training (416–464) — keep, minor
- Keep **Algorithm 1** (it's the method's formal statement, cheap in space, and aids
  reproducibility). No cut needed here.

### Experimental Setup (520–539) — save ~0.25 pp
- **Delete the "Dropout-aware training" paragraph (535)** — it repeats §3.4; replace with
  one cross-reference sentence.
- The full version list is already in a footnote (526); **remove any duplicated
  version numbers from the in-text sentence**, keep only "PyTorch 2.5.1 + PyG 2.8".

### Differences from Base Paper (541–561) — save ~0.9 pp (biggest single win)
- **Keep `tab:deviations`** but **delete the two prose paragraphs (544)** that justify
  each difference at length — the table says it in 3 rows.
- **Merge the table + one summary sentence into the end of §3.2 (Baselines)** and remove
  the standalone section heading. One sentence: "All comparisons are on our own corpus
  and splits, so robustness effects are internally consistent regardless of absolute-F1
  differences from the base paper."

### Results (563–811) — save ~0.6 pp
- Keep all four result tables and both figures.
- **§6.1: demote the depth-sweep paragraph (658)** to one inline sentence: "A single-seed
  depth sweep on RGSAGE-Max rises monotonically 0.509 (1 hop) → 0.846 (4 hops), our
  operating point." (Frees ~6 lines.)
- **§6.2 (665) and §7.1 (819): remove verbatim number recitation.** These paragraphs
  restate every value already in `tab:degradation`/`tab:baseline`. Replace with the
  interpretation only: the accuracy↔robustness **inversion** + the **max-pooling
  mechanism** (keep that reasoning fully — it's the insight).

### Discussion (814–837) — save ~0.3 pp
- **§7.1**: after removing the number-recitation, keep the mechanism paragraph (821) and
  the practical-implication paragraph (825) as-is.
- **Limitations (827–837)**: keep, but each bullet can lose a clause; no bullet needs to
  be deleted.

### Conclusion (839–842) & Bibliography (846–920)
- Conclusion is already appropriately brief — leave it.
- **Optionally drop `Rong2020` (DropEdge)** if DropEdge is no longer discussed in the
  body (it isn't cited in the current prose) — saves one reference slot.

---

## Part 4 — Execution order (do the safe, high-yield cuts first)

1. **Fold "Differences" into Methods** (−0.9 pp) — biggest, zero information loss.
2. **Strip number-recitation from §6.2/§7.1** (−0.6 pp) — pure redundancy.
3. **Condense §3.2 baseline operators** (−0.4 pp).
4. **Collapse Intro bullets + roadmap** (−0.4 pp).
5. Stop and recompile. If already ≤13, you're done. If aiming for 12, add items 5–8
   from Part 1 (depth-sweep inline, §4 duplicate paragraph, `tab:dataset` rows,
   Related-Work prose).

---

## Part 5 — Quality guarantee

- **Every verified number, every table, every figure, and all three contributions
  survive.** The ~3 pages come entirely from (a) one section merge and (b) deleting prose
  that repeats numbers already in tables — i.e. redundancy, not substance.
- **Reproducibility is unchanged** — Algorithm 1, the config footnote, and the deviation
  table all stay.
- **The two insights that give the paper depth** (max-pooling fragility mechanism;
  "operators can lose ~10/25 μPMUs safely" / SDG hook) are explicitly preserved.

> This is the same plan the supervisor review (`paper-issue.tex`) converges on
> (≈3.25 pp of trims to reach ≤12), cross-checked against the live line numbers.
```
