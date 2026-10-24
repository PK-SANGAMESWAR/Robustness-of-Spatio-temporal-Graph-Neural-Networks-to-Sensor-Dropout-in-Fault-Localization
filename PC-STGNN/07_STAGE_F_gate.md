# Stage F — Trust Gate, PCRG-STGNN (Weeks 12–16)

**Purpose.** Address silent sensor failure — m = 1 but the value is wrong — which no mask-based
method, including Paper 1's, can see even in principle.
**Compute.** `[Corrected 2026-09-22]` 8 runs × ≈ 7–9 h (G2 has a second forward pass) ≈ 60–70
GPU-h.

## Context you need

`01_SPEC.md` §7. gᵢ = mᵢ · σ(−α(r̃ᵢ^LOO − c)). Gate the evidence (input embedding + soft-vote
weight), never the messages at every hop — Paper 1's SDR-STGNN reached AUDC 0.892 against 0.913
for plain dropout-aware training, because hop-level silencing cuts the multi-hop paths that route
around a lost node.

## S5.1 — Training-free detection — hard gate

**Do.** Using PSM hypotheses only, no training: compute r̃ᵢ^LOO on clean vs corrupted sensors, for
every corruption type and severity (`dropout/silent.py`, Stage D).

**Check.** AUROC per corruption type per severity. Also AUPRC and TPR at FPR = 0.05.

**Hard threshold.** If AUROC < 0.75 under the mildest corruption (±2 % bias), do not train PCRG.

**Branch on failure.** Publish PC-STGNN alone and report this as a bounded negative finding:
"physics residuals from magnitude-only measurements cannot detect sensor bias below X %."
Include the AUROC-vs-severity curve. That is a useful sentence for the field, it costs nothing,
and it is a better outcome than a gate that does not work.

**Figures produced.** F-F1 — ROC curves per corruption type. F-F2 — AUROC vs severity.

## S5.2 — LOO vs in-sample residual (free)

**Do.** Compute both residual forms; compare detection AUROC.

**Check.** LOO should be clearly better.

**Why.** An in-sample fit is pulled toward the corrupted sensor and masks its own error. This is a
one-figure, zero-cost justification of the LOO design — and a reviewer will ask.

**Figure produced.** F-F3 — LOO vs in-sample detection AUROC.

## S5.3 — Gate training

| # | Variant | Purpose | Seeds |
|---|---|---|---|
| G0 | dropout-aware + corruption augmentation, no physics | the critical control: "why not just augment with corruption?" | 3 |
| G1 | PC-STGNN + the same augmentation | isolates the gate from the augmentation | 1 |
| G2 | PCRG-STGNN — gate on evidence + vote weight | the headline result | 3 |
| G3 | gate at every hop | re-tests the Paper-1 SDR-STGNN lesson | 1 |

G0 is the one that matters. If G2 does not beat G0, the gate is not earning its complexity and the
honest move is to report that.

G3 is expected to underperform G2, reproducing the Paper-1 finding in a new setting — a nice
internal consistency result.

**Check.** Also record the gate value distribution gᵢ for clean vs corrupted sensors, and the
false-suppression rate (clean sensors with gᵢ < 0.5).

**Figure produced.** F-F4 — gᵢ distribution, clean vs corrupted.

## GATE 5

- [ ] S5.1 AUROC ≥ 0.75 at the mildest corruption
- [ ] S5.2 LOO beats in-sample
- [ ] G2 > G0 under silent failures, sign-consistent across seeds
- [ ] False-suppression rate on clean sensors acceptable (report it either way)

### S5.2 result, first run `[Recorded 2026-10-24]`

**The check does not hold.** The first run (sigma0 = 0.0, `results/pcstgnn/audit/s5_1_silent_detection/`,
`s5_2_psm_auroc`) finds the **in-sample** AUROC slightly **higher** than the LOO AUROC at bias
±2 % (the S5.1 gate corruption), for every s:

| s | LOO AUROC | in-sample AUROC | in-sample − LOO |
|---|---|---|---|
| 1 | 0.999134 | 0.999311 | +0.000178 |
| 2 | 0.998092 | 0.998519 | +0.000427 |
| 3 | 0.997061 | 0.997690 | +0.000629 |

The same direction holds for every other corruption type and severity in the first run (stuck,
bias ±5 %, Gaussian, swap), not only bias ±2 %. **"LOO should be clearly better" (S5.2's check,
above) does not hold at sigma0 = 0** — the sign is the wrong way, though the gap is tiny (third
decimal) and both forms are near the AUROC ceiling (> 0.997), so this is not read as evidence
against the LOO design, only as a miss on this specific check's stated expectation.

**Measurement-noise rerun (`scripts/run_silent_detection_noise.py`,
`results/stage_f/silent_detection_noise/`), sigma0 ∈ {5e-4, 2e-3, 1e-2}:**

- sigma0 = 5e-4: same direction, same order of magnitude. Bias ±2 %: LOO = 0.999044 / 0.997984 /
  0.996939 (s = 1/2/3), in-sample = 0.999228 / 0.998414 / 0.997573 — in-sample higher by
  +0.000184 / +0.000430 / +0.000634.
- sigma0 = 2e-3, 1e-2: run in progress at the time of this edit (background, CPU-only, low
  priority) — not yet appended. Add when `results/stage_f/silent_detection_noise/sigma_0.002/`
  and `sigma_0.01/` exist.
