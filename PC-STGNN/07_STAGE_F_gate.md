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
| G-TF | R5 + training-free gate from the S5.1 detector (no training) | a floor for G2; how much a threshold alone recovers | 3 (existing R5 checkpoints) |

G0 is the one that matters. If G2 does not beat G0, the gate is not earning its complexity and the
honest move is to report that.

G3 is expected to underperform G2, reproducing the Paper-1 finding in a new setting — a nice
internal consistency result.

**Check.** Also record the gate value distribution gᵢ for clean vs corrupted sensors, and the
false-suppression rate (clean sensors with gᵢ < 0.5).

**Figure produced.** F-F4 — gᵢ distribution, clean vs corrupted.

### G-TF — training-free gate arm `[Added after S5.1, before any G-TF result — 2026-10-25 03:43 IST / 2026-10-24T22:13:54Z]`

Full working record: `stage_f_training_free_gate_draft.md` (same directory). Frozen here before any G-TF result.
**No G-TF scoring has been run.**

**What.** R5, unchanged (the 3 Gate 4 checkpoints, seeds 42/43/44; no training). The S5.1 detector flags sensors;
each flagged sensor is dropped (all six channels zeroed, the `zero_messages` operation). k = 0. Arms on the same
corrupted windows: **no gate**, **G-TF**, **oracle** (drops exactly the corrupted sensors). Clean-window arm: no
gate vs G-TF on clean windows (false-suppression cost). G-TF does not replace G0–G3; it sets a floor for G2.

| Item | Setting |
|---|---|
| Split | Validation windows split into halves A and B **by fault event**, `numpy.random.RandomState(7)`; no event in both halves (3,100 / 3,100 events; 124,000 / 124,000 windows) |
| Threshold | Per sensor, the 99th percentile of that sensor's available **clean** half-A scores at the same sigma0 (per-sensor false-positive rate 1 %). No corrupted window is used |
| Scoring | Half B only. Validation soft-voted 26-class macro-F1 |
| Detector | **In-sample** residual (rule: LOO only if LOO − in-sample ≥ 0.01 at bias ±2 %, sigma0 = 1e-2, for each of s = 1, 2, 3; measured +0.0046 / +0.0041 / +0.0036; applied 2026-10-25 03:43 IST) |
| Reported | Realized false-positive rate on clean half-B windows and on the clean sensors of corrupted half-B windows; TPR; every result per corruption type and per s |
| Core scope (first) | **all five corruption types** (stuck, bias ±2 %, bias ±5 %, Gaussian, swap), sigma0 = 1e-2, s = 1, 2, 3, seeds 42/43/44; arms no gate / G-TF / oracle; plus the clean-window arm. Amended 2026-10-25 15:42 IST, before any G-TF result. (Before: bias ±2 % only.) |
| Full scope | 5 types × s = 1, 2, 3 × sigma0 ∈ {0, 5e-4, 2e-3, 1e-2}; **only after the author approves** |

**Decision rule.** G-TF is useful if its macro-F1 is higher than R5 with no gate in **all 3 seeds** at bias ±2 %,
sigma0 = 1e-2 (read at each s = 1, 2, 3). Report the recovered fraction
`(G-TF − no gate) / (oracle − no gate)` with its denominator; **defined only if the denominator is ≥ 0.01**, otherwise reported as undefined and not averaged (Amended 2026-10-25 15:42 IST, before any G-TF result.). The corruption loss, F1(clean, no gate) − F1(corrupted, no gate), is reported per condition (Amended 2026-10-25 15:42 IST, before any G-TF result.). The rule uses bias ±2 % only; the other four types are reported separately. Paired t over seeds with Holm over the three s-levels
is secondary; n = 3, no trend upgraded.

**Cost (upper bound, from the measured R5 scoring rate 446.8 s per 248,000-window pass; half B is half).** Core 141
passes (all five types) ≈ **8.75 GPU-h** (Amended 2026-10-25 15:42 IST, before any G-TF result.; was 2.05 GPU-h for bias ±2 % only); full 564 passes ≈ **35.0 GPU-h**. CPU: threshold pass ≈ 17 min per sigma0 (extrapolated).

**Consistency checks (passed 2026-10-25 15:42 IST, before any G-TF result; Amended 2026-10-25 15:42 IST, before any G-TF result.).** (1) The gate drop equals the training dropout: `gate_zero` reproduces `apply_sensor_dropout_batch` bit for bit for k = 1, 2, 3, 5, 10, with identical R5 predictions. (2) The stored S5.1 detector scores equal the recomputed scores (largest absolute difference 0.0 over 6 batches, 1,920 windows, 15 conditions, sigma0 = 1e-2), with equal corrupted-sensor labels. The scorer repeats both checks and stops on a failure.

**Code.** `scripts/run_gtf.py` (stages split / thresholds / score / summarize / decide / check-ready),
`scripts/run_s4_4_queue_part2.py` (G-TF core, then R1, then R2 ×3, after the first queue completes).

#### G-TF core result `[Recorded 2026-10-26 00:16 IST / 2026-10-25T18:46:22Z]`

Core scope, as pre-registered above: R5 seeds 42/43/44, detector **in-sample**, sigma0 = 1e-2, k = 0, half B only (124,000 windows
per seed; 3,100 events), per-sensor threshold at 1 % on clean half A. Run by the merged queue from 2026-10-25T10:15Z to 18:45Z
(`results/stage_f/gtf/core/`; scoring 9,954–10,006 s per seed). Both consistency checks were repeated inside the scorer and passed.
F1 = validation soft-voted 26-class macro-F1. The detector flags do not depend on the model seed, so TPR and FPR are the same for the three seeds.

**Decision rule (bias ±2 %, all 3 seeds, each s): G-TF is USEFUL.** G-TF is higher than no gate in 9 of 9 seed-by-s pairs.
The gains are small (+0.0016 to +0.0166 macro-F1) and, with n = 3, **not significant after Holm correction** (p_Holm 0.130 / 0.130 / 0.075);
the sign rule is the primary criterion. This is not upgraded.

| s | G-TF − no gate (seeds 42/43/44) | all 3 positive | raw paired t p | Holm p (3 s-levels) |
|---|---|---|---|---|
| 1 | +0.0048 / +0.0018 / +0.0016 | yes | 0.116 | 0.130 |
| 2 | +0.0119 / +0.0052 / +0.0062 | yes | 0.065 | 0.130 |
| 3 | +0.0166 / +0.0099 / +0.0113 | yes | 0.025 | 0.075 |

**Corruption loss and F1** (loss = F1(clean half B, no gate) − F1(corrupted, no gate); clean no-gate F1 = 0.7460 / 0.7596 / 0.7458). Seed values 42 / 43 / 44:

| Corruption | s | Corruption loss (seeds 42/43/44) | mean | F1 no gate | F1 G-TF | F1 oracle |
|---|---|---|---|---|---|---|
| stuck | 1 | 0.0800 / 0.0719 / 0.0822 | 0.0780 | 0.6660 / 0.6877 / 0.6637 | 0.6660 / 0.6809 / 0.6624 | 0.7392 / 0.7538 / 0.7402 |
| stuck | 2 | 0.1701 / 0.1590 / 0.1743 | 0.1678 | 0.5759 / 0.6006 / 0.5715 | 0.5726 / 0.5893 / 0.5676 | 0.7331 / 0.7477 / 0.7339 |
| stuck | 3 | 0.2516 / 0.2444 / 0.2530 | 0.2497 | 0.4944 / 0.5152 / 0.4929 | 0.4859 / 0.5032 / 0.4844 | 0.7279 / 0.7424 / 0.7278 |
| bias ±2 % | 1 | 0.0195 / 0.0132 / 0.0121 | 0.0150 | 0.7265 / 0.7464 / 0.7337 | 0.7313 / 0.7482 / 0.7353 | 0.7392 / 0.7538 / 0.7402 |
| bias ±2 % | 2 | 0.0388 / 0.0261 / 0.0254 | 0.0301 | 0.7072 / 0.7335 / 0.7204 | 0.7191 / 0.7387 / 0.7266 | 0.7331 / 0.7477 / 0.7339 |
| bias ±2 % | 3 | 0.0567 / 0.0414 / 0.0410 | 0.0464 | 0.6893 / 0.7183 / 0.7048 | 0.7059 / 0.7282 / 0.7161 | 0.7279 / 0.7424 / 0.7278 |
| bias ±5 % | 1 | 0.0269 / 0.0489 / 0.0419 | 0.0392 | 0.7191 / 0.7107 / 0.7040 | 0.7358 / 0.7491 / 0.7370 | 0.7392 / 0.7538 / 0.7402 |
| bias ±5 % | 2 | 0.0594 / 0.0910 / 0.0767 | 0.0757 | 0.6867 / 0.6686 / 0.6691 | 0.7287 / 0.7399 / 0.7284 | 0.7331 / 0.7477 / 0.7339 |
| bias ±5 % | 3 | 0.0923 / 0.1302 / 0.1064 | 0.1096 | 0.6538 / 0.6294 / 0.6394 | 0.7207 / 0.7306 / 0.7199 | 0.7279 / 0.7424 / 0.7278 |
| Gaussian 0.02 p.u. | 1 | 0.0075 / 0.0074 / 0.0051 | 0.0067 | 0.7385 / 0.7522 / 0.7407 | 0.7380 / 0.7512 / 0.7389 | 0.7392 / 0.7538 / 0.7402 |
| Gaussian 0.02 p.u. | 2 | 0.0156 / 0.0131 / 0.0103 | 0.0130 | 0.7305 / 0.7465 / 0.7355 | 0.7322 / 0.7458 / 0.7335 | 0.7331 / 0.7477 / 0.7339 |
| Gaussian 0.02 p.u. | 3 | 0.0232 / 0.0215 / 0.0155 | 0.0201 | 0.7228 / 0.7381 / 0.7303 | 0.7247 / 0.7397 / 0.7288 | 0.7279 / 0.7424 / 0.7278 |
| swap | 1 | 0.1790 / 0.1848 / 0.1963 | 0.1867 | 0.5670 / 0.5748 / 0.5495 | 0.6387 / 0.6534 / 0.6384 | 0.7372 / 0.7513 / 0.7366 |
| swap | 2 | 0.3255 / 0.3409 / 0.3342 | 0.3335 | 0.4205 / 0.4188 / 0.4116 | 0.5254 / 0.5477 / 0.5277 | 0.7268 / 0.7409 / 0.7287 |
| swap | 3 | 0.4275 / 0.4501 / 0.4248 | 0.4341 | 0.3185 / 0.3095 / 0.3210 | 0.4333 / 0.4546 / 0.4321 | 0.7189 / 0.7342 / 0.7196 |

**Recovered fraction** `rec = (G-TF − no gate) / (oracle − no gate)`, defined only if the denominator is ≥ 0.01:

| Corruption | s | rec (seeds 42/43/44) | denominator = oracle − no gate (seeds) | mean rec |
|---|---|---|---|---|
| stuck | 1 | 0.000 / -0.102 / -0.016 | 0.0732 / 0.0661 / 0.0765 | -0.039 |
| stuck | 2 | -0.021 / -0.077 / -0.024 | 0.1571 / 0.1471 / 0.1624 | -0.041 |
| stuck | 3 | -0.036 / -0.053 / -0.036 | 0.2334 / 0.2272 / 0.2349 | -0.042 |
| bias ±2 % | 1 | 0.374 / undefined / undefined | 0.0127 / 0.0074 / 0.0064 | undefined (a seed is undefined) |
| bias ±2 % | 2 | 0.459 / 0.369 / 0.457 | 0.0259 / 0.0142 / 0.0135 | 0.428 |
| bias ±2 % | 3 | 0.430 / 0.410 / 0.492 | 0.0386 / 0.0241 / 0.0230 | 0.444 |
| bias ±5 % | 1 | 0.831 / 0.890 / 0.914 | 0.0201 / 0.0431 / 0.0362 | 0.878 |
| bias ±5 % | 2 | 0.906 / 0.902 / 0.914 | 0.0464 / 0.0790 / 0.0648 | 0.907 |
| bias ±5 % | 3 | 0.904 / 0.896 / 0.910 | 0.0741 / 0.1130 / 0.0884 | 0.903 |
| Gaussian 0.02 p.u. | 1 | undefined / undefined / undefined | 0.0007 / 0.0016 / -0.0005 | undefined (a seed is undefined) |
| Gaussian 0.02 p.u. | 2 | undefined / undefined / undefined | 0.0026 / 0.0011 / -0.0016 | undefined (a seed is undefined) |
| Gaussian 0.02 p.u. | 3 | undefined / undefined / undefined | 0.0050 / 0.0043 / -0.0025 | undefined (a seed is undefined) |
| swap | 1 | 0.422 / 0.445 / 0.475 | 0.1701 / 0.1765 / 0.1871 | 0.447 |
| swap | 2 | 0.343 / 0.400 / 0.366 | 0.3063 / 0.3222 / 0.3171 | 0.370 |
| swap | 3 | 0.287 / 0.342 / 0.279 | 0.4004 / 0.4246 / 0.3986 | 0.302 |

**Detector behaviour** (threshold set for 1 % per sensor on clean half A):

| Corruption | s | TPR of corrupted sensors | realized FPR, clean sensors of corrupted windows | sensors flagged per window |
|---|---|---|---|---|
| stuck | 1 | 0.296 | 0.0565 | 1.65 |
| stuck | 2 | 0.294 | 0.1051 | 3.00 |
| stuck | 3 | 0.291 | 0.1422 | 4.00 |
| bias ±2 % | 1 | 0.404 | 0.0108 | 0.66 |
| bias ±2 % | 2 | 0.408 | 0.0115 | 1.08 |
| bias ±2 % | 3 | 0.411 | 0.0122 | 1.50 |
| bias ±5 % | 1 | 0.972 | 0.0156 | 1.35 |
| bias ±5 % | 2 | 0.970 | 0.0215 | 2.43 |
| bias ±5 % | 3 | 0.968 | 0.0275 | 3.51 |
| Gaussian 0.02 p.u. | 1 | 0.378 | 0.0106 | 0.63 |
| Gaussian 0.02 p.u. | 2 | 0.380 | 0.0112 | 1.02 |
| Gaussian 0.02 p.u. | 3 | 0.382 | 0.0118 | 1.41 |
| swap | 1 | 0.997 | 0.1214 | 3.91 |
| swap | 2 | 0.997 | 0.2357 | 7.42 |
| swap | 3 | 0.998 | 0.3375 | 10.42 |

**Clean-window arm** (cost of false suppression):

| Seed | F1 no gate | F1 G-TF | G-TF − no gate | realized FPR, clean windows | sensors flagged per window |
|---|---|---|---|---|---|
| 42 | 0.7460 | 0.7450 | -0.0010 | 0.0101 | 0.252 |
| 43 | 0.7596 | 0.7576 | -0.0020 | 0.0101 | 0.252 |
| 44 | 0.7458 | 0.7443 | -0.0015 | 0.0101 | 0.252 |

**Reading (descriptive; only the bias ±2 % rows enter the decision rule).**

- Bias ±2 %: the loss is small (0.015 / 0.030 / 0.046 at s = 1 / 2 / 3). G-TF recovers about 0.43 and 0.44 of it at s = 2 and 3 (all seeds defined); at s = 1 `rec` is undefined in 2 of 3 seeds because the loss is below 0.01 in the denominator. The detector finds 40 % of the corrupted sensors (TPR 0.40–0.41) at a realized FPR of 1.1–1.2 %, close to the 1 % target.
- Bias ±5 %: TPR 0.97, mean `rec` 0.88–0.91 (per seed 0.83–0.91).
- Swap: TPR 0.997, mean `rec` 0.30–0.45 (per seed 0.28–0.48), but the realized FPR on the clean sensors of corrupted windows is **12 % (s = 1) to 34 % (s = 3)**, 3.9 to 10.4 sensors flagged per window: a swapped sensor contaminates the scores of its neighbours (open issue 2). The gain is still positive.
- Stuck: TPR 0.29, **`rec` is zero or negative in every seed and s** (0.000 to −0.10; negative in 8 of 9), i.e. G-TF is no better than no gate and usually a little worse; the realized FPR is 5.7–14 %.
- Gaussian 0.02 p.u.: the loss is 0.007–0.020, so `rec` is undefined in most seeds; G-TF is within ±0.002 of no gate.
- Clean windows: G-TF is lower than no gate by 0.0010 / 0.0020 / 0.0015, and the realized FPR is 1.01 %, as designed.
- Limits: one fixed corruption and noise draw (n = 3 is over model seeds), sigma0 = 1e-2 only (the weakest detector setting), half B only, k = 0 only. Full scope has not been run and needs the author's approval.

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
- sigma0 = 2e-3, 1e-2: complete; recorded in the next subsection. (This bullet was "run in progress" in
  the first version of this record.)

### S5.2 result, measurement-noise rerun `[Recorded 2026-10-25 03:43 IST]`

Rerun of S5.1/S5.2 at sigma0 ∈ {5e-4, 2e-3, 1e-2} p.u. (`scripts/run_silent_detection_noise.py`, `results/stage_f/silent_detection_noise/`; same corruptions, same corruption seed, full 248,000-window validation split, k = 0; noise is added before the corruption and sigma0 in the residual equals the injected sigma). AUROC of the PSM-refit residual, per sensor, positives = corrupted sensors, negatives = clean sensors of the corrupted windows. **LOO** = leave-one-sensor-out score, **ins** = in-sample score.

**S5.1 hard gate (bias ±2 %, LOO AUROC; threshold 0.75):**

| sigma0 (p.u.) | s = 1 | s = 2 | s = 3 | passed |
|---|---|---|---|---|
| 0 | 0.9991 | 0.9981 | 0.9971 | yes |
| 0.0005 | 0.9990 | 0.9980 | 0.9969 | yes |
| 0.002 | 0.9971 | 0.9959 | 0.9946 | yes |
| 0.01 | 0.8537 | 0.8504 | 0.8482 | yes |

**LOO − in-sample AUROC, bias ±2 % (S5.2 check: "LOO should be clearly better"):**

| sigma0 (p.u.) | s = 1 | s = 2 | s = 3 | conditions (all 5 types × 3 s) with LOO > in-sample |
|---|---|---|---|---|
| 0 | -0.0002 | -0.0004 | -0.0006 | 0 of 15 (range -0.0012 to -0.0002) |
| 0.0005 | -0.0002 | -0.0004 | -0.0006 | 0 of 15 (range -0.0012 to -0.0002) |
| 0.002 | -0.0003 | -0.0005 | -0.0007 | 0 of 15 (range -0.0011 to -0.0002) |
| 0.01 | +0.0046 | +0.0041 | +0.0036 | 9 of 15 (range -0.0006 to +0.0046) |

**sigma0 = 0.0005 p.u. — all five corruption types (AUROC):**

| Corruption | s | LOO | in-sample | LOO − in-sample |
|---|---|---|---|---|
| bias ±2 % | 1 | 0.9990 | 0.9992 | -0.0002 |
| bias ±2 % | 2 | 0.9980 | 0.9984 | -0.0004 |
| bias ±2 % | 3 | 0.9969 | 0.9976 | -0.0006 |
| bias ±5 % | 1 | 0.9995 | 0.9997 | -0.0002 |
| bias ±5 % | 2 | 0.9990 | 0.9993 | -0.0003 |
| bias ±5 % | 3 | 0.9984 | 0.9988 | -0.0004 |
| Gaussian 0.02 p.u. | 1 | 0.9725 | 0.9732 | -0.0007 |
| Gaussian 0.02 p.u. | 2 | 0.9642 | 0.9652 | -0.0010 |
| Gaussian 0.02 p.u. | 3 | 0.9580 | 0.9591 | -0.0012 |
| swap | 1 | 0.9969 | 0.9973 | -0.0004 |
| swap | 2 | 0.9908 | 0.9913 | -0.0005 |
| swap | 3 | 0.9788 | 0.9793 | -0.0005 |
| stuck | 1 | 0.6424 | 0.6436 | -0.0011 |
| stuck | 2 | 0.6092 | 0.6101 | -0.0009 |
| stuck | 3 | 0.5893 | 0.5899 | -0.0007 |

**sigma0 = 0.002 p.u. — all five corruption types (AUROC):**

| Corruption | s | LOO | in-sample | LOO − in-sample |
|---|---|---|---|---|
| bias ±2 % | 1 | 0.9971 | 0.9974 | -0.0003 |
| bias ±2 % | 2 | 0.9959 | 0.9964 | -0.0005 |
| bias ±2 % | 3 | 0.9946 | 0.9953 | -0.0007 |
| bias ±5 % | 1 | 0.9994 | 0.9995 | -0.0002 |
| bias ±5 % | 2 | 0.9987 | 0.9990 | -0.0003 |
| bias ±5 % | 3 | 0.9981 | 0.9985 | -0.0004 |
| Gaussian 0.02 p.u. | 1 | 0.9512 | 0.9521 | -0.0008 |
| Gaussian 0.02 p.u. | 2 | 0.9444 | 0.9452 | -0.0008 |
| Gaussian 0.02 p.u. | 3 | 0.9399 | 0.9409 | -0.0009 |
| swap | 1 | 0.9968 | 0.9972 | -0.0004 |
| swap | 2 | 0.9907 | 0.9912 | -0.0005 |
| swap | 3 | 0.9787 | 0.9792 | -0.0005 |
| stuck | 1 | 0.6391 | 0.6403 | -0.0011 |
| stuck | 2 | 0.6073 | 0.6081 | -0.0009 |
| stuck | 3 | 0.5878 | 0.5885 | -0.0007 |

**sigma0 = 0.01 p.u. — all five corruption types (AUROC):**

| Corruption | s | LOO | in-sample | LOO − in-sample |
|---|---|---|---|---|
| bias ±2 % | 1 | 0.8537 | 0.8491 | +0.0046 |
| bias ±2 % | 2 | 0.8504 | 0.8464 | +0.0041 |
| bias ±2 % | 3 | 0.8482 | 0.8446 | +0.0036 |
| bias ±5 % | 1 | 0.9908 | 0.9911 | -0.0003 |
| bias ±5 % | 2 | 0.9892 | 0.9896 | -0.0004 |
| bias ±5 % | 3 | 0.9878 | 0.9883 | -0.0005 |
| Gaussian 0.02 p.u. | 1 | 0.7833 | 0.7829 | +0.0004 |
| Gaussian 0.02 p.u. | 2 | 0.7812 | 0.7809 | +0.0003 |
| Gaussian 0.02 p.u. | 3 | 0.7802 | 0.7801 | +0.0001 |
| swap | 1 | 0.9959 | 0.9963 | -0.0004 |
| swap | 2 | 0.9894 | 0.9900 | -0.0006 |
| swap | 3 | 0.9771 | 0.9777 | -0.0006 |
| stuck | 1 | 0.6235 | 0.6227 | +0.0008 |
| stuck | 2 | 0.6002 | 0.5998 | +0.0004 |
| stuck | 3 | 0.5823 | 0.5820 | +0.0003 |

**Reading.**

- **S5.1 passes at every sigma0** (bias ±2 %, LOO AUROC ≥ 0.75). The AUROC is 0.995–0.999 up to sigma0 = 2e-3
  and falls to **0.848–0.854 at sigma0 = 1e-2**, still above the 0.75 threshold.
- **S5.2 ("LOO should be clearly better") is not met at any sigma0.** At sigma0 = 0, 5e-4 and 2e-3 the in-sample
  AUROC is higher in all 15 conditions (by 0.0002 to 0.0012). At sigma0 = 1e-2 the sign changes at bias ±2 %:
  LOO is higher by +0.0046 / +0.0041 / +0.0036 at s = 1/2/3, and in 9 of 15 conditions overall (range −0.0006
  to +0.0046). This is the direction the design predicted, but the largest gap is under half of one percentage point
  of AUROC.
- **Detector rule for G-TF** (`stage_f_training_free_gate_draft.md` § 2 item c: LOO only if LOO − in-sample ≥ 0.01
  at bias ±2 %, sigma0 = 1e-2, for each s): not met (+0.0046 / +0.0041 / +0.0036), so the **in-sample** residual
  is used. Decision file: `results/stage_f/gtf/detector_decision.json`.
- The Gate 5 checkboxes above are not ticked; what S5.2 means for the LOO design in `01_SPEC.md` §7 is recorded
  in the amendment note there.
