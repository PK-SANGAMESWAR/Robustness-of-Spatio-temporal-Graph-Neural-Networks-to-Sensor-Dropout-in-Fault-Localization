# Stage F — G-TF2: a training-free gate with two detectors (pre-registration)

**Status.** Pre-registered 2026-10-26 09:13 IST / 2026-10-26T03:43:56Z, **before any G-TF2 result**. No G-TF2 scoring has been run. No G-TF2
threshold has been computed. The decision to replace PCRG (G0 to G3) by this arm is in `07_STAGE_F_gate.md`,
section "Stage F amendment: PCRG replaced by G-TF2". The G-TF record (`07_STAGE_F_gate.md` § G-TF, and
`stage_f_training_free_gate_draft.md`) is not changed.

## 1. Question

G-TF (one detector, the in-sample residual) recovered about 0.9 of the bias ±5 % loss and 0.30 to 0.45 of the swap loss,
but nothing for stuck sensors (`rec` −0.04 at every s; 8 of 9 seed-by-s values negative). The residual does not see a
stuck sensor: its AUROC is 0.58 to 0.62 at sigma0 = 1e-2. A temporal flatline check does: AUROC 1.000.
Does adding the flatline check as a second detector make a training-free gate on R5 better than no gate for stuck
sensors, without losing what G-TF recovers for the other corruption types?

## 2. Detector 2 was chosen after an exploratory test — read this first

The temporal flatline score was **not** chosen before looking at data. It was chosen after the exploratory test
`scripts/run_temporal_detector_exploratory.py` (`results/stage_f/temporal_detector/temporal_detector.json`, labelled
"EXPLORATORY, NOT PRE-REGISTERED, not used in any decision rule"). What that test showed, at sigma0 = 1e-2, half B,
all 15 conditions:

| Corruption | AUROC, temporal flatline score | AUROC, in-sample residual |
|---|---|---|
| stuck, s = 1 / 2 / 3 | 1.000 / 1.000 / 1.000 | 0.624 / 0.600 / 0.583 |
| bias ±2 % | 0.500 / 0.500 / 0.500 | 0.849 / 0.847 / 0.844 |
| bias ±5 % | 0.501 / 0.501 / 0.502 | 0.991 / 0.989 / 0.988 |
| Gaussian 0.02 p.u. | 0.207 / 0.207 / 0.207 | 0.783 / 0.781 / 0.780 |
| swap | 0.455 / 0.455 / 0.454 | 0.996 / 0.990 / 0.978 |

Three facts the reader must keep in mind:

1. **Selection used half B.** The exploratory test ran on half B, the same half G-TF2 is scored on. It looked at AUROC
   only (no F1, no gate). The G-TF2 thresholds are set on half A, from clean windows only. The selection of the detector
   is still not independent of the scoring half. It is a disclosed limit, not a hidden one.
2. **The stuck corruption is exactly flat.** In `dropout/silent.py` a stuck sensor repeats one reading, so its standard
   deviation over the window is exactly 0 (fraction 1.0 of corrupted sensors; 0.0 of clean sensors). AUROC 1.000 follows from
   how the corruption is simulated. A real stuck sensor that keeps a little measurement noise would be harder. G-TF2 tests
   the gate on this simulated stuck sensor only.
3. **The flatline score does not detect the other types** (AUROC 0.50 for bias, 0.45 for swap, 0.21 for Gaussian). It should add
   nothing for them and cost only its false positives. That is a prediction the results can contradict.

## 3. Pre-registered design

Everything not listed here is as in G-TF (`stage_f_training_free_gate_draft.md` § 2): the split, the model, the gate action,
the oracle, the corruption and noise draws, the metric, the validation-only rule.

| Item | Setting |
|---|---|
| **Split** | As G-TF: validation windows split into halves A and B by fault event, `RandomState(7)`; 3,100 / 3,100 events; 124,000 / 124,000 windows. Thresholds from half A; scoring on half B only |
| **Model** | R5, `results/s4_4/r5/pcstgnn/seed_{00042,00043,00044}/checkpoint_best.pt`. No training. k = 0 |
| **Detector 1** | The **in-sample** PSM residual (`psm_ins`), the same score as G-TF. Never LOO |
| **Detector 2** | The **temporal flatline score**: the standard deviation of a sensor's voltage readings over the 20-step window (p.u., population std), averaged over the sensor's instrumented phases. `run_gtf.flatline_score` returns minus this value so that "higher = more flat" |
| **Flag rule** | A sensor is flagged if its residual is **above** its detector-1 threshold **OR** its standard deviation is **below** its detector-2 threshold. An unavailable sensor is never flagged |
| **Thresholds** | Per sensor, on the **clean** windows of half A (noise at sigma0 = 1e-2, no corruption): detector 1 at the 99.5th percentile of that sensor's residual scores; detector 2 at the 0.5th percentile of that sensor's standard deviation (`numpy.quantile`, linear). That is a per-sensor false-positive rate of **0.5 % for each detector**; the union is at most about 1 % and is **measured, not assumed**. 25 + 25 thresholds. No corrupted window is used |
| **Gate action** | A flagged sensor is dropped: all six channels set to 0 (the `zero_messages` operation), as in G-TF |
| **Arms** | **no gate**, **G-TF2**, **oracle** (drops exactly the corrupted sensors), on the same corrupted half-B windows. **Clean-window arm**: no gate vs G-TF2 on clean half-B windows |
| **Scope** | **All 5 corruption types** (stuck, bias ±2 %, bias ±5 %, Gaussian 0.02 p.u., swap), **sigma0 = 1e-2**, **s = 1, 2, 3**, **seeds 42 / 43 / 44** |
| **Metric** | Validation soft-voted 26-class macro-F1 on half B, one pass per arm |
| **Not in scope** | Other sigma0; LOO; k > 0; the test split; the sealed OOD set; any trained gate |

Nothing else is free. The two thresholds' rate (0.5 %) is the only number that differs from G-TF's 1 %; see § 8, limit 2.

## 4. Decision rules

n = 3 discipline (`CLAUDE.md`): the sign rule is primary; paired t over seeds with Holm over the three s-levels (within
each corruption type) is secondary; a trend is not upgraded.

1. **Primary (stuck).** G-TF2 is **useful for stuck sensors** if its macro-F1 is higher than R5 with no gate in
   **all 3 seeds at each of s = 1, 2, 3** (9 of 9 seed-by-s pairs). If it holds at only some s, report which.
2. **Secondary.** The same rule, separately, for **bias ±2 %**, **bias ±5 %** and **swap**. Gaussian 0.02 p.u. is reported
   only (its loss is 0.007 to 0.020 and `rec` is mostly undefined in G-TF).
3. **Reported for every condition** (type × s): the **corruption loss** F1(clean half B, no gate) − F1(corrupted, no gate);
   **`rec`** = (G-TF2 − no gate) / (oracle − no gate), **interpreted only if the denominator is ≥ 0.01**, otherwise reported
   as undefined and not averaged, with the denominator printed beside it; the **TPR** of corrupted sensors; the **realized
   false-positive rate** on the clean sensors of corrupted half-B windows **for detector 1, for detector 2 and for the
   union**; flagged sensors per window.
4. **Clean-window arm.** F1(G-TF2) − F1(no gate), and the realized false-positive rate of each detector and of the union on
   clean half-B windows. No pass or fail threshold.
5. **Comparison with G-TF** is descriptive: the same windows, model and corruption draw, so the table in § 7 puts G-TF
   beside G-TF2. It is not a hypothesis test and no rule is attached to it.

## 5. Consistency checks

Both G-TF checks are kept: (1) the gate drop equals the training dropout bit for bit; (2) the stored S5.1 detector
scores equal the recomputed scores, and the regenerated corrupted-sensor labels equal the stored ones. The scorer
runs both and stops on a failure. Two further items are **reported, not stop conditions**:

3. **Expected:** detector-2 TPR on stuck sensors = 1.000 at every s (a stuck sensor has std exactly 0, below any threshold
   from clean windows). A lower value is reported as a failed expectation.
4. **To do when the result exists (by hand, not in the scorer):** the no-gate and oracle F1 should equal the G-TF core values
   (`results/stage_f/gtf/core/`) because the model, windows and corruption draw are the same. Report the largest absolute
   difference. A difference is a nondeterminism finding, not a reason to drop the run.

## 6. Procedure and code

`scripts/run_gtf.py --gate gtf2` (stages `thresholds`, `score --seed S`, `summarize`, `check-ready`); output under
`results/stage_f/gtf2/core/`. `--gate gtf` (the default) is G-TF and is unchanged. The scorer refuses to score
unless this file and `07_STAGE_F_gate.md` are committed and unmodified (`check_ready_gtf2`). **G-TF2 is not in the
running queue and no queue was changed.** The code was unit-tested (`tests/test_gtf2.py`, CPU). It has **not** been run on
the real data, so the first thresholds stage and the first scoring are also its first end-to-end run.

## 7. Cost

| Item | Value |
|---|---|
| GPU passes | per seed 5 types × 3 s × 3 arms + 2 clean = 47; × 3 seeds = **141** |
| GPU time, measured rate | G-TF core scored the same 141 passes in 9,954 / 9,965 / 10,006 s per seed = 29,926 s = **8.3 GPU-h** |
| GPU time, upper bound | 223.4 s per half-B pass (from the R5 scoring rate) × 141 = 31,499 s = **8.75 GPU-h** |
| Flatline score | CPU, a standard deviation over the window: no measurable change in the totals |
| CPU, threshold pass | one pass over the 248,000 windows; G-TF measured **7.5 min** (2026-10-25 10:15:56 to 10:23:28 Z) |
| Training | **none** |
| Compared with the plan it replaces | G0 to G3: 198.4 GPU-h (floor; `stage_f_training_free_gate_draft.md` § 9) |

One T1000, no parallel runs. The queue holds the GPU now (R5-gamma0 training from 2026-10-25T18:45Z). G-TF2 takes a
GPU slot only when the author says so.

### The G-TF result, for comparison

From `results/stage_f/gtf/core/` (seeds 42 / 43 / 44, sigma0 = 1e-2, half B; in-sample detector, 1 % per sensor). Means over the three
seeds. **The G-TF2 columns are empty: no G-TF2 result exists.**

| Corruption | s | Corruption loss (mean) | G-TF − no gate (mean; seeds > 0) | G-TF `rec` (mean) | Smallest denominator | G-TF TPR | G-TF realized FPR, clean sensors of corrupted windows | G-TF2 − no gate | G-TF2 `rec` | G-TF2 TPR (det. 1 / det. 2 / union) | G-TF2 FPR (det. 1 / det. 2 / union) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stuck | 1 | 0.0780 | -0.0027 (1/3 seeds > 0) | -0.04 | 0.0661 | 0.296 | 0.057 | not run | not run | not run | not run |
| stuck | 2 | 0.1678 | -0.0062 (0/3 seeds > 0) | -0.04 | 0.1471 | 0.294 | 0.105 | not run | not run | not run | not run |
| stuck | 3 | 0.2497 | -0.0097 (0/3 seeds > 0) | -0.04 | 0.2272 | 0.291 | 0.142 | not run | not run | not run | not run |
| bias ±2 % | 1 | 0.0150 | +0.0027 (3/3 seeds > 0) | undefined | 0.0064 | 0.404 | 0.011 | not run | not run | not run | not run |
| bias ±2 % | 2 | 0.0301 | +0.0078 (3/3 seeds > 0) | 0.43 | 0.0135 | 0.408 | 0.011 | not run | not run | not run | not run |
| bias ±2 % | 3 | 0.0464 | +0.0126 (3/3 seeds > 0) | 0.44 | 0.0230 | 0.411 | 0.012 | not run | not run | not run | not run |
| bias ±5 % | 1 | 0.0392 | +0.0294 (3/3 seeds > 0) | 0.88 | 0.0201 | 0.972 | 0.016 | not run | not run | not run | not run |
| bias ±5 % | 2 | 0.0757 | +0.0575 (3/3 seeds > 0) | 0.91 | 0.0464 | 0.970 | 0.021 | not run | not run | not run | not run |
| bias ±5 % | 3 | 0.1096 | +0.0829 (3/3 seeds > 0) | 0.90 | 0.0741 | 0.968 | 0.028 | not run | not run | not run | not run |
| Gaussian 0.02 p.u. | 1 | 0.0067 | -0.0011 (0/3 seeds > 0) | undefined | -0.0005 | 0.378 | 0.011 | not run | not run | not run | not run |
| Gaussian 0.02 p.u. | 2 | 0.0130 | -0.0004 (1/3 seeds > 0) | undefined | -0.0016 | 0.380 | 0.011 | not run | not run | not run | not run |
| Gaussian 0.02 p.u. | 3 | 0.0201 | +0.0006 (2/3 seeds > 0) | undefined | -0.0025 | 0.382 | 0.012 | not run | not run | not run | not run |
| swap | 1 | 0.1867 | +0.0797 (3/3 seeds > 0) | 0.45 | 0.1701 | 0.997 | 0.121 | not run | not run | not run | not run |
| swap | 2 | 0.3335 | +0.1167 (3/3 seeds > 0) | 0.37 | 0.3063 | 0.997 | 0.236 | not run | not run | not run | not run |
| swap | 3 | 0.4341 | +0.1236 (3/3 seeds > 0) | 0.30 | 0.3986 | 0.998 | 0.337 | not run | not run | not run | not run |

Clean-window arm, G-TF: F1(G-TF) − F1(no gate) = −0.0010 / −0.0020 / −0.0015 (seeds 42 / 43 / 44); realized false-positive rate 1.01 %.
G-TF2: not run.

## 8. Limits stated now

1. One fixed corruption and noise draw (as S5.1 and G-TF). n = 3 is over model seeds.
2. **G-TF2 differs from G-TF in two ways, not one:** the second detector, and the residual threshold at 0.5 % instead of 1 %.
   The scope does not include a "residual only at 0.5 %" arm, so the F1 effect of the two changes cannot be separated. The
   per-detector true-positive and false-positive rates are reported, which show what each detector contributes to the
   flags. An extra arm would add 15 passes per seed (about 2.8 GPU-h); it needs the author's approval.
3. The stuck sensor in the simulation is exactly flat (§ 2, fact 2). The result is about that corruption.
4. The detector was chosen on half B (§ 2, fact 1).
5. sigma0 = 1e-2 only, k = 0 only, half B only, validation split only.
6. Swap contaminates neighbouring sensors' residuals (G-TF open issue 2). Detector 1 keeps this behaviour; the realized
   false-positive rate on the clean sensors is reported.
7. The code has not run on the real data (§ 6).

## 9. What this document does not change

`07_STAGE_F_gate.md` § G-TF and its result; `stage_f_training_free_gate_draft.md`; `14_PREREGISTRATION.md` (the author's own
pre-registration; G-TF2 is a Stage F arm and is not added to it); the S4.4 queue; any frozen number.
