# Stage F — G-TF, a training-free gate arm (DRAFT, pre-registration in progress)

**Status.** Decisions a–g were given by the author on 2026-10-24 and are fixed. The detector rule (§ 2 item c)
was applied on 2026-10-25 03:43 IST / 2026-10-24T22:13:54Z, after the sigma0 = 1e-2 S5.2 result was complete: **in-sample** (§ 7). The final design
was copied to `07_STAGE_F_gate.md` as arm "G-TF" on the same date, marked "added after S5.1, before any G-TF result".
No G-TF result exists. No G-TF scoring has been run (a 2-batch CPU smoke test of the scorer on sigma0 = 5e-4 was run for
correctness only; its numbers are not results and are not used).

## 1. Question

Can the S5.1 detector act as a gate with no training? A flagged sensor is treated as dropped. The model is R5,
unchanged (the 3 Gate 4 checkpoints, seeds 42/43/44). G-TF does not replace G0–G3: G2 vs G0 still decides
whether a trained gate earns its complexity. G-TF tells us how much of the silent-failure loss a threshold on
the S5.1 score recovers, and sets a floor that G2 must beat.

## 2. Pre-registered design

| Item | Setting |
|---|---|
| **a. Split** | The 248,000 validation windows are split into halves **A** and **B** by **fault event** (event = window id up to `_w`). Fixed seed `SPLIT_SEED = 7` (`numpy.random.RandomState(7).permutation` over the sorted event ids; first half of the permutation = A). No event is in both halves (asserted). Realized split: 3,100 / 3,100 events, 124,000 / 124,000 windows, 0 shared events; event-list SHA-256 A `333cc10a…54c78`, B `c342fc6b…26d3a`. Windows per class in B / in A: 0.77 to 1.36. Windows per resistance (0.1 / 1 / 10 Ω): A 41,480 / 41,000 / 41,520; B 41,320 / 42,360 / 40,320. |
| **b. Threshold and scoring** | Thresholds are chosen on the **clean** windows of half **A** (noise at the same sigma0, no corruption). All G-TF scoring is on half **B** only. No corrupted window is used for the threshold. |
| **c. Detector rule** | Use the **LOO** residual only if its AUROC is at least **0.01** higher than the in-sample AUROC for bias ±2 % at sigma0 = 1e-2. **Reading, fixed here:** the condition must hold at each of s = 1, 2, 3. Otherwise use the **in-sample** residual. Applied once, by `run_gtf.py decide`, which refuses to run before the sigma0 = 1e-2 result is complete. The result goes to `results/stage_f/gtf/detector_decision.json` and to § 7. |
| **d. Threshold rate** | A **per-sensor** false-positive rate of **1 %**: for each of the 25 sensors, the 99th percentile of its available clean half-A scores (`numpy.quantile`, linear). 25 thresholds per sigma0. The **realized** false-positive rate is reported on (i) clean half-B windows and (ii) the clean sensors of corrupted half-B windows, per condition. |
| **e. Per-type reporting** | Every result is reported per corruption type and per s. No pooling over types. |
| **f. Core scope** | bias ±2 %, sigma0 = 1e-2, s = 1, 2, 3, seeds 42/43/44; arms **no gate**, **G-TF**, **oracle**; plus the **clean-window arm** (no gate vs G-TF on clean half-B windows at sigma0 = 1e-2). |
| **g. Full scope** | 5 corruption types × s = 1, 2, 3 × sigma0 ∈ {0, 5e-4, 2e-3, 1e-2}. **Only after the author approves** (`results/stage_f/gtf/FULL_SCOPE_APPROVED`, created by the author). The scorer and the queue refuse it otherwise. |
| Model | R5, `results/s4_4/r5/pcstgnn/seed_{00042,00043,00044}/checkpoint_best.pt`. No training. k = 0. |
| Gate action | A sensor with score > its threshold is dropped: all six channels set to 0 (voltage and mask), the `zero_messages` operation. R5's fusion then uses the new availability mask. |
| Oracle | R5 with exactly the corrupted sensors dropped. |
| Metric | Validation soft-voted 26-class macro-F1 on half B, one pass per arm (k = 0: no random masks). |
| Corruption and noise | Regenerated with the S5.1 seeds (`CORRUPTION_SEED` 0; noise seed `(1 + idx) × 10000 + batch`). The scorer asserts the regenerated corrupted-sensor labels equal the stored `*__label.npy`, and the batch labels equal the stored `y.npy`. |
| Detector scores | Read from the S5.1 raw scores (`psm_loo` or `psm_ins`) for corrupted windows; computed on the fly for clean windows. |

## 3. Procedure

1. `run_gtf.py split` — writes `split.json` (CPU).
2. `run_gtf.py thresholds` — clean half-A thresholds (both detectors) and clean half-B flags, per sigma0 (CPU,
   one pass over the 248,000 windows).
3. `run_gtf.py score --seed S` — three R5 passes per condition, two for the clean arm (GPU).
4. `run_gtf.py summarize` — decision rule, recovered fraction, paired t, Holm (CPU).

Resumable: a stage skips when its output exists.

## 4. Decision rule

1. **G-TF is useful** if its macro-F1 is higher than R5 with no gate in **all 3 seeds** at bias ±2 %,
   sigma0 = 1e-2. Reading: at each s = 1, 2, 3 (9 of 9 seed-by-s pairs positive). If it holds at only some s, report which.
2. **Recovered fraction**, per seed and per s, then the mean over seeds:
   `rec = (F1[G-TF] − F1[no gate]) / (F1[oracle] − F1[no gate])`.
   Undefined where the denominator is ≤ 0 (reported as "no loss to recover"). The denominator is printed with every `rec`.
   `rec` may exceed 1; the oracle is a reference, not a bound.
3. n = 3 discipline (`CLAUDE.md`): the sign rule is primary; paired t over seeds with Holm over the three
   s-levels is secondary; a trend is not upgraded.
4. **Clean-window arm:** report F1(G-TF) − F1(no gate) on clean half-B windows and the realized false-positive
   rate. This is the cost of false suppression. No pass/fail threshold is set for it.
5. Validation split only. The test split and the sealed OOD set are not used.

## 5. Cost — measured rates

Measured R5 scoring: 26,809 / 26,777 / 26,743 s for 60 passes over 248,000 windows = 446.8 / 446.3 / 445.7 s per
pass. Scoring on half B only (124,000 windows) is half of that: **223.4 s per pass**, using the largest
measured value as an upper bound. The G-TF scorer batches the noise and the corruption, where the Gate 4 scorer
loops per window, so its true rate is probably faster; that is not measured.

| Scope | Passes (3 seeds) | GPU time (upper bound) |
|---|---|---|
| **Core**: per seed 3 s × 3 arms + 2 clean = 11 | 33 | 7,372 s = **2.05 GPU-h** |
| Full: per seed (60 conditions × 3 arms) + (4 sigma0 × 2 clean) = 188 | 564 | 125,998 s = **35.0 GPU-h** |

CPU, no GPU: the threshold stage is one pass over the 248,000 windows per sigma0, about 17 minutes
(extrapolated from the running S5.1 noise job, ≈ 250 min for 15 conditions; not measured for this pass). Core: 1
sigma0 ≈ 17 min. Full: 4 sigma0 ≈ 67 min. The S5.1 raw scores for corrupted windows already exist (sigma0 = 0,
5e-4 done; 2e-3 and 1e-2 running).

GPU constraint: one T1000, no parallel runs. G-TF core is queued **after the first S4.4 queue** (§ 6).
Nothing in the first queue was changed. *State on 2026-10-25 03:45 IST:* the first queue stopped itself on
2026-10-24 16:58Z (R6 seed 42 triggered the epoch-3 collapse guard), so the GPU is idle and part 2 has not started
(the watcher exited on the STOPPED marker). Restarting is the author's decision.

## 6. Queue

The running queue is a live process with its step list in memory; editing its script changes nothing and the
only way to insert a step into it is to stop and restart it, which was not done. So G-TF core, R1 and R2 are
in a second script, `scripts/run_s4_4_queue_part2.py`: **G-TF core → R1 (seed 42) → R2 (seeds 42/43/44)**. It
starts only after the first queue writes `queue_complete` and exits (`scripts/wait_queue_then_part2.sh`,
waiting on the queue's PID, reading files only). The G-TF step runs only if this pre-registration is complete
(`run_gtf.py check-ready`: the arm is in `07_STAGE_F_gate.md` with the marker, the detector decision is
recorded, both documents are committed and unmodified); otherwise it logs `gtf_skipped_not_ready` and the queue
goes on to R1. The full scope is never run by the queue.

## 7. Detector decision — recorded 2026-10-25 03:43 IST / 2026-10-24T22:13:54Z

Rule (item c) applied by `run_gtf.py decide` on the complete sigma0 = 1e-2 S5.1/S5.2 result
(`results/stage_f/silent_detection_noise/sigma_0.01/silent_detection.json`, 248,000 windows, not a partial run).

| s | LOO AUROC | in-sample AUROC | LOO − in-sample | ≥ 0.01 |
|---|---|---|---|---|
| 1 | 0.8537 | 0.8491 | +0.0046 | no |
| 2 | 0.8504 | 0.8464 | +0.0041 | no |
| 3 | 0.8482 | 0.8446 | +0.0036 | no |

**LOO is higher at every s, but by less than 0.01 (+0.0036 to +0.0046), so the rule selects the in-sample
residual.** Decision file: `results/stage_f/gtf/detector_decision.json` (`"detector": "ins"`). For reference, at
sigma0 = 0, 5e-4 and 2e-3 LOO − in-sample at bias ±2 % is −0.0002 to −0.0007. The S5.1 AUROC at bias ±2 %,
sigma0 = 1e-2, is 0.848–0.854 (in-sample 0.845–0.849): a weaker detector than at lower noise, so G-TF at
sigma0 = 1e-2 is a harder test than the S5.1 numbers at lower noise suggest.

## 8. Open issues carried over from the first draft

1. **Single corruption draw.** n = 3 is over model seeds; the corruption and noise draw is one fixed draw (as S5.1).
2. **Contamination.** A corrupted sensor raises the scores of other sensors in its window, so the realized
   false-positive rate on clean sensors of corrupted windows will exceed 1 %. It is reported (item d), not corrected.
3. **Weak types.** Stuck sensors have AUROC 0.59–0.64 and swap at s = 3 flags few sensors; these are full-scope
   conditions and are reported separately (item e).
4. **k = 0 only.** G-TF is not tested with random sensor loss.
5. **Class balance of the halves** is random, not stratified (B/A per class 0.77 to 1.36); macro-F1 on half B is
   not comparable to the whole-validation macro-F1 of Gate 4. G-TF compares its three arms on the same half B only.

## 9. Corrected cost estimate for G0–G3 (`07_STAGE_F_gate.md` § S5.3)

The old text says "8 runs × ≈ 7–9 h ≈ 60–70 GPU-h". The measured rates are higher.

| Arm | Runs | Per run | Basis | Total |
|---|---|---|---|---|
| G0 (no physics, corruption augmentation) | 3 | **10.14 h** = 8.79 h train + 1.34 h score | measured R0ⁿ: train 31,651 / 31,711 / 31,602 s; score 4,844 / 4,849 / 4,815 s | **30.4 GPU-h** |
| G1 (PC-STGNN + augmentation) | 1 | **33.6 h** = 26.16 h train + 7.44 h score | measured R5 (train 94,185 s; score 26,785 s, means of 3 seeds) | **33.6 GPU-h** |
| G2 (PCRG-STGNN) | 3 | 33.6 h | as instructed | **100.8 GPU-h** |
| G3 (gate at every hop) | 1 | 33.6 h | as instructed | **33.6 GPU-h** |
| **Total** | 8 | | | **198.4 GPU-h = 8.27 days** |

2.8 to 3.3 times the old 60–70 GPU-h. G0 uses the R0ⁿ rate (the cost of the corruption augmentation is not
measured). G2 has a second forward pass for the gate that the 33.6 h does not include, and the G0–G3 scoring
grid is not fixed, so 198.4 h is a floor for G2/G3. G-TF (§ 5) is separate from this table.
