# G-TF2 on the test split — pre-registration

**Status.** Registered on 2026-11-19 by the git commit that first added this file (adca737; the commit time in git is authoritative), **before any test-split G-TF2 result.** `[Wording fix, same day: the first version gave a clock time written before the commit was made; the design below is unchanged.]` The author approved the draft and the GPU run on 2026-11-19 (answer to the
question "approve the draft pre-registration, commit it, and run the GPU scoring"; this file is the registration). No test-split G-TF2 number has been viewed. Two smoke runs
(2026-11-19, 3 and 2 loader batches, GPU, output in a scratch folder outside `results/`, deleted) only tested that the code runs; their F1 values were not read.
It does not change `stage_f_gtf2_preregistration.md` or `14_PREREGISTRATION.md`. The sealed OOD set is not used.

## 1. Why
T9 and the silent-failure claims rest on validation half B. Detector 2 was chosen after an exploratory test on that same half (`stage_f_gtf2_preregistration.md` § 2, fact 1), so the
stuck-sensor gain is not yet shown on data that took no part in the choice. The test split (6,175 events, 247,000 windows, 0 leakage) is the split for reported numbers.

## 2. Design (one change from the validation design: the split)
| Item | Setting |
|---|---|
| Split | `test`, all 6,175 events, 247,000 windows (772 loader batches of 320). No halves; nothing is refitted |
| Thresholds | **Frozen from validation half A**: `results/stage_f/gtf2/core/thresholds_gtf2_sigma_0.01.npz`. sha256 of thresholds_gtf2_sigma_0.01.npz: f0c4734acb60f570ca324dfa17201684c5292f1366e6e44f3449581b508c295d (the scorer refuses to start if the file differs) |
| Detectors, flag rule, gate action | As `stage_f_gtf2_preregistration.md` § 3: in-sample PSM residual OR temporal flatline score, per-sensor thresholds, a flagged sensor is zeroed (six channels), an unavailable sensor is never flagged |
| Model | R5, `results/s4_4/r5/pcstgnn/seed_{00042,00043,00044}/checkpoint_best.pt`. No training. k = 0 |
| Scope | All 5 corruption types × s = 1, 2, 3, σ₀ = 1e-2, seeds 42 / 43 / 44, arms no gate / G-TF2 / oracle, clean-window arm |
| Noise and corruption draws | The validation functions and seeds (`run_gtf._noisy`, `S1.CORRUPTION_SEED`), keyed by the **test loader's batch index**. The draws are the same for all models and seeds |
| Metric | Test soft-voted 26-class macro-F1 over all 247,000 windows, one pass per arm |
| Not scored | The residual-only arm 3b (descriptive only on validation; dropped on test to save about 25 % of the GPU time); other σ₀; k > 0; LOO; the sealed OOD set; any trained gate |

## 3. Decision rules (copied from `stage_f_gtf2_preregistration.md` § 4, not changed)
Sign rule: G-TF2 above no gate in all 3 seeds at each s (9 of 9 pairs) for **stuck** (primary), then bias ±2 %, bias ±5 % and swap (secondary; paired t over seeds with Holm over s within a type).
Gaussian 0.02 p.u. is reported only. `rec` is read only when its denominator (oracle − no gate) is at least 0.01. The clean-window arm and the comparison with the validation result are descriptive.
n = 3: a pass is a sign result, not a significance claim. If the stuck rule fails on test, report that; the detector and the thresholds are not changed afterwards.

## 4. Checks before scoring (run 2026-11-19, before any test number existed)
- `run_gtf2_test.py selfcheck`: the live scoring path equals the stored S5.1 scores on validation batches 0 and 300 (15 conditions, maximum absolute difference 0.0, labels equal) and
  reproduces the stored G-TF2 clean-window flags of batch 0 exactly.
- Consistency check 1 (gate drop equals the training dropout bit for bit) runs at the start of every scoring run and stops it on failure.

## 5. Procedure, cost, and limits
- `scripts/run_gtf2_test.py score` (GPU, resumable in shards of 20 batches, all three seeds in one pass), then `aggregate`, then `summarize`. Output under `results/stage_f/gtf2_test/`.
- **Cost estimate:** about 85 s per loader batch measured in the smoke run (three seeds, 16 conditions, three arms), × 772 batches ≈ **17 to 19 GPU-hours**. This is more than the validation half-B run per seed because the test split is twice as large and the detector scores are computed live.
- The simulated stuck sensor is exactly flat (std 0), so detector 2 has AUROC 1.000 by construction. A test result does not remove that limit.
- Detector 2 was selected on validation half B; the thresholds are from half A. The test split has taken no part in either choice.
- On the test split the result is shown **beside** the validation figures in T9, labelled by split. It does not replace them.
