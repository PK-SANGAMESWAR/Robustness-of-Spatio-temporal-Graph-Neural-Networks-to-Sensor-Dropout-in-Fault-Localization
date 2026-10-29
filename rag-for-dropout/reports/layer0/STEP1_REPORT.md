# Layer 0 step 1 report — O17 and the scenario-held-out control (seed 42, CPU, 2026-10-29)

Written for: the author (decision list at the end). Numbers come from `reports/layer0/*.json`; code in `src/gridsense/eval/`.

## 1. O17 — k = 10 on `test`, 5 mask offsets (3–7), batch 32
Method (author): macro-F1 per offset; mean and sample SD (ddof = 1) over the 5 offsets; pass if the stored value is within 2 SD of the mean.
The tolerance of 1e-3 is unchanged. Offsets 3–7 were chosen so that none repeats the paper's offsets 0–2.
| offset | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| macro-F1 | 0.689270 | 0.689328 | 0.690009 | 0.690996 | 0.690068 |
Mean **0.689934**, SD **7.00e-04**. Stored value (seed 42, k = 10): 0.688959; stored − mean = -9.75e-04; z = **-1.39**.
**Result: pass** (|z| ≤ 2).
Stricter reading, informational only: the stored value is a mean of 3 offsets, so SD/√3 = 4.04e-04 would be the matching yardstick; then z = -2.41, which would not pass.
The 1.31e-3 per-offset SD in `S2_REPORT.md` § 2 came from 9 values over three batch sizes; the 5-offset estimate here (7.0e-4) is the one that follows the author's method.

## 2. Scenario-held-out control (O19 step 1), `val`, k = 0
For each query window, all store entries of the same scenario (fault bus, fault type, resistance) are removed before the kNN search.
Store: the 10 % train-run subsample (115,520 windows; 825 of 825 scenarios present), representation (b) mean + max.
Queries: 241,800 val windows without w0020 (117,800 post-fault, w ≥ 21). Primary result: post-fault; secondary: all windows.
STGNN top-1 on the same windows: **0.8807** post-fault, 0.9419 all.
In the standard store, **73.0 %** of the top-10 neighbours of a post-fault window come from the query's own scenario (35.6 % over all windows).

kNN-vote accuracy (95 % CI, run bootstrap):
| k_nn | standard, post-fault | held-out, post-fault | standard, all | held-out, all |
|---|---|---|---|---|
| 5 | 0.9302 (0.9253–0.9348) | 0.8511 (0.8437–0.8581) | 0.9660 | 0.9274 |
| 10 | 0.9252 (0.9202–0.9296) | 0.8551 (0.8478–0.8620) | 0.9635 | 0.9294 |
| 20 | 0.9212 (0.9164–0.9257) | 0.8588 (0.8517–0.8657) | 0.9616 | 0.9312 |

Paired differences on post-fault windows, in percentage points (95 % CI, run bootstrap):
| k_nn | held-out − standard | standard − STGNN | held-out − STGNN |
|---|---|---|---|
| 5 | -7.91 (-8.42 to -7.44) | +4.95 (+4.36 to +5.52) | -2.96 (-3.56 to -2.37) |
| 10 | -7.01 (-7.47 to -6.56) | +4.45 (+3.86 to +5.01) | -2.55 (-3.16 to -1.98) |
| 20 | -6.24 (-6.69 to -5.82) | +4.05 (+3.49 to +4.60) | -2.18 (-2.76 to -1.61) |

kNN agreement (fraction of the k_nn neighbours whose label equals the STGNN top-1 class), post-fault windows:
| store | k_nn | mean | STGNN correct | STGNN wrong | gap |
|---|---|---|---|---|---|
| standard | 5 | 0.8804 | 0.9646 | 0.2591 | 0.7055 |
| standard | 10 | 0.8802 | 0.9614 | 0.2814 | 0.6800 |
| standard | 20 | 0.8802 | 0.9581 | 0.3052 | 0.6529 |
| heldout | 5 | 0.8695 | 0.9205 | 0.4925 | 0.4281 |
| heldout | 10 | 0.8710 | 0.9231 | 0.4869 | 0.4362 |
| heldout | 20 | 0.8727 | 0.9259 | 0.4795 | 0.4464 |

## 3. What this means
- The standard store's lead over the STGNN (+4.45 pp at k_nn = 10) is a near-duplicate effect: 73 % of the neighbours are from the same scenario.
  With the scenario held out, the kNN vote is **2.55 pp below** the STGNN (CI −3.16 to −1.98). The kNN vote does not beat the STGNN on unseen scenarios.
- Agreement separates correct from wrong STGNN predictions much less with the held-out store (gap 0.44 against 0.68 at k_nn = 10): when the STGNN
  is wrong, held-out neighbours still agree with the wrong class 49 % of the time (standard: 28 %). Whether it adds error-detection signal is for step 2 (AUROC).

## 4. Limits of this control
- Only the store is held out. The encoder was trained on all 825 scenarios, so this is not a scenario-held-out model; the `scenario_heldout_*` checkpoints would be the stronger control (not run).
- The held-out store still holds the same fault bus from the other 32 scenarios (other fault types and resistances), so bus-level neighbours remain available.
- Store = 10 % train subsample (about 3.5 runs per scenario); k = 0; seed 42; `val`. A full train store was not built (O15: ask first).

## 5. Compute
The S4.4 queue is now scoring R4 seed 43 (since 2026-10-29 15:01 IST), so there is no epoch time. Its CPU rate is the signal: 100.6 % without a GridSense job
(10 samples) and 100.8 % with one (13 samples, minimum 100.6 %). No effect; 4 threads kept. Extraction 404 s; control about 80 s.

## 6. Decisions for the author
1. **O18:** run seeds 43 and 44 now (about 34 min each)? You asked me to ask again after this control.
2. **O20:** which store version defines τ_k and the gate? I recommend held-out (conservative; the standard agreement is inflated by near-duplicates). Step 2 reports both either way.
3. Go-ahead for step 2 (Layer 0 metrics with both stores, then the leakage tests and the FAISS case store, then O6 thresholds).
