# OPEN_ITEMS.md — decisions that block later stages

Written 2026-10-28 (S0). Do not choose any of these values. Stop and ask the author. When the author answers, write
the answer and the date here, copy it to `claude-rag.md` § 3 or `configs/eval.yaml`, and move the item to "Closed".

## Open
| ID | Question | Blocks | Proposal (not approved) |
|---|---|---|---|
| O1 | OpenAI generator, judge and (optional) embedding models: exact snapshot names, price per 1M input/output tokens, with the date | S4 | Cheap generator candidates; judge candidates of a higher tier; prices from the provider page, given by the author |
| O6 | Gate target for τ_p and τ_k (selected on cal-B) | S2 | Accuracy ≥ 95 % on CONFIDENT windows, per k if one threshold does not hold |
| O7 | Extra corpus sources: generated feeder docs, curated notes, illustrative SOP | S3 | Not approved on 2026-10-28. Without an SOP, the control-command trap uses an injected chunk |
| O8 | LangSmith: project name, workspace, approval of the first upload. (Key: in `rag-for-dropout/.env`, valid — HTTP 200 on the US endpoint, 2026-10-28; ignored by Git) | S8 | Project `gridsense-dev`; reviewed datasets only; traces redacted |
| O16 | `OPENAI_API_KEY` (the author adds it when coding starts) | S4 | Same `.env`; `gridsense.settings` reads it |
| O12r | Remaining proposed defaults marked (P): bootstrap 2,000 resamples; ECE 15 equal-mass bins; k_rrf = 60; judge κ bar 0.6. (Report-window rule and cal split: closed as O12) | S2 report, S4 | As listed |
| O13 | Human labels for judge validation: who labels, how many (≈50), which rubric scale | S4 | The author labels 50 reports on the rubric scales in `rubrics/` |
| O14 | Event-level aggregation rule for the informational Layer 0 metric | S2 | Sum of window vote shares over the fault windows of a run, argmax |

| O19 | (approved 2026-10-29; steps 1 and 2 done 2026-10-30, stopped as instructed) Next step after S2: Layer 0 metrics, saved case store (FAISS), leakage assertion and sanity run, scenario-held-out control (uses the `scenario_heldout_*` checkpoints; confirm). The gate target O6 is still open | S2 close | As listed |

| O25 | Which checkpoint is the system (seed 42)? The frozen thresholds do not transfer: on seeds 43 and 44 CONFIDENT accuracy at the 95 % point is 0.897–0.932 (`reports/layer0/FINAL_REPORT.md` § 5). Any other checkpoint needs its own fit and frozen run | Layer 1 | Seed 42 is the system |
| O26 | Candidate sets on LOW_CONFIDENCE reports: the 90 % sets cover only 55–81 % of those windows (seed 42, test). State the measured conditional coverage, fit on LOW_CONFIDENCE windows only, or use a fixed top-k set | report design | State the measured coverage |
| O27 | gate_v1 is margin-only at its 95 % point (agreement and probability off). Keep as frozen, or define gate_v2 (needs its own fit and a test that is not the used `test` numbers) | optional | Keep gate_v1 |
| O28 | Thread count: `eval/threads.txt` = 2 since 2026-10-30 (S4.4 epoch +9.4 % at 4 threads); the neighbour search was not slower at 2 threads. Keep 2 until the S4.4 queue ends | compute | Keep 2 |

## Closed (author answers 2026-10-28 to 2026-10-30)
| ID | Answer |
|---|---|
| O18 | 2026-10-30: run seeds 43 and 44 now with the frozen method, Layer 0 only, no new thresholds. Done per `reports/layer0/FINAL_REPORT.md` |
| O21 | 2026-10-30: three-feature gate (isotonic probability, margin, kNN agreement); combination on cal-A, thresholds on cal-B; the report states the composition was selected on val after the pre-registered test |
| O22 | 2026-10-30: per-k thresholds; isotonic calibration fitted separately for each k; a k or dropout pattern not fitted → REJECT, `UNSUPPORTED_DROPOUT_PATTERN` |
| O23 | 2026-10-30: conformal sets, 90 % coverage, per k, fitted on cal-A, checked on cal-B |
| O24 | 2026-10-30: full train store → refit → freeze (commit, hash) → one run on test; no change after; report coverage at k = 10 if below 50 %, do not stop |
| O20 | 2026-10-30: use the held-out store for the gate and for τ_k; report the standard store as secondary only |
| Rule | 2026-10-30: kNN agreement stays in the gate only if the CI of the AUROC difference (probability + agreement) − (isotonic probability) is above 0 (cal-B, post-fault, held-out, bootstrap by run, 1000). Fixed in `reports/layer0/DECISION_RULE.md` (commit `aa7454b`). Result: CI above 0 → kept (`STEP2_REPORT.md`) |
| O17 | 2026-10-29: tolerance unchanged; k = 10 on `test` with 5 offsets (3–7): mean 0.689934, SD 7.00e-4; stored 0.688959 is z = −1.39 from the mean → **pass** (within 2 SD). Method in limitation 9 (`dropout-rag-intro.md`). `reports/layer0/o17_k10_offsets_seed42.json` |
| O6 | 2026-10-29: select τ_p and τ_k on cal-B; maximum coverage with ≥ 95 % accuracy on CONFIDENT post-fault windows; also report coverage at 98 %; compare one threshold vs per-k thresholds on cal-B; if coverage at 95 % is < 50 %, stop and ask. (Store version: O20) |
| O11b | Use `resubmit/1170_revised.tex`; copied to `corpus/v1/` with its SHA-256 in `corpus/v1/MANIFEST.json` (`RESULTS.md` copied too). `tests/gridsense/test_corpus.py` checks hashes and curve numbers |
| O12 | Report window = the earliest w ≥ 21 whose `val` accuracy is within 1 percentage point of the maximum `val` accuracy over w ≥ 21 (applied at k = 0; per-k curves reported). cal-A/cal-B = 50/50 by run, stratified by scenario. `unlabelable_window_mask` (drops w0020) in all Layer 0 metrics |
| O15 | 10 % subsample of train runs: whole runs, stratified by scenario (a scenario fixes bus, fault type and R_f, so fault type is stratified too); one subsample for all methods. Compare (a) concatenation (1,600-d), (b) mean + max pooling (128-d), (c) concatenation + PCA to 256-d fitted on the subsample. Ask before building a full store |
| S2-id | Identity gate option 1 (author, 2026-10-28): CPU predictions are the pipeline predictions; differ only where the vote gap < 2e-3 (rule set after seeing the data; record the 6 window IDs, max gap 1.8e-3 and the macro-F1 difference in the limitations); byte-identical determinism re-run; k > 0 masks use the paper's seeded generator and macro-F1 within 1e-3; PCA arm standardized, fitted on the train subsample only; CPU only, nice 19, 4 threads, 2 if the S4.4 epoch time rises > 10 %; order: report window, latent comparison, calibration, k > 0; commit S2 to `gridsense-eval`; stop after S2 |
| S2 | Approved 2026-10-28: `rag` optional dependency group; `nice -n 19`, `torch.set_num_threads(4)`; after extraction the predictions must equal the paper predictions, else stop and report |
| O2 | Vote share = soft vote / 25. Fit a monotone map on the top-1 share: Platt and isotonic, compare both. No temperature on node logits; predictions stay identical to the paper. Also record the top-1 − top-2 margin |
| O2b | Split `val` by run into cal-A and cal-B. Fit calibration on cal-A; select thresholds on cal-B. State the checkpoint-selection bias in the limitations |
| O3 | Report window = a rule on post-fault samples. Measure accuracy per window index on `val`; use the earliest window where accuracy stops rising. Never use `test` for this choice. (Window w has w − 20 post-fault samples; w0020 has 0, so w ≥ 21) |
| O4 | Per-node embeddings before the head (`h` [25, 64]). Compare by kNN-vote accuracy on `val`: (a) concatenation in fixed node order (1,600-d), (b) mean and max pooling (128-d). Use the better method. RAM: O15 |
| O5 | The system uses seed 42. Repeat Layer 0 for seeds 43 and 44 if the CPU time is below 30 min for each |
| O9 | Cost limit: **5 USD in total for the track** (author update 2026-10-28; replaces "5 USD per run, 20 USD total"). Stop and ask before more |
| O10 | Commit to a new branch `gridsense-eval`. Never commit to `main` |
| O11 | Use the current paper version with a version tag. Its numbers must match `results/RESULTS.md` for the checkpoint in use (a test checks it). Add the camera-ready version later as a new corpus version. Which file: O11b |
