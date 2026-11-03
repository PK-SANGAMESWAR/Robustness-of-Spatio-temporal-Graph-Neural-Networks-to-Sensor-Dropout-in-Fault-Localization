# OPEN_ITEMS.md — decisions that block later stages

Written 2026-10-28 (S0). Do not choose any of these values. Stop and ask the author. When the author answers, write
the answer and the date here, copy it to `claude-rag.md` § 3 or `configs/eval.yaml`, and move the item to "Closed".

## Open
| ID | Question | Blocks | Proposal (not approved) |
|---|---|---|---|
| O1 | (2026-11-02: generator gpt-6-luna and judge gpt-6.1-terra with prices given by the author and confirmed on the pages: closed in `configs/eval.yaml`; the embedding model is closed too: no paid embedder again) OpenAI generator, judge and (optional) embedding models: exact snapshot names, price per 1M input/output tokens, with the date | S4 | Cheap generator candidates; judge candidates of a higher tier; prices from the provider page, given by the author |
| O6 | Gate target for τ_p and τ_k (selected on cal-B) | S2 | Accuracy ≥ 95 % on CONFIDENT windows, per k if one threshold does not hold |
| O8 | LangSmith: project name, workspace, approval of the first upload. (Key: in `rag-for-dropout/.env`, valid — HTTP 200 on the US endpoint, 2026-10-28; ignored by Git) | S8 | Project `gridsense-dev`; reviewed datasets only; traces redacted |
| O16 | `OPENAI_API_KEY` (the author adds it when coding starts) | S4 | Same `.env`; `gridsense.settings` reads it |
| O12r | Remaining proposed defaults marked (P): bootstrap 2,000 resamples; ECE 15 equal-mass bins; k_rrf = 60; judge κ bar 0.6. (Report-window rule and cal split: closed as O12) | S2 report, S4 | As listed |
| O13 | Human labels for judge validation: who labels, how many (≈50), which rubric scale | S4 | The author labels 50 reports on the rubric scales in `rubrics/` |
| O14 | Event-level aggregation rule for the informational Layer 0 metric | S2 | Sum of window vote shares over the fault windows of a run, argmax |

| O19 | (approved 2026-10-29; steps 1 and 2 done 2026-10-30, stopped as instructed) Next step after S2: Layer 0 metrics, saved case store (FAISS), leakage assertion and sanity run, scenario-held-out control (uses the `scenario_heldout_*` checkpoints; confirm). The gate target O6 is still open | S2 close | As listed |


| O29 | **SUPERSEDED 2026-10-31 by O30** (session B wording, kept). Review the 30 draft notes (`corpus/v2/notes/`, each with a `source_ref` to verify) and the 54 draft golden queries (`data/eval/golden_retrieval.jsonl`); until then every Layer 1 number is PRELIMINARY and none may be reported. `rag-golden-review` prepares the table | before any reported Layer 1 number | Table of items with accept / edit / reject |
| O36 | Golden events: 60 drafted; fixed statements approved 2026-11-03 (180 copies reviewed); S4 selection of 30 approved. Open: review the 67 judgement points, the 33 of the S4 events first (`reports/events/judgement_points_s4.csv`), and the golden passages | any S4 run | Review |
| O38 | Approve the five GEval rubrics (`rubrics/*.md`, status draft) | the S4 judge run | Approve or edit |
| O39 | R-0055, R-0056 accepted; R-0059 (trap, expected "insufficient evidence") and R-0060 (new wording) accepted 2026-11-02 (golden v4). Open: review the 10 other near-miss queries R-0057, 58, 61 to 68 (draft; `reports/layer1/NEAR_MISS_CHECK.md`) | their use in a reported result | Accept, edit or reject each |
| O40 | Label the 162 passages of `reports/layer1/labelling_table.csv` (relevant, partly relevant, not relevant) | graded precision@3 and nDCG@3 | Fill the `label` column |
| O42 | Read the illustrative SOP (`corpus/sop_v1/`, draft) | reviewed status | Accept or edit |

## Closed (author answers 2026-10-28 to 2026-10-31)
| ID | Answer |
|---|---|
| O41 | 2026-11-02: keep v2 (it passed the rule fixed before the run). v1 or v2 is not chosen with test results. The test difference (Q −0.036, interval −0.106 to +0.014) and the R-0036 cause are a known limitation (`reports/sweeps/SUMMARY.md` § 2). A diversity limit for row chunks is future work, to be selected on dev only |
| O37 | 2026-11-02: manuscript chunks (`paper1_rev1170`) are allowed in S4 prompts. The corpus is unchanged |
| O35 | 2026-11-02: generator temperature 0. If `gpt-6-luna` does not accept a temperature, no value is sent and the model default is recorded in the snapshot |
| O34 | 2026-11-02: the 11 notes accepted (status reviewed); the SOP stays draft until the author has read it (O42). R-0055 and R-0056 stay draft (O39) |
| O30 | 2026-11-02: review applied; record `reports/layer1/O30_REVIEW_RECORD.md` (golden set v1, 19 notes reviewed, feeder documents accepted) |
| O31 | 2026-11-01: approved. Run the OpenAI embedding (`text-embedding-3-small`, 0.02 USD / 1M input, about 0.0013 USD) after the O30 review, together with the sweeps. The cost ledger and the 5 USD limit apply |
| O32 | 2026-11-01: proposed paper text and one limitation sentence are in `reports/findings/paper1_open_switch.md` § 9 (no paper file changed; the author edits the paper). The thesis future work has the measurement (retrain with the corrected graph; needs the GPU after the S4.4 queue; not started) |
| O33 | 2026-11-01: approved. `_quarantine_session2/` archived outside Git and deleted (`DONE.md` § 5) |
| O7 | 2026-10-31: corpus = generated feeder docs; curated notes (about 30, `status: draft` until the author reviews them); one illustrative SOP; Paper 1 and RESULTS.md as `model_card`. Local embedding models only |
| O25 | 2026-10-31: seed 42 is the system checkpoint. Deployment rule: each new checkpoint needs its own calibration, gate fit and frozen test run (`claude-rag.md` § 9) |
| O26 | 2026-10-31: conformal quantile fitted on LOW_CONFIDENCE windows only, per k, on cal-A; checked on cal-B; run once on test (second use of test, this component only; `reports/layer0/TEST_USE.md`); gate_v1 unchanged |
| O27 | 2026-10-31: no gate_v2; keep gate_v1. The report states that at the 95 % target only the margin is used and kNN agreement has an effect only at the 98 % target |
| O28 | 2026-10-31: 2 threads from now on (`eval/threads.txt`, default 2) |
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
