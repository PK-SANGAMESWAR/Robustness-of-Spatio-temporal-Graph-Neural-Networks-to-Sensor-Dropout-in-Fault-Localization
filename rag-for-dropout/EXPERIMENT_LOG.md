# EXPERIMENT_LOG.md — GridSense track

One entry per stage and per run (`rag-eval.md` §11). Newest last. Fields: date, label, git commit (+ dirty flag),
config change, result (with source file), cost, decision. Do not edit an old entry; add a correction entry.
When this file reaches about 180 lines, move the old entries to `EXPERIMENT_LOG_<yyyy-mm>.md` and link it here.

## E0 — 2026-10-28 — S0 support files and scaffold
- Commit: `4caa47c` (dirty; the GridSense files are untracked, O10).
- Change: wrote the track docs (`claude-rag.md`, `dropout-rag-intro.md`, `PLAN.md`, `IMPLEMENTATION.md`, `EVAL.md`,
  `DONE.md`, `OPEN_ITEMS.md`, `PROPOSED_CHANGES.md`), the verbatim `spec/` split (16 parts), `configs/eval.yaml`
  and `configs/registry.yaml` skeletons, 5 GEval rubric drafts, `data/eval/SCHEMA.md`,
  `reports/stage1_inspection.md`, 6 `rag-*` skills, `src/gridsense/__init__.py`, `tests/gridsense/test_docs.py`;
  a deny rule for the frozen sources in `.claude/settings.json`; a 2-line import in the root `CLAUDE.md`.
- Result: `uv run pytest tests/gridsense -q` → 47 passed. No model run, no dependency change, no API call.
- Cost: 0.
- Findings: P1 (figure 0.846 is the standard model), P2 (k grid), P3 (soft vote ≠ probability), P4 (`val` reused),
  P5 (shared scenarios), P6 (window unit).
- Decision needed: author review of S0; answers to O2–O5, O9–O11 to start S2.

## E1 — 2026-10-28 — S1 author answers; key check; first commit
- Answers: O2 (vote share + Platt/isotonic on top-1 share, margin, no logit temperature), O2b (cal-A/cal-B by run),
  O3 (report window by the post-fault-sample rule on `val`), O4 (per-node `h`; concat vs mean+max by kNN-vote on
  `val`), O5 (seed 42), O9 (**5 USD total**, author update), O10 (branch `gridsense-eval`), O11 (current paper,
  version tag, numbers must match RESULTS.md). Recorded in `claude-rag.md` § 3, `configs/eval.yaml`, `OPEN_ITEMS.md`.
- Finding: window w has w − 20 post-fault samples; w0020 is fault-labelled with none (already in
  `models/common.py:unlabelable_window_mask`). So the report index is ≥ 21.
- Paper version check (counts only): `resubmit/1170_revised.tex` (worktree, blob `ee097f8`), `paper/overleaf.tex`
  on main (blob `88ce732`) and the submitted version all contain 0.839 / 0.827 / 0.815 / 0.774 / 0.674 / 0.913 /
  80.3. Which file is current: O11b. A full number test runs at ingestion (S3).
- LangSmith key in `rag-for-dropout/.env`: valid (HTTP 200, US endpoint; EU 403). Value never printed. `.env` is
  Git-ignored (`rag-for-dropout/.gitignore`); `.env.example` added. Note: the line uses `KEY = "value"` (spaces);
  python-dotenv reads it, shell `source` does not.
- New open items: O11b (paper file), O15 (RAM for the 1,600-d concatenation), O16 (OpenAI key).
- Cost: 0. Commit: first commit on branch `gridsense-eval` (no checkout; `main` untouched).
- Decision needed: approval to start S2 (add the `rag` dependency group; CPU latent extraction).

## E2 — 2026-10-28 — S2 started: deps, k = 0 extraction, identity gate FAILED (stopped)
- Answers O11b, O12, O15 and S2 approval recorded. `corpus/v1/` = frozen copies of `1170_revised.tex` (sha256
  `4e34b95f…`) and `RESULTS.md`; `tests/gridsense/test_corpus.py` checks hashes and the 5 curve values.
- Dependencies: `rag` group (15 direct libraries, 105 new locked packages, **no** existing pin changed; torch
  2.5.1+cu121, numpy 2.5.1 kept). Pinned list: `requirements.txt` (uv export). Installed with `--inexact`.
  Telemetry (DeepEval, Chroma) off in `gridsense.settings`; an empty `.deepeval/` made by the first import was removed.
- Extraction (CPU, nice 19, 4 threads, batch 32, paper loader), seed 42, k = 0: test 247,000 (98 s), val 248,000
  (98 s), train 10 % run subsample 115,520 windows = 2,888 runs (45 s). First attempt stored float16 latents;
  `h` exceeds 65,504 (inf) → discarded, re-run in float32. Peak RSS 2.4 GB.
- **Identity gate: not identical.** Test: 6 / 247,000 window predictions differ from `predictions_test.csv`
  (`reports/s2/identity_k0_seed42.json`); macro-F1 0.8515212 vs 0.8515282. Val macro-F1 0.85549072 vs 0.85549097.
  All 6 are near-ties: vote gap 1.9e-6 to 1.8e-3 (votes sum to 25); 4 of 6 are w0020 (excluded from Layer 0).
  Cause: CPU float32 vs the paper's CUDA run. Labels match 247,000 / 247,000.
- Not run (stopped by rule): report window, latent comparison, calibration (`gridsense.eval.s2_k0` is written), k > 0.
- Cost: 0. Decision needed: how to treat near-tie differences (author).

## E3 — 2026-10-28 — S2 closed (seed 42, CPU): identity gate under the author tolerance, report window, latent method, calibration, k > 0
- Commit: see `git log gridsense-eval` (this entry is in the S2 commit). Cost: 0. Full results: `reports/s2/S2_REPORT.md`.
- Identity gate (option 1): k = 0 test 6 of 247,000 window predictions differ, all with vote gap < 2e-3 (max 1.835e-3);
  macro-F1 diff −7.04e-6 (test), −2.6e-7 (val). Rule set after the data was seen (limitation 7). Re-run byte-identical.
- Masks: 480 of 480 mask-generator calls identical to the paper's `evaluate_dropout_curve`; F1 identical in 12 of 12 cells.
  (My first check version forced `labels=range(26)` on a 1,280-window subset and showed a false F1 gap; fixed to the paper's call.)
- k > 0 vs stored curve (test, 3 offsets): k = 1, 2, 5 within 1e-3; **k = 10 off by 1.39e-3 (tolerance 1e-3 not met)**.
  Batch-size diagnostic at k = 10: 8.0e-4 spread. Open: O17.
- Report window: w = 25 (val acc 0.9061; max 0.9097 at w = 32); w = 25 at every k. Latent: (b) mean + max 128-d
  (kNN-vote 0.9635 at k_nn = 10; (a) 0.9627, (c) 0.9629; tie within CI). Calibrator: isotonic (cal-B ECE 0.0022; Platt 0.0145;
  raw 0.0585; post-fault 0.0071 / 0.0271 / 0.1200).
- S4.4 epoch time: baseline 2,586.4 s; epoch 7 +0.90 %; epoch 8 +2.79 % (limit +10 %). 4 threads kept.
- Extraction: k = 0 test/val/train-10 %; k > 0 test and val (votes for 3 offsets; mean + max latents). Disk: eval/latents about 8 GB (Git-ignored).
- Not done: Layer 0 metrics, case-store index, leakage tests, scenario-held-out control, seeds 43/44 (O18, O19).
- Decision needed: O17, O18, O19.

## E4 — 2026-10-29 — O17 five-offset check and scenario-held-out control (Layer 0 step 1); stop before step 2
- Cost: 0. CPU only, nice 19, 4 threads. Report: `reports/layer0/STEP1_REPORT.md`. The S4.4 queue is scoring R4 seed 43 (no epoch times);
  its CPU rate was 100.6 % without a GridSense job and 100.8 % with one.
- O17 (author method): k = 10, `test`, offsets 3–7, batch 32: mean 0.689934, SD 7.00e-4; stored 0.688959 at z = −1.39 → pass. The
  1.31e-3 SD quoted in E3/S2_REPORT § 2 came from 9 values over three batch sizes and is superseded by this 5-offset estimate.
- Scenario-held-out control (`val`, k = 0, 10 % store, (b) mean + max): post-fault kNN vote at k_nn = 10 0.9252 standard vs 0.8551 held-out
  (−7.01 pp); STGNN 0.8807; held-out − STGNN −2.55 pp (CI −3.16 to −1.98). 73 % of standard top-10 neighbours are same-scenario.
  Agreement gap (STGNN correct − wrong): 0.68 standard, 0.44 held-out.
- Not done (stopped as instructed): Layer 0 metrics with both stores, leakage tests, FAISS case store, O6 thresholds.
- Decision needed: O18 (seeds 43/44), O20 (store version for the gate), go-ahead for step 2.

## E5 — 2026-10-30 — Layer 0 step 2 (`val`, both stores), decision rule, leakage tests, FAISS case store, O6 thresholds; stopped as instructed
- Cost 0; CPU only, nice 19, 4 threads. Report: `reports/layer0/STEP2_REPORT.md`. Rule fixed first: `DECISION_RULE.md`, commit `aa7454b`.
- Decision: kNN agreement KEPT. D = +0.0112, 95 % CI [+0.0086, +0.0138] (mean over k of within-k AUROC differences; 1000 run-bootstrap samples).
  All secondary readings agree. A smoke run (k = 0, 1; 20 resamples) was seen before the full run; the pre-registered blocks were identical on a re-run.
- Post hoc (labelled): agreement adds beyond p + margin (+0.0056, CI [+0.0037, +0.0074]); 3-feature gate coverage 0.7514 / 0.6579 at 95 / 98 %.
- Gate (p + agreement, held-out, per-k thresholds): coverage at 95 %: 0.839, 0.825, 0.803, 0.724, 0.502 for k = 0, 1, 2, 5, 10 (stop rule not triggered; margin 0.15 pp).
  One global pair: 93.3 % accuracy at k = 10. Calibration (k = 0 map) ECE 0.007 at k = 0, 0.048 at k = 10 (post-fault).
- Leakage: runs and seeds disjoint; FAISS = torch (votes identical on 3,000 queries); queries put into the store: k_nn = 1 accuracy 1.0000; leave-run-out 0.9737 (clean 0.9739).
  The k_nn = 10 vote reached 0.9857, not ~1.0 (explained in the report); my own threshold of 0.99 for that was not met.
- S4.4: epoch 14 +1.75 % (limit +10 %). `test` untouched. Not done: conformal candidate sets (O23), seeds 43/44 (O18), gate freeze and `test` (O24).
- Decision needed: O18, O21, O22, O23, O24.

## E6 — 2026-10-30 — Full train store, refit, gate_v1 FROZEN (before any test run)
- Full train store (seed 42): 1,155,000 windows, 28,875 runs, 825 scenarios, fingerprint `e1bafcbf…`; extraction 426 s; build 26 s. Leakage checks: runs and seeds disjoint;
  FAISS = torch (votes identical on 1,000 queries); held-out search 0 own-scenario neighbours; standard store: 83 % of the top-20 are same-scenario.
- Full-store neighbour search on val (post-fault, offset 0): 117,800 queries per k; about 11–16 min per k.
- `gate_fit`: isotonic per k, conformal 90 % per k, logistic combination (reported only) on cal-A; thresholds per k on cal-B. `gate_v1.json` sha256 `693ff351157dc70ee748657b92c036d1b5e0ab672aeed1fdcab099457ac8b497`.
- Finding before freezing: at the 95 % point the selected tau_k is 0.0 at every k and tau_p is 0.0 at k = 0, 1, 5, 10 (tau_p 0.5424 at k = 2): the gate is effectively margin-only there;
  at the 98 % point agreement is active. Procedure not changed after seeing this. Conformal sets cover 56–81 % of the LOW_CONFIDENCE windows although 90 % overall.
- S4.4: epoch 17 (nearly continuous neighbour search) +8.2 % (limit +10 %); thread count now read from `eval/threads.txt` (default 4).
- Next (chain, one job at a time): seed-42 test neighbours → ONE frozen run on test → seeds 43 and 44.
- 2026-10-30 12:22 IST: S4.4 epoch 18 = 2,790.7 s (+9.4 %; limit 2,807 s; baseline 2,551.8 s) with the neighbour search running almost continuously. Precaution: `eval/threads.txt` = 2 for every job started from now (author rule: 2 threads above +10 %).

## E7 — 2026-10-30 — the ONE run of frozen gate_v1 on `test` (seed 42); no change to the gate afterwards
- Run 2026-10-30 07:43:41 UTC (`gate_test.py`, config sha256 `693ff351…`, committed in `b7a80b9` first). 117,325 post-fault windows, 6,175 runs. Result file = lock: `reports/layer0/test_run_seed42.json`.
- Coverage at the 95 % point (CI) k = 0, 1, 2, 5, 10: 0.8449 (0.838–0.852), 0.8247, 0.8008, 0.7310, **0.5126 (0.504–0.521)** — not below 50 % at k = 10. cal-B had 0.850, 0.830, 0.811, 0.740, 0.520.
- CONFIDENT accuracy: 0.9493, 0.9487, 0.9502, 0.9491, 0.9488 (every CI contains 0.95; four point estimates are 0.0–0.13 points below). 98 % point: coverage 0.772, 0.745, 0.719, 0.628, 0.376; accuracy 0.9782, 0.9786, 0.9799, 0.9799, 0.9804.
- Conformal 90 %: coverage 0.8986, 0.8980, 0.8979, 0.8971, 0.9003 (CIs contain 0.90); mean size 1.05–2.37; among LOW_CONFIDENCE windows coverage only 0.547, 0.575, 0.592, 0.671, 0.807.
- Per-k isotonic calibration holds on test: ECE 0.0084, 0.0056, 0.0064, 0.0086, 0.0057. Held-out kNN vote 0.831 (k = 0) vs STGNN 0.876; standard store 0.958 (near-duplicates).
- Seeds 43 and 44 are running in the chain (`eval/chain_after_freeze.sh`), 2 threads (`eval/threads.txt`).

## E8 — 2026-10-30 — Seeds 43 and 44 (Layer 0, frozen thresholds); Layer 0 closed
- Frozen method per seed: full train store of that seed (leakage assertions at build), post-fault test windows, held-out neighbours; thresholds from gate_v1 unchanged; isotonic map per k and conformal quantile per k refitted on that seed's cal-A. Chain: `eval/chain_after_freeze.sh`, one job at a time, 2 threads (neighbour search 11–16 min per k).
- CONFIDENT accuracy at the 95 % point: seed 43 0.918–0.932 (k = 0: 0.9234), seed 44 0.897–0.922 (k = 10: 0.8971); seed 42 0.949–0.950. The frozen thresholds do not transfer. Coverage at k = 10: 0.5106 (seed 43), **0.4966 (seed 44, below 50 %)**.
- Conformal 90 % after refit: coverage 0.896–0.902 at every k on both seeds; LOW_CONFIDENCE coverage 0.51–0.83. Per-k isotonic ECE 0.006–0.012.
- S4.4 epochs while the chain ran: 17: +8.2 %, 18: +9.4 %, 19: +6.9 %, then +2.5 to +3.9 % (limit +10 %, never exceeded). Threads 4 → 2 from the end of epoch 18.
- Final report: `reports/layer0/FINAL_REPORT.md` (generated by `final_report.py`). Next stage: Layer 1 (document ingestion and retrieval).

## E9 — 2026-10-31 — Candidate sets for LOW_CONFIDENCE windows frozen (O26); O25, O27, O28 recorded
- `configs/candidate_sets_v1.json` sha256 `686e7d0edd70f5b8a59b8f33b2dad43e1acb395345ee808abdbe187299454980`; gate_v1 unchanged (`693ff351…`). Quantile per k fitted on the cal-A windows that gate_v1 labels LOW_CONFIDENCE (95 % point); checked on cal-B.
- cal-B (LOW_CONFIDENCE windows only), k = 0, 1, 2, 5, 10: coverage 0.9048, 0.9057, 0.9066, 0.9053, 0.9011; mean set size 3.08, 3.20, 3.45, 3.70, 4.27. The old all-window sets covered 0.56, 0.58, 0.59, 0.68, 0.81 of the same windows with sizes 1.1–3.0.
- O25: seed 42 is the system; deployment rules in `claude-rag.md` § 9 (each new checkpoint needs its own calibration, gate fit and frozen test run). O27: gate_v1 kept, no gate_v2. O28: 2 threads (default now 2).
- The one test run of this component is the second use of test (ledger: `reports/layer0/TEST_USE.md`).
