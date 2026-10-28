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
