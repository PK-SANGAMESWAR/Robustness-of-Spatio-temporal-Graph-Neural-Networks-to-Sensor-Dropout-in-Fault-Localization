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
