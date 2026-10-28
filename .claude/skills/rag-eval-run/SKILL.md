---
name: rag-eval-run
description: Run one GridSense evaluation layer (0 to 6) or the full suite on the real pipeline, deterministic metrics first, then write a JSON snapshot and an EXPERIMENT_LOG.md entry. Stops before any paid or GPU work.
---

# rag-eval-run

Use this skill with a layer list (for example `0`, `1`, `0 1`, `all`) and a label.

1. Read `rag-for-dropout/claude-rag.md`, `rag-for-dropout/EVAL.md` (the sections of the layers), and
   `rag-for-dropout/OPEN_ITEMS.md`. If a key that the layer needs is `null` with `# OPEN` in
   `rag-for-dropout/configs/eval.yaml`, stop and ask. Do not choose the value.
2. Check `rag-for-dropout/DONE.md` § 3: the stage that builds the layer must be complete. If the module is missing,
   write the status NOT RUN with "pending: <module>" and stop. Never use fixture or mock numbers as a result.
3. Check the fingerprints (index and case store) against the config. A mismatch → rebuild (if the stage allows it)
   or stop. Never use a stale index.
4. Cost and compute:
   - Layer 0 and the deterministic part of Layer 1 are free. They run on CPU (`nice -n 19`, at most 6 workers,
     through `pinns.cpu_guard` while a GPU queue runs). Ask first if the job will take more than 30 minutes.
   - Any layer with an LLM call (judge, generator, OpenAI embedding): run the `rag-cost-check` skill first and wait
     for the author's approval. Run the 5-item pilot first.
   - No GPU run without the author's approval.
5. Run the layer through the CLI (`PYTHONPATH=src uv run python -m gridsense eval --layers <list> --label <label>`).
   Use reviewed golden items only. Use `val` for tuning, `test` only for a frozen configuration.
6. Layer 0 first check: macro-F1 at each k must equal `results/dropout_study_aware/degradation_curves.csv`
   (`f1_per_train_seed`). If it does not, stop and report the difference. Do not continue.
7. Record each metric failure as a failure with its cause, never as 0 or 1.
8. Write the snapshot to `rag-for-dropout/eval/snapshots/<utc>_<label>.json` (format: `IMPLEMENTATION.md` § 7) and
   per-item records to `rag-for-dropout/eval/items/`. Do not edit an old snapshot.
9. Append one entry to `rag-for-dropout/EXPERIMENT_LOG.md`: date, label, git commit and dirty flag, config change,
   layers, n items, key metrics with 95 % CIs (run-grouped bootstrap), failures, cost, decision needed.
10. Report to the author in ASD-STE100 style: the numbers with their source file, the failures, and the next decision.
