# RUN-100RUN-SWEEP.md — handoff runbook for the full-scale (100-run / 3.30M) sweep

**Purpose:** this file lets a FRESH Claude session run the 100-run full-scale training
sweep without needing prior context. The user switches to a new session only AFTER the
50-run sweep finishes. Read `CLAUDE.md` (repo map) and `PROGRESS.md` §14 (status) for
full background.

---

## What this is
Retrain all 5 baselines (GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2) on the **100-run /
3.30M-window** corpus (`dataset_fullscale_100`), instead of the 50-run / 1.65M corpus the
first sweep used. **The model architecture is IDENTICAL** — same depth-4, 30 epochs,
inverse-freq weighted CE, grad-clip 1.0 + warmup 3. The ONLY change is the dataset
(more training data) and the output folder. This is a data-scale study, not an
architecture change (change one variable at a time).

- Script: `scripts/run_fullscale100_seed42_sweep.py` (already written, staged, NOT yet run)
- Output: `results/fullscale_final_100/<model>/seed_00042/`
- Seed 42 only. All 5 models train FRESH (no reuse — the 50-run rgsage_max checkpoint is
  a different corpus).
- Resumable + crash-isolated: rerunning skips any model with a `metrics.json` already present.

---

## STEP 0 — Prerequisites (verify ALL before launching)
Run these; do not launch until they pass:

```bash
cd /home/ubuntu/Fault-Localization

# (a) the 50-run sweep must be COMPLETE — expect 5
ls results/fullscale_final/*/seed_00042/metrics.json | wc -l

# (b) NO training process may be running (GPU is a single T1000 — must be free). Expect empty.
pgrep -af "cli train --config"

# (c) the 100-run dataset must be present. Expect ~2,310,000
wc -l < dataset_fullscale_100/splits/train.txt
```
If (a) < 5 or (b) is non-empty → the 50-run sweep is still going; **wait, do not launch**.

---

## STEP 1 — Launch (background, survives session end)
```bash
cd /home/ubuntu/Fault-Localization
nohup uv run python scripts/run_fullscale100_seed42_sweep.py \
    > results/fullscale100_seed42_sweep.log 2>&1 &
echo "launched PID $!"
```
Note the PID. The run is **~4–5 days** (2× the data, all 5 models, ~25 h/model at depth-4;
GRU is faster). Order is `rgatv2 → rgcn → gru → rgsage_mean → rgsage_max` (rgatv2 first to
surface instability early).

---

## STEP 2 — Monitor
```bash
# which model is training + latest epoch
grep '\[run\]' results/fullscale100_seed42_sweep.log | tail -1
for m in rgatv2 rgcn gru rgsage_mean rgsage_max; do
  f=results/fullscale_final_100/$m/seed_00042/metrics_by_epoch.csv
  [ -f "$f" ] && [ ! -f results/fullscale_final_100/$m/seed_00042/metrics.json ] && \
    tail -1 "$f" | awk -F, -v m=$m '{printf "%s ep %s/30 loss=%.3f val_gf1=%.3f\n",m,$1,$2,$4}'
done
# completed models
grep -E '\[ok\]|\[FAILED\]' results/fullscale100_seed42_sweep.log
```

**What is normal (do NOT panic):**
- val-F1 climbs slowly for the first ~3 epochs (LR warmup), especially RGATv2.
- The **danger zone is epochs 10–26**. A late-epoch loss spike / val-F1 collapse can happen
  (rgatv2 did this at ep29–30 in the 50-run sweep) — **best-checkpoint selection absorbs it**,
  so the reported `test_graph.f1` uses the peak-val checkpoint, not the spike. Not fatal.
- Expected rough ranking (from 50-run): rgsage_max > rgcn > rgatv2 ≫ gru. GRU stays low
  (~0.14) by design — it is the spatially-blind baseline, not a bug.

**Real problem signals to flag to the user:** a `[FAILED]` line, a model that never rises
above ~0.1 val-F1 by epoch 15, or no-fault recall dropping well below ~0.99 (detection lost).

---

## STEP 3 — After the sweep completes (all 5 have metrics.json)
1. **Build the benchmark table + figures for 100-run.** The summary/figure scripts currently
   default to the 50-run dir. Point them at the 100-run dir first:
   - `scripts/build_fullscale_final_summary.py` line 24: `BENCHMARK_DIR = "fullscale_final"`
     → `"fullscale_final_100"`; and `SCALES = {... : (BENCHMARK_DIR, 50)}` → `100`.
   - `scripts/build_fullscale_final_figures.py` line 41: `BENCHMARK_DIR = "fullscale_final"`
     → `"fullscale_final_100"`.
   (Or ask Claude to make 100-run copies of these two scripts.) Then run both.
2. **Regenerate the paper's dataset table for 100-run** (real counts):
   ```bash
   uv run python scripts/build_dataset_assets.py \
       --data-root dataset_fullscale_100 --runs-per-scenario 100
   ```
   Then update the 3 prose spots in `paper/overleaf.tex` (~lines 300/317/321) from
   50/1.65M to 100/3.30M.
3. **Report to the user:** the 5-model 100-run table (soft-voted + per-window F1,
   per-resistance, no-fault prec/recall) and how it compares to the 50-run numbers
   (rgsage_max 0.853, rgcn 0.729, rgatv2 0.610 at 50-run). **Then STOP for review** before
   any dropout-study (Step 4) work.

---

## Do NOT
- Do not change the model config/architecture — identical to 50-run by design.
- Do not launch while any `cli train` process is running (single GPU).
- Do not auto-chain into the dropout study — STOP and report after the 100-run table.
- Do not delete/overwrite `results/fullscale_final/` (the 50-run results — the fallback).

## Context pointers
- `CLAUDE.md` — repo map. `PROGRESS.md` §14 — live execution log + 50-run results.
- `NEXT-STEPS.md` — the overall plan. `scripts/run_fullscale100_seed42_sweep.py` — the sweep.
