# RUN-SDR-50RUN-RUNBOOK.md — fresh-session runbook for the SDR-STGNN contribution + best-model multi-seed baseline (50-run corpus)

**Purpose:** this file lets a FRESH Claude session run the sensor-dropout contribution on the
50-run corpus without prior context. Read `CLAUDE.md` (repo map), `PROGRESS.md` §14 (status), and
`/home/ubuntu/.claude/plans/sparkling-rolling-brook.md` (the SDR-STGNN design plan) for background.

**Locked ordering (user decision):**
1. Train the **new novel model SDR-STGNN** on **one seed (42)**, on the **50-run corpus**
   (`dataset_fullscale_50`) — the same corpus/config every baseline was trained on.
2. Train the **best-performing baseline (RGSAGE-Max, 0.853)** on the **remaining seeds (43, 44)**
   on the 50-run corpus — seed 42 already exists — to give the best baseline **multi-seed error
   bars**. (Only the best model gets multi-seed at 50-run; full multi-seed for all models is done
   at 100-run, later.)
3. This creates the 50-run baseline. **Then, in a separate later session, proceed to the 100-run
   sweep on all models/seeds** (see `RUN-100RUN-SWEEP.md`).

**Single GPU (T1000).** Everything runs sequentially — never launch two `cli train` at once.
Every step is resumable (a run with `metrics.json` present is skipped).

---

## STEP 0 — Prerequisites (verify ALL before launching)
```bash
cd /home/ubuntu/Fault-Localization

# (a) 50-run baselines must exist (expect 5) — provides the standard context curves + rgsage_max seed 42
ls results/fullscale_final/*/seed_00042/metrics.json | wc -l

# (b) GPU must be free — no training running. Expect empty.
pgrep -af "cli train --config"

# (c) 50-run corpus present. Expect ~1,155,000 (825*50*40*0.7 train split)
wc -l < dataset_fullscale_50/splits/train.txt

# (d) SDR-STGNN code present + tests green (fast, CPU)
uv run pytest tests/test_sdr_stgnn.py -q
```
If (a) < 5 or (b) is non-empty → do not launch. Fix first.

---

## STEP 1 — Train SDR-STGNN, seed 42, 50-run  (~6–8 h)
The novel availability-gated model, trained WITH dropout augmentation. The runner auto-emits the
degradation curve (`dropout_curve.json`) after test eval.
```bash
cd /home/ubuntu/Fault-Localization
nohup uv run python scripts/run_sdr_stgnn.py \
    > results/sdr_stgnn_seed42.log 2>&1 &
echo "launched SDR-STGNN PID $!"
```
- Config: `config/sdr_stgnn.yaml` (frozen recipe applied by the script: depth-4, 30 epochs,
  grad-clip 1.0, warmup 3; `data_root=dataset_fullscale_50`, seed 42).
- Output: `results/sdr_stgnn/sdr_stgnn/seed_00042/` (`metrics.json`, `checkpoint_best.pt`,
  `dropout_curve.json`, `metrics_by_epoch.csv`).
- **Do NOT start Step 2 until this finishes** (single GPU). Confirm with:
  ```bash
  grep -E '\[ok\]|\[FAILED\]' results/sdr_stgnn_seed42.log
  ls results/sdr_stgnn/sdr_stgnn/seed_00042/metrics.json   # exists ⇒ done
  ```

**Sanity gate (report before proceeding):** SDR-STGNN k=0 macro-F1 (in `dropout_curve.json`, key
`"0"`) should be ≈ standard RGSAGE-Max k=0 (~0.85, within noise) — this is the designed
equivalence when no sensor is dropped. If it is far off, STOP and report.

---

## STEP 2 — Train RGSAGE-Max on the remaining seeds (43, 44), 50-run  (~12–16 h total)
Best model only, remaining seeds, SAME frozen recipe/corpus/benchmark dir as seed 42 → they land
next to the existing seed 42 under `results/fullscale_final/rgsage_max/`.
```bash
cd /home/ubuntu/Fault-Localization
for SEED in 43 44; do
  uv run python -m fault_localization.experiments.cli train \
      --config config/rgsage_max.yaml \
      --seed "$SEED" \
      --data-root dataset_fullscale_50 \
      --benchmark-version fullscale_final \
      --epochs 30 --grad-clip-norm 1.0 --warmup-epochs 3 --gnn-layers 4
done 2>&1 | tee results/rgsage_max_seeds4344.log
```
(Run under `nohup ... &` if you want it detached.) Outputs:
`results/fullscale_final/rgsage_max/seed_00043/` and `.../seed_00044/`.
After this, RGSAGE-Max has seeds 42/43/44 → report the multi-seed clean-test macro-F1 as
**mean ± std** (the best-baseline error bar for the paper).

---

## STEP 3 (recommended for the full ablation) — Dropout-aware RGSAGE-Max, seed 42  (~6–8 h)
This is the **mask-aware / ungated baseline**: it receives the availability mask (all baselines do
— channels 3–5) and is trained under dropout, but does NOT gate aggregation. The delta from it to
SDR-STGNN isolates the **structural gate** — the paper's key ablation ("the gate, not just the
mask, buys robustness"). Cheap (1 seed); skip only if you deliberately drop the 3-way ablation.
```bash
cd /home/ubuntu/Fault-Localization
uv run python scripts/run_dropout_aware_sweep.py \
    --models rgsage_max --data-root dataset_fullscale_50 \
    2>&1 | tee results/dropout_aware_rgsage_max.log
```
Output: `results/dropout_aware/rgsage_max/seed_00042/`.

---

## STEP 4 — Degradation studies + ALL figures  (inference only, ~15–40 min)
Produces the contribution's result figures. Run AFTER Steps 1–3.
```bash
cd /home/ubuntu/Fault-Localization

# 4a. Standard baselines (5 models, seed 42) — context curves
uv run python scripts/run_dropout_study.py \
    --results-dir results/fullscale_final --data-root dataset_fullscale_50 \
    --output-dir results/dropout_study

# 4b. Dropout-aware RGSAGE-Max (mask-aware, ungated)  [skip if Step 3 skipped]
uv run python scripts/run_dropout_study.py \
    --results-dir results/dropout_aware --models rgsage_max \
    --data-root dataset_fullscale_50 --output-dir results/dropout_study_aware

# 4c. SDR-STGNN (ours)
uv run python scripts/run_dropout_study.py \
    --results-dir results/sdr_stgnn --models sdr_stgnn \
    --data-root dataset_fullscale_50 --output-dir results/dropout_study_sdr

# 4d. Build every result figure + robustness summary CSV
uv run python scripts/build_sdr_result_figures.py \
    --baselines results/dropout_study/degradation_curves.csv \
    --aware     results/dropout_study_aware/degradation_curves.csv \
    --sdr       results/dropout_study_sdr/degradation_curves.csv \
    --out-dir   paper/figures

# 4e. Standard-vs-dropout-aware overlay (with the SDR curve on top)
uv run python scripts/build_dropout_comparison.py \
    --standard      results/dropout_study/degradation_curves.csv \
    --dropout-aware results/dropout_study_aware/degradation_curves.csv \
    --sdr           results/dropout_study_sdr/degradation_curves.csv \
    --output-dir    results/dropout_study
```
**Shortcut:** Steps 1 + 3 + 4 (but NOT the Step-2 extra seeds) can be run in one command via the
orchestrator: `nohup uv run python scripts/run_sdr_contribution.py > results/sdr_contribution.log 2>&1 &`.
Do Step 2 separately (the orchestrator does not train extra baseline seeds).

**Figures produced** (all in `paper/figures/`):
- schematics (already generated, no run needed): `sdr_architecture.png`, `sdr_mechanism.png`, `sdr_protocol.png`
- `sdr_degradation_all.png` — F1 vs k, all baselines + SDR-STGNN
- `sdr_ablation_backbone.png` — 3-way ablation on RGSAGE-Max (the money figure)
- `sdr_retention.png` — % of k=0 F1 retained
- `sdr_audc_bar.png` — normalized AUDC robustness score
- `sdr_heatmap.png` — model × k macro-F1 grid
- `sdr_robustness_summary.csv` — F1@k, retention@k, AUDC per variant

---

## STEP 5 — Refresh the baseline table with multi-seed error bars (optional)
With RGSAGE-Max now at seeds 42/43/44, regenerate the 50-run baseline summary/figures so the best
model shows mean ± std. Scripts: `scripts/build_fullscale_final_summary.py`,
`scripts/build_fullscale_final_figures.py`. **Check** these aggregate across seeds (they were
written for seed 42); if they only read seed 42, report and I (or the next session) will add the
multi-seed aggregation rather than silently reporting a single seed.

---

## STEP 6 (LATER, separate session) — 100-run on all
Once the 50-run contribution is complete and reported, do the full multi-seed 100-run study on all
models. Follow `RUN-100RUN-SWEEP.md`; the SDR-STGNN + dropout-aware runs are re-run there with
`--data-root dataset_fullscale_100` (the model/scripts are corpus-agnostic). **STOP and report the
50-run contribution table + figures before starting 100-run** (don't auto-chain).

---

## Monitoring
```bash
# current epoch of whatever is training
for d in results/sdr_stgnn/sdr_stgnn/seed_00042 \
         results/fullscale_final/rgsage_max/seed_000{43,44} \
         results/dropout_aware/rgsage_max/seed_00042; do
  f="$d/metrics_by_epoch.csv"
  [ -f "$f" ] && [ ! -f "$d/metrics.json" ] && \
    { echo -n "$d  "; tail -1 "$f"; }
done
```
**Normal:** val-F1 climbs slowly for ~3 warmup epochs; a late-epoch (10–26) spike is absorbed by
best-checkpoint selection. **Flag:** any `[FAILED]`, val-F1 stuck < ~0.1 by epoch 15, or SDR-STGNN
k=0 F1 far from the standard RGSAGE-Max k=0.

## Do NOT
- Do not run two `cli train` at once (single GPU).
- Do not change the frozen recipe (depth-4 / 30 epochs / grad-clip 1.0 / warmup 3) — identical to
  every baseline by design (change one variable at a time).
- Do not overwrite `results/fullscale_final/` seed 42 or `results/sdr_stgnn/` once complete.
- Do not auto-chain into the 100-run sweep — STOP and report the 50-run contribution first.

## Context pointers
- `CLAUDE.md` — repo map. `PROGRESS.md` §14 — status + 50-run baseline results.
- `/home/ubuntu/.claude/plans/sparkling-rolling-brook.md` — SDR-STGNN design + review fixes.
- `fault_localization/models/sdr_stgnn.py` — the model. `config/sdr_stgnn.yaml` — its config.
- `scripts/run_sdr_stgnn.py`, `scripts/run_dropout_aware_sweep.py`, `scripts/run_dropout_study.py`,
  `scripts/build_sdr_result_figures.py`, `scripts/build_dropout_comparison.py`,
  `scripts/build_sdr_schematics.py`, `scripts/run_sdr_contribution.py` (orchestrator).
```
