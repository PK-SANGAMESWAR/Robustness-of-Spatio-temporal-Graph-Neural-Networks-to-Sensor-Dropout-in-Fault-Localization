# 00 — Execution: what may be run, and what must stop first

**Supersedes `CUDA.md`** ("give me the code, I'll run it"), amended by the author on 2026-08-16.
Autonomous execution is now allowed for ordinary work; expensive or destructive runs still stop
and report first.

## May run without asking

- Evaluation and analysis: `run_dropout_study.py`, `run_imputation_baseline.py`,
  `run_targeted_dropout_study.py`, `run_persistent_dropout_study.py`,
  `run_significance_tests.py`, `run_runtime_benchmark.py`.
- All summary/figure builders (`build_*.py`) — they only read results and write assets.
- Short training: smoke tests, single-seed diagnostic probes, `smoke_test_dropout.py`.
- The 50-run SDR chain — it **runs autonomously to completion**; report when done.
- Anything read-only.

## Must STOP and report before

- **The 100-run / 3.30 M-window sweep** (`run_fullscale100_seed42_sweep.py`). It is *parked*.
  Do not launch it without being explicitly asked, in this session, for that run.
- Any full retrain of the 5-model × 3-seed sweep (4.7–12.3 GPU-hours *per run*).
- Anything that would overwrite, move, or delete files under `results/fullscale_final/`,
  `results/dropout_aware/`, or `results/sdr_stgnn/`.
- Regenerating a dataset under `dataset_fullscale_*`.

## How

- Always `uv run python scripts/<x>.py` from the repo root — never a bare `python`.
- Long jobs go in the background; report the outcome rather than blocking.
- Several sweeps are **resumable**: `run_dropout_aware_sweep.py` skips any `(model, seed)` that
  already has a `metrics.json`. Prefer resuming over restarting.
- Report failures with the actual error output. A run that crashed is not a run that "mostly
  worked".

## Environment

- GPU: **NVIDIA T1000 8 GB**. NVML monitoring is broken on this machine; training is unaffected —
  do not treat missing `nvidia-smi` telemetry as a fault.
- Stabilization is applied to **every** training run: `--grad-clip-norm 1.0 --warmup-epochs 3`.
- Frozen sweep config goes in via CLI flags (`--gnn-layers 4 --epochs 30 ...`), not baked into
  the baseline YAML.

## Never delete

`*.prebugfix.{csv,json,png}` and `*.bak.csv` are the pre-bugfix / seed-42-only predecessors,
kept deliberately for comparison. Do not delete them, and **never mix them with current numbers**.
