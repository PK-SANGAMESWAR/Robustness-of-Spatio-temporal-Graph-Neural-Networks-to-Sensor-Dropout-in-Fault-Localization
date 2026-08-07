# Robustness of Spatio-temporal GNNs to Sensor Dropout in Fault Localization

Spatio-temporal graph neural networks (STGNNs) for **fault localization** on the
IEEE-123 distribution feeder under **partial observability** (25 sparse μPMU
voltage sensors), plus a study of their **robustness to μPMU sensor dropout** —
degradation curves under missing sensors and dropout-aware training that recovers
accuracy.

This repository reproduces the STGNN baselines of Karabulut et al. (2026),
*"Robustness of Spatio-temporal Graph Neural Networks for Fault Location in
Partially Observable Distribution Grids"*, and adds the sensor-dropout robustness
contribution on top of the existing models (not a new architecture).

## Task

Locate **where** a fault occurs on the IEEE-123 bus feeder from short windows of
μPMU voltage measurements. 26 classes = 25 candidate fault locations + 1 no-fault.
Data is simulated in OpenDSS via OpenDSSDirect.py: 25 fault locations, 11 fault
types, 3 fault resistances (0.1 / 1 / 10 Ω), load scaling U(0.5, 1.3), 1 ms
sampling, 20 ms sliding windows. Node features are `F=6` per timestep —
`[V1,V2,V3]` voltages + `[m1,m2,m3]` availability mask.

## Models

All baselines are spatio-temporal (temporal GRU encoder + a spatial GNN):
**GRU** (no spatial layer), **RGCN**, **RGSAGE-Mean**, **RGSAGE-Max**, **RGATv2**.

## Contribution: sensor-dropout robustness

- **4a — degradation curves:** evaluate each trained model with `k ∈ {0,1,2,5,10}`
  μPMUs dropped at inference. Dropout = feature masking (voltage + availability
  channels zeroed; graph intact). Masks are **paired** across models — the same
  nodes are dropped for every model at a given `(k, seed)` — for a fair comparison.
- **4b — dropout-aware training:** retrain with random node masking during
  training, then re-run 4a to measure how much accuracy is recovered at each `k`.

## Repository layout

```
fault_localization/     # source package
  data/                 # dataset + IEEE-123 OpenDSS source files (.dss)
  models/               # GRU, RGCN, RGSAGE, RGATv2
  simulation/ graph/    # OpenDSS time-stepped sim + substation-rooted graph
  dropout/              # sensor-dropout masking, evaluator, dropout-aware training
  experiments/          # CLI, config loading, training runner
config/                 # base + per-model YAML configs
scripts/                # dataset generation, sweeps, summary + figure builders
paper/                  # paper drafts
docs/                   # design / methodology notes
tests/                  # unit + regression tests
```

## Setup

Requires **Python 3.12**. The project uses [uv](https://docs.astral.sh/uv/) for
dependency management (authoritative spec: `pyproject.toml` + `uv.lock`):

```bash
uv sync          # create .venv from pyproject.toml + uv.lock
```

Alternatively, with plain pip (versions pinned to the tested environment):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs `torch` from the PyTorch CUDA 12.1 wheel index; for a
CPU-only install, drop the `+cu121` pin and the `--extra-index-url` line. The
`uv run ...` commands below become plain `python ...` inside the activated venv.

## Data

Datasets are **not** committed (multi-GB). Regenerate them locally:

```bash
uv run python scripts/generate_trend_dataset.py --help   # dataset generation options
```

The 70/15/15 event-level split with train-only normalization is produced by the
generation scripts; the IEEE-123 OpenDSS source files ship in
`fault_localization/data/ieee123/`.

## Running the baselines

Frozen config: GNN depth 4, 30 epochs, unweighted cross-entropy,
gradient-clip norm 1.0 + 3-epoch LR warmup (training stabilization), applied to
every run.

```bash
# 5-model seed-42 baseline sweep (crash-isolated + resumable)
uv run python scripts/run_fullscale_final_sweep.py

# aggregate table + figures
uv run python scripts/build_fullscale_final_summary.py
uv run python scripts/build_fullscale_final_figures.py
```

Outputs land under `results/fullscale_final/` (metrics, summary CSVs, figures).

## Running the sensor-dropout study

```bash
# 4a — inference-time degradation curves for the trained models
uv run python scripts/run_dropout_study.py \
    --results-dir results/fullscale_final --data-root <dataset> \
    --output-dir results/dropout_study

# 4b — dropout-aware retraining, then re-evaluate + overlay
uv run python scripts/run_dropout_aware_sweep.py
uv run python scripts/run_dropout_study.py \
    --results-dir results/dropout_aware --data-root <dataset> \
    --output-dir results/dropout_study_aware
uv run python scripts/build_dropout_comparison.py
```

## Tests

```bash
uv run pytest
```

## Status

Active research code (targeting an ICMLDE 2026 submission). The baseline is
reproduced at depth-4. All five baselines are trained at seeds 42/43/44, and
full-observability accuracy is reported as mean ± std over those seeds. The
sensor-dropout degradation curves are seed-averaged for RGSAGE-Max and the two
robustness variants (dropout-aware RGSAGE-Max and SDR-STGNN); the other four
baselines' degradation curves are reported on the seed-42 checkpoint. See
`PROGRESS.md` and `NEXT-STEPS.md` for the current execution state.

## License

MIT — see [LICENSE](LICENSE).

## Reference

Base paper: Karabulut et al. (2026), *Robustness of Spatio-temporal Graph Neural
Networks for Fault Location in Partially Observable Distribution Grids.* The base
paper PDF is not redistributed here.
