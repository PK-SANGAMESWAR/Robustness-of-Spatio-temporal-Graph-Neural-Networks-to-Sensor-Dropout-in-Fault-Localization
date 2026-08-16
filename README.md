# Robustness of Spatio-temporal GNNs to Sensor Dropout in Fault Localization

Spatio-temporal graph neural networks (STGNNs) for **fault localization** on the
IEEE-123 distribution feeder under **partial observability** (25 sparse μPMU
voltage sensors), plus a study of their **robustness to μPMU sensor dropout** —
degradation curves under missing sensors and dropout-aware training that recovers
accuracy.

This repository **adopts the methodology** of Karabulut et al. (2026),
*"Robustness of Spatio-temporal Graph Neural Networks for Fault Location in
Partially Observable Distribution Grids"*, and reports **our own** numbers on a
corpus we regenerate ourselves — it is *not* a numeric reproduction of that paper.
On top of those baselines it adds the sensor-dropout robustness contribution.

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

- **Degradation curves:** evaluate each trained model with `k ∈ {0,1,2,5,10}`
  μPMUs dropped at inference. Dropout = feature masking (voltage + availability
  channels zeroed; graph intact). Masks are **paired** across models — the same
  nodes are dropped for every model at a given `(k, seed)` — for a fair comparison.
- **Dropout-aware training** (the headline mitigation): retrain with random node
  masking during training, then re-run the sweep to measure recovery at each `k`.
- **SDR-STGNN** (ablation): an availability-gated aggregator in which dropped nodes
  emit no messages. Reported honestly — it *matches* but does **not** beat
  dropout-aware training, at +18 % inference latency.
- **Comparison baselines:** neighbour-mean imputation (impute-then-infer on the
  standard model), **targeted** worst-case dropout (drop the k μPMUs nearest the
  fault), and **persistent** per-event dropout (a failed sensor stays dark for the
  whole event).

### Headline results (RGSAGE-Max backbone, seeds 42/43/44)

| | clean F1 | F1 @ k=10 | retention | AUDC |
|---|---|---|---|---|
| Standard training | 0.846 | 0.048 | 5.7 % | 0.402 |
| + neighbour-mean imputation | 0.846 | 0.099 | 11.8 % | 0.476 |
| **+ dropout-aware training** | 0.839 | **0.674** | **80.3 %** | **0.913** |
| SDR-STGNN (gated) | 0.839 | 0.632 | 75.2 % | 0.892 |

`k=10` is 10 of 25 μPMUs, i.e. 40 % of the fleet offline. Dropout-aware training
costs **no** inference latency or parameters (measured) and +2.7 % training time.
The most accurate architecture under full observability is also the most fragile
under sensor loss.

## Repository layout

```
fault_localization/     # source package
  data/                 # dataset + IEEE-123 OpenDSS source files (.dss)
  models/               # GRU, RGCN, RGSAGE, RGATv2
  simulation/ graph/    # OpenDSS time-stepped sim + substation-rooted graph
  dropout/              # sensor-dropout masking, evaluator, dropout-aware training,
                        #   targeted (worst-case) + persistent (per-event) selection
  experiments/          # CLI, config loading, training runner
config/                 # base + per-model YAML configs
scripts/                # dataset generation, sweeps, summary + figure builders
paper/                  # overleaf.tex + generated figures/ and tables/
results/                # all run outputs, summary CSVs, statistics, runtime
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
# inference-time degradation curves for the trained models
uv run python scripts/run_dropout_study.py \
    --results-dir results/fullscale_final --data-root <dataset> \
    --output-dir results/dropout_study --seeds 42 43 44

# dropout-aware retraining, then re-evaluate + overlay
uv run python scripts/run_dropout_aware_sweep.py
uv run python scripts/run_dropout_study.py \
    --results-dir results/dropout_aware --data-root <dataset> \
    --output-dir results/dropout_study_aware --seeds 42 43 44
uv run python scripts/build_dropout_comparison.py

# SDR-STGNN ablation
uv run python scripts/run_sdr_stgnn.py --seed 42

# comparison baselines
uv run python scripts/run_imputation_baseline.py          # impute-then-infer
uv run python scripts/run_targeted_dropout_study.py       # worst-case dropout
uv run python scripts/run_persistent_dropout_study.py     # per-event dropout

# analysis (no GPU needed)
uv run python scripts/run_significance_tests.py           # paired t-tests + bootstrap
uv run python scripts/run_runtime_benchmark.py            # latency / memory / params
```

## Tests

```bash
uv run pytest
```

## Status

Active research code (ICMLDE 2026 submission). All experiments are complete and
the paper is a full draft.

- All five baselines trained at seeds **42/43/44**; accuracy reported mean ± std.
- **Every** degradation curve is 3-seed (3 training seeds × 3 mask seeds) — for all
  five architectures, the dropout-aware variants, and SDR-STGNN.
- Imputation, targeted (worst-case), and persistent (per-event) dropout baselines
  are done on the RGSAGE-Max backbone.
- Significance testing (paired t-tests + Holm, event-level cluster bootstrap) and a
  runtime/memory benchmark are in `results/statistics/` and `results/runtime/`.

Frozen at depth-4 (a documented deviation from the base paper's single layer, which
was unstable here). The 100-run / 3.30 M-window corpus is generated but the sweep on
it is **parked**, not launched.

`CLAUDE.md` is the repo map and current-status file; `paper/overleaf.tex` is the
source of truth for all reported numbers.

## License

MIT — see [LICENSE](LICENSE).

## Reference

Base paper: Karabulut et al. (2026), *Robustness of Spatio-temporal Graph Neural
Networks for Fault Location in Partially Observable Distribution Grids.* The base
paper PDF is not redistributed here.
