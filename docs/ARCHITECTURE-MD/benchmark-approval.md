# Phase 8 — Benchmark aggregation, error analysis, and approval

Read [experiments-overview.md](experiments-overview.md) first. This phase does not redefine model hyperparameters, optimizer, voting, metrics, or dataset identity.

## Purpose

Convert completed baseline runs into an auditable decision: is `benchmark_v1.0` sufficiently reproduced for a later edge-feature comparison?

## Multi-seed aggregation

Implement `fault_localization.experiments.reporting`:

1. Require at least three comparable successful seeds per model before uncertainty is reported.
2. Compute mean, sample standard deviation, and two-sided 90% Student-t CI for canonical soft-voted test F1 and headline metrics; save `n`, critical value, formula.
3. Retain node metrics as diagnostics only.
4. Reject mismatched benchmark, split/normalizer, metric definition, or shared resolved config.
5. Report timing with hardware context; do not compare unmatched hardware.

Write `results/benchmark_v1.0/benchmark_comparison.csv` and `benchmark_validation.md` with assumptions, [BASE] references, results, deviations, and disposition (`approved`, `investigate`, `not comparable`).

| Model | Seeds | Graph F1 mean ± 90% CI | Node F1 diagnostic | Precision | Recall | Train time | Inference time | Fingerprint |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Error analysis

Use the best validated RGSAGE chosen by its predeclared aggregate result, not its best isolated run. Generate `analysis/errors.py` and `results/benchmark_v1.0/error_analysis.md` with:

1. 26×26 fixed-order confusion and most-confused bus pairs;
2. metrics by fault bus/type/resistance/load bin/window position;
3. error by measured-graph shortest-path and electrical-impedance distance where a defensible fault-to-graph mapping exists;
4. representative failures keyed by window ID, truth, vote prediction, metadata;
5. interpretation that distinguishes observation from causal claim.

Analysis may motivate edge features; no edge attributes may enter a reproduced baseline or be claimed as a baseline improvement.

## Approval criteria

- [ ] Complete corpus passes strict validation and frozen manifest is recorded.
- [ ] Material protocol decisions have reproduction-log entries.
- [ ] GRU and both RGSAGE variants converge across at least three seeds.
- [ ] Headline F1 is verified soft-voted graph/window F1; no test tuning occurred.
- [ ] Relative behavior qualitatively agrees with [BASE] §5.1: GNNs substantially exceed GRU and variants cluster (references: GRU 86.3 ± 4.2; RGCN 94.7 ± 0.3; RGSAGE mean/max 94.7 ± 0.1; RGATv2 94.9 ± 0.6).
- [ ] Runs contain config, environment, seed, manifest, checkpoints, curves, confusion, and predictions.
- [ ] No unresolved issue explains a major deviation.

Conditional Phase 7 models may be omitted only with documented scope decision. GRU and both RGSAGE variants are mandatory.

## Disposition

- **Approved:** plan the proposed model against this exact fingerprint.
- **Investigate:** retain artifacts and return to earliest implicated phase.
- **Not comparable:** do not rank models; create a new benchmark version only with written rationale.
