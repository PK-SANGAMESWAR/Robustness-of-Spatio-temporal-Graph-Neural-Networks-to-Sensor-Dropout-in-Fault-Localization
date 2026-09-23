# S0.5 -- Timing calibration

Generated 2026-09-23 by `scripts/audit_timing.py` on this machine (24 cores, T1000 8 GB).

## Measured

| quantity | value |
|---|---|
| signature solve, `reuse` mode, 3 fault steps | **2.49 ms** (402/s single-core) |
| signature solve, `daily` mode (recompile each time) | 7.3 ms (137.1/s) |
| full 60-step Daily-mode event (the OOD unit) | 17 ms |

The plan assumed ~45 solves/s.  The measured single-core rate is **402/s**, i.e. 9x faster, because `reuse` mode keeps the compiled circuit and the compile -- not the solve -- was the cost being estimated.

## Signature table

| L step | L points | solves | serial | 20 workers | size |
|---|---|---|---|---|---|
| 0.05 | 17 | 70,142 | 2.9 min | 0.1 min | 21 MB |
| 0.025 | 33 | 136,158 | 5.6 min | 0.3 min | 41 MB |

**Consequence for S1.2.** Both candidate L steps cost minutes, not hours, so the grid step should be chosen purely on interpolation error -- there is no budget argument for the coarser grid.

## OOD corpora

| set | events | serial | 20 workers |
|---|---|---|---|
| pilot | 1,100 | 0.01 h | 0 min |
| sealed test | 11,000 | 0.05 h | 0 min |

## GPU (not re-measured)

From `results/runtime/runtime_training.csv`: RGSAGE-Max 30 epochs = 6.7 h standard / 6.9 h dropout-aware; RGATv2 12.3 h.  One run therefore exceeds 6 h and the reduced-seed rule of `02_STAGE_A_audit.md` S0.5 stands: 3 seeds for R0, R2, R4, R5, R6, G0, G2; 1 seed for R1, R3, R7, G1, G3, B2a.
