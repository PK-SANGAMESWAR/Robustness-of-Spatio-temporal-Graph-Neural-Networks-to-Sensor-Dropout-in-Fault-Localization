# S0.2 -- Metadata coverage and regeneration

Generated 2026-09-23 by `scripts/audit_metadata.py`.

**Verdict: PASS.**

## Coverage

Events scanned: **50**.  Record shapes: `{'(25, 60, 6)': 50}`.

| key | events carrying it |
|---|---|
| `fault_bus` | 50 / 50 |
| `fault_type` | 50 / 50 |
| `resistance` | 50 / 50 |
| `run_id` | 50 / 50 |
| `load_multiplier` | 50 / 50 |
| `seed` | 50 / 50 |
| `feeder_config` | 50 / 50 |

Seeds recomputed from `deterministic_seed(bus, type, R, run_id)` and compared with the stored value: **0 mismatches**.

## Regeneration from the stored seed

| event | stored L | regenerated L | max abs diff | bit-identical |
|---|---|---|---|---|
| `65_AB_0p1_0047` | 0.601953591 | 0.601953591 | 0.000e+00 | yes |
| `72_BG_1p0_0012` | 1.193557744 | 1.193557744 | 0.000e+00 | yes |
| `108_BC_1p0_0003` | 0.599985215 | 0.599985215 | 0.000e+00 | yes |

## Why it matters

The load field is a **scalar** (S0.0), which is the reason L is a profiled axis rather
than an ensemble.  Bit-identical regeneration keeps the Gate-1 fallback open: if
magnitude-only physics is too weak, the same events can be re-logged with voltage
angles, which `monitors.py:78-80` already computes and throws away.
