"""Stage A S0.2 -- metadata coverage, and does the generator still reproduce?

Two questions, both prerequisites for later stages:

1. **Coverage.** Does every one of the 41,250 events carry the seven metadata
   keys?  S1.4 and S1.5 use `fault_bus`, `fault_type`, `resistance` and
   `load_multiplier` as ground truth for the operator's fidelity, so a gap here
   would silently shrink those tests.
2. **Regeneration.** Re-running the generator with an event's stored seed must
   reproduce that event bit-for-bit.  This is what keeps the voltage-angle
   escape hatch open: if magnitude-only physics fails Gate 1, the fallback is to
   re-log the *same* events with angles (`monitors.py:78-80` already computes and
   discards them), which only works if the events are reproducible.

Usage:
    uv run python scripts/audit_metadata.py [--n-regen 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from random import Random

# Running `python scripts/x.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"

REQUIRED_KEYS = ["fault_bus", "fault_type", "resistance", "run_id",
                 "load_multiplier", "seed", "feeder_config"]


def scan_coverage(raw_dir: Path, limit: int | None = None) -> dict:
    files = sorted(raw_dir.glob("*.npz"))
    if limit:
        files = files[:limit]
    present = Counter()
    shapes = Counter()
    bad_seed: list[str] = []
    from fault_localization.config.constants import deterministic_seed

    for path in files:
        with np.load(path, allow_pickle=True) as loaded:
            meta = json.loads(str(loaded["meta"]))
            shapes[loaded["data"].shape] += 1
        for key in REQUIRED_KEYS:
            if key in meta:
                present[key] += 1
        expected = deterministic_seed(meta["fault_bus"], meta["fault_type"],
                                      meta["resistance"], meta["run_id"])
        if meta.get("seed") != expected:
            bad_seed.append(path.name)
    return {
        "n_events": len(files),
        "coverage": {k: present[k] for k in REQUIRED_KEYS},
        "full_coverage": all(present[k] == len(files) for k in REQUIRED_KEYS),
        "shapes": {str(k): v for k, v in shapes.items()},
        "n_seed_mismatch": len(bad_seed),
        "seed_mismatch_examples": bad_seed[:5],
    }


def regenerate(raw_dir: Path, n: int, seed: int = 0) -> list[dict]:
    """Re-simulate n events from their stored seed and diff against disk."""
    from fault_localization.simulation.runner import simulate_fault_event

    files = sorted(raw_dir.glob("*.npz"))
    picks = Random(seed).sample(files, min(n, len(files)))
    rows = []
    for path in picks:
        with np.load(path, allow_pickle=True) as loaded:
            stored = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        rng = Random(meta["seed"])
        fresh, fresh_meta = simulate_fault_event(
            meta["fault_bus"], meta["fault_type"], meta["resistance"],
            meta["run_id"], rng, meta.get("feeder_config", "default"),
        )
        diff = float(np.nanmax(np.abs(fresh - stored)))
        rows.append({
            "event": path.stem,
            "load_stored": meta["load_multiplier"],
            "load_regen": fresh_meta["load_multiplier"],
            "load_exact": fresh_meta["load_multiplier"] == meta["load_multiplier"],
            "max_abs_diff": diff,
            "bit_identical": bool(np.array_equal(np.nan_to_num(fresh, nan=-1.0),
                                                 np.nan_to_num(stored, nan=-1.0))),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-regen", type=int, default=5)
    args = ap.parse_args()
    logging.disable(logging.INFO)

    raw = Path(args.data_root) / "raw_sim"
    cover = scan_coverage(raw, args.limit)
    regen = regenerate(raw, args.n_regen) if args.n_regen else []

    all_bit = all(r["bit_identical"] for r in regen) if regen else None
    all_load = all(r["load_exact"] for r in regen) if regen else None
    passed = cover["full_coverage"] and cover["n_seed_mismatch"] == 0 and (all_bit is not False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S0.2 -- Metadata coverage and regeneration",
        "",
        f"Generated {date.today().isoformat()} by `scripts/audit_metadata.py`.",
        "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}.**",
        "",
        "## Coverage",
        "",
        f"Events scanned: **{cover['n_events']}**.  Record shapes: `{cover['shapes']}`.",
        "",
        "| key | events carrying it |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} / {cover['n_events']} |" for k, v in cover["coverage"].items()]
    lines += [
        "",
        f"Seeds recomputed from `deterministic_seed(bus, type, R, run_id)` and compared "
        f"with the stored value: **{cover['n_seed_mismatch']} mismatches**.",
        "",
        "## Regeneration from the stored seed",
        "",
        "| event | stored L | regenerated L | max abs diff | bit-identical |",
        "|---|---|---|---|---|",
    ]
    for r in regen:
        lines.append(f"| `{r['event']}` | {r['load_stored']:.9f} | {r['load_regen']:.9f} | "
                     f"{r['max_abs_diff']:.3e} | {'yes' if r['bit_identical'] else 'NO'} |")
    lines += [
        "",
        "## Why it matters",
        "",
        "The load field is a **scalar** (S0.0), which is the reason L is a profiled axis rather",
        "than an ensemble.  Bit-identical regeneration keeps the Gate-1 fallback open: if",
        "magnitude-only physics is too weak, the same events can be re-logged with voltage",
        "angles, which `monitors.py:78-80` already computes and throws away.",
    ]
    (out_dir / "metadata_coverage.md").write_text("\n".join(lines) + "\n")
    (out_dir / "metadata_coverage.json").write_text(
        json.dumps({"pass": passed, **cover, "regeneration": regen}, indent=2) + "\n")

    print(f"coverage: {cover['full_coverage']}  seed mismatches: {cover['n_seed_mismatch']}")
    for r in regen:
        print(f"  {r['event']}: max|diff|={r['max_abs_diff']:.3e} "
              f"bit_identical={r['bit_identical']} load_exact={r['load_exact']}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"wrote {out_dir / 'metadata_coverage.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
