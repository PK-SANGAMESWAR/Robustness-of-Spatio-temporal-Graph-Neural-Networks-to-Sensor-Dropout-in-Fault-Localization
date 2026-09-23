"""Stage A S0.0 -- is the corpus load one global scalar, or a per-bus vector?

The entire L axis of ``PC-STGNN/01_SPEC.md`` §2 rests on the answer.  If the
load were a per-bus vector, profiling a single scalar L would be the wrong
operator and the first draft's load-ensemble version would be the right one;
the 50 runs of a scenario would not lie on a 1-D curve, and S1.5 -- the decisive
load-axis test -- would be meaningless.

The answer is already known from reading ``simulation/runner.py:116``.  This
script exists to turn that reading into a dated artefact over the *whole* corpus,
because "we checked the code once" is not the same claim as "every one of the
41,250 events stores a scalar".

Usage:
    uv run python scripts/audit_load_model.py [--data-root dataset_fullscale_50]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"


def scan(raw_dir: Path, limit: int | None = None) -> dict:
    files = sorted(raw_dir.glob("*.npz"))
    if limit:
        files = files[:limit]
    kinds: Counter[str] = Counter()
    missing = 0
    values: list[float] = []
    offenders: list[str] = []
    for path in files:
        with np.load(path, allow_pickle=True) as loaded:
            meta = json.loads(str(loaded["meta"]))
        if "load_multiplier" not in meta:
            missing += 1
            offenders.append(path.name)
            continue
        value = meta["load_multiplier"]
        kinds[type(value).__name__] += 1
        if isinstance(value, (list, tuple)) or isinstance(value, np.ndarray):
            offenders.append(path.name)
        else:
            values.append(float(value))
    arr = np.asarray(values)
    return {
        "n_events": len(files),
        "n_missing": missing,
        "type_counts": dict(kinds),
        "n_non_scalar": len(offenders),
        "offenders": offenders[:10],
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "n_distinct": int(np.unique(arr).size) if arr.size else 0,
    }


def quote_paper(tex: Path) -> list[str]:
    """The lines of the Paper-1 draft that describe the load draw."""
    if not tex.exists():
        return []
    hits = []
    for number, line in enumerate(tex.read_text(errors="replace").splitlines(), start=1):
        lowered = line.lower()
        if "load" in lowered and ("multiplier" in lowered or "u(0.5" in lowered
                                  or "0.5, 1.3" in lowered or "per run" in lowered):
            hits.append(f"overleaf.tex:{number}: {line.strip()}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="smoke-test subset")
    args = ap.parse_args()

    root = Path(args.data_root)
    result = scan(root / "raw_sim", args.limit)
    quotes = quote_paper(REPO / "paper/overleaf.tex")

    scalar = (result["n_non_scalar"] == 0 and result["n_missing"] == 0
              and set(result["type_counts"]) <= {"float", "int"})
    per_bus_claim = [q for q in quotes if "per bus" in q.lower() or "per-bus" in q.lower()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S0.0 -- Corpus load model",
        "",
        f"Generated {date.today().isoformat()} by `scripts/audit_load_model.py`.",
        f"Corpus: `{root}`" + (f" (subset of {args.limit} events)" if args.limit else ""),
        "",
        f"**Verdict: the load is {'a single global SCALAR' if scalar else 'NOT a plain scalar'} "
        f"per event.**",
        "",
        "## What was checked",
        "",
        f"- Events scanned: **{result['n_events']}**",
        f"- Events missing `load_multiplier`: **{result['n_missing']}**",
        f"- Stored Python types: `{result['type_counts']}`",
        f"- Events whose value is a list/array (i.e. per-bus): **{result['n_non_scalar']}**",
        f"- Value range: {result['min']:.6f} to {result['max']:.6f}, "
        f"mean {result['mean']:.6f}" if result["min"] is not None else "- No values read",
        f"- Distinct values: **{result['n_distinct']}** of {result['n_events']} events "
        f"(one independent draw per event)",
        "",
        "## Why it matters",
        "",
        "`simulation/runner.py:116` draws `rng.uniform(0.5, 1.3)` once per event and",
        "`monitors.py:_scale_loads_from_base` applies it uniformly to every load.  The 50 runs",
        "of a scenario therefore lie on a **1-D curve** parametrised by L, which is what makes",
        "the profiled load axis of `01_SPEC.md` §2 the right operator and the first draft's",
        "per-bus load ensemble the wrong one.  It is also what makes S1.5 an exact test: the",
        "stored multiplier is the ground truth that `argmin_L` must recover.",
        "",
        "## Paper-1 draft on disk",
        "",
    ]
    lines += ([f"- `{q}`" for q in quotes] if quotes else ["- (no matching lines found)"])
    lines += [
        "",
        f"- Lines claiming a **per-bus** load: **{len(per_bus_claim)}**"
        + ("" if per_bus_claim else " -- none, so no erratum is needed for the draft on disk."),
        "",
        "> Paper 2 must never claim a per-bus load model (`12_CHECKLISTS.md` §D).  If the",
        "> *submitted* copy of Paper 1 differs from this draft, grep it once and record the",
        "> result here.",
    ]
    (out_dir / "load_model.md").write_text("\n".join(lines) + "\n")
    (out_dir / "load_model.json").write_text(json.dumps(
        {"scalar": scalar, "paper_quotes": quotes, **result}, indent=2) + "\n")

    print(f"scanned {result['n_events']} events")
    print(f"types: {result['type_counts']}  non-scalar: {result['n_non_scalar']}  "
          f"missing: {result['n_missing']}")
    print(f"VERDICT: {'SCALAR' if scalar else 'NOT SCALAR'}")
    print(f"wrote {out_dir / 'load_model.md'}")
    return 0 if scalar else 1


if __name__ == "__main__":
    raise SystemExit(main())
