"""Measured run-to-run noise -> proposed tolerances (rag-eval.md § 8.3). No API call. The registry is NOT changed: the author approves the final list.

Input: the items file of a noise run (the same events, 3 to 5 repeats on one commit and configuration). Per repeat, the mean of each metric over the events; then the standard deviation
over the repeats (ddof = 1); tolerance = 2 x sd. Caveat: the noise events are 10 events with 2 to 4 events per stratum, so the tolerance describes the mean over those 10 events, not over the 30 S4 events.
`python -m gridsense.eval.tolerances <items.jsonl>`
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from gridsense.config import ROOT, TRACK
from gridsense.eval.compare import S4_CHECKS, S4_JUDGE, load_registry
from gridsense.s4 import report as RP

OUT = TRACK / "reports" / "s7"


def measured(items: list[dict]) -> dict:
    by_rep: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        if not it.get("error"):
            by_rep[int(it.get("repeat", 0))].append(RP.metric_values(it))
    reps = sorted(by_rep)
    if len(reps) < 3:
        raise ValueError(f"need at least 3 repeats for a measured tolerance, got {len(reps)}")
    reg, out = load_registry(), {}
    for rname, key in {**S4_CHECKS, **S4_JUDGE}.items():
        per_rep = []
        for r in reps:
            vals = [m[key] for m in by_rep[r] if m.get(key) is not None]
            if vals:
                per_rep.append(float(np.mean(vals)))
        if len(per_rep) >= 3:
            sd = float(np.std(per_rep, ddof=1))
            out[rname] = {"repeats": len(per_rep), "means_per_repeat": per_rep, "sd": sd, "proposed_tolerance": 2 * sd, "role": reg[rname]["role"], "current_tolerance_source": reg[rname]["tolerance_source"]}
    return {"metrics": out, "repeats": len(reps), "events": len({it["event_id"] for it in items}), "rule": "tolerance = 2 x sd over the repeats (rag-eval.md § 8.3)",
            "caveat": "mean over the noise events only (2 to 4 per stratum); the registry is unchanged until the author approves"}


def main(argv: list[str] | None = None) -> None:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 1:
        raise SystemExit("usage: python -m gridsense.eval.tolerances <noise items.jsonl>")
    items = [json.loads(x) for x in Path(a[0]).read_text(encoding="utf-8").splitlines() if x.strip()]
    res = measured(items) | {"source_items": a[0]}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "measured_tolerances.json"
    p.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(p.relative_to(ROOT)), **{k: {"sd": round(v["sd"], 4), "proposed_tolerance": round(v["proposed_tolerance"], 4)} for k, v in res["metrics"].items()}}, indent=1))


if __name__ == "__main__":
    main()
