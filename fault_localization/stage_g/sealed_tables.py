"""T8 (OOD resistance) and F-G3 data from a finished sealed-set run (``results/stage_g/ood``).

Inputs: ``gate6_analysis.json`` (``scripts/run_gate6_analysis.py``) and ``<name>_seed<seed:05d>_scoring.json`` of the
pipeline. Everything is read from those files; the module refuses a directory without ``COMPLETED.json`` or a smoke run.
The H3 / Gate 6 verdict is copied from ``gate6_analysis.json["gate6"]``, never recomputed here.
"""
from __future__ import annotations

import json
from pathlib import Path

from fault_localization.stage_g import tables as T

SEEDS = T.SEEDS
CELL_ORDER = ["k0_sigma0.0005", "k0_sigma0.002", "k0_sigma0.01", "k0_sigma0", "k5_sigma0", "k10_sigma0"]
PRIMARY = CELL_ORDER[:3]
SUBSET_ORDER = ["pooled", "R=3", "R=5", "R=20", "R=30", "grounded", "ungrounded"]
SUBSET_NOTE = {"pooled": "all events", "R=3": "3 Ω (off the operator R grid)", "R=5": "5 Ω (off the grid)",
               "R=20": "20 Ω (off the grid)", "R=30": "30 Ω (on the grid, axis edge)",
               "grounded": "AG BG CG ABG BCG CAG ABCG", "ungrounded": "AB BC CA ABC"}


def check_reportable(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    if not (out_dir / "COMPLETED.json").exists():
        raise FileNotFoundError(f"no COMPLETED.json in {out_dir}: the sealed run has not finished")
    if json.loads((out_dir / "STARTED.json").read_text()).get("smoke", False):
        raise ValueError("smoke run: not reportable")
    path = out_dir / "gate6_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/run_gate6_analysis.py first")
    a = json.loads(path.read_text())
    if not a.get("reportable", False):
        raise ValueError("gate6_analysis.json is not marked reportable")
    return a



def build_t8(analysis: dict) -> list[dict]:
    rows = []
    for cell in CELL_ORDER:
        c = analysis["cells"].get(cell)
        if c is None:
            continue
        k, s = T.parse_cell(cell)
        for sub in SUBSET_ORDER:
            e = c.get(sub)
            if e is None:
                continue
            f1 = e["f1"]
            row = {"cell": f"k={k}, σ={s:g}", "role": "primary (Gate 6)" if cell in PRIMARY and sub == "pooled" else "descriptive",
                   "subset": sub, "subset_meaning": SUBSET_NOTE.get(sub, ""), "n_events": e["n_events"],
                   "R5": T.fmt(*T.mean_std(f1["r5"])), "R0ⁿ": T.fmt(*T.mean_std(f1["r0n"])),
                   "R0": T.fmt(*T.mean_std(f1["r0"])) if "r0" in f1 else "—",
                   "ΔF1 (R5 − R0ⁿ)": f"{e['delta_mean']:+.3f}", "per-seed Δ": " / ".join(f"{d:+.3f}" for d in e["delta_per_seed"]),
                   "seeds > 0": f"{sum(d > 0 for d in e['delta_per_seed'])}/{len(e['delta_per_seed'])}",
                   "95 % CI (event bootstrap)": f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]" if "ci95" in e else "—",
                   "delta": e["delta_mean"], "ci95": e.get("ci95")}
            rows.append(row)
    return rows


def gate6_lines(analysis: dict) -> list[str]:
    g = analysis["gate6"]
    out = [f"Gate 6 evaluable: {g['evaluable']}; passed: {g['passed']}.", g["rule"]]
    for s, v in g["per_sigma"].items():
        out.append(f"σ = {s}: Δ per seed {[round(d, 4) for d in v['delta_per_seed']]}, CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
                   f"sign rule {'met' if v['sign_all_seeds_positive'] else 'not met'}, CI lower bound "
                   f"{'above' if v['ci_lower_above_zero'] else 'not above'} 0 → {'pass' if v['passes'] else 'fail'}.")
    return out


def f1_by_resistance_series(out_dir: Path, test_dir: Path | None = None, sigma: float = 0.0) -> dict:
    """F1 vs R at k = 0 for every model: OOD resistances from the sealed run, in-distribution ones from the test split.

    ``{model: {"ood": {R: [per seed]}, "in": {R: [per seed]}}}``; ``in`` only where a test-split file with ``f1_by_resistance`` exists.
    """
    out_dir = Path(out_dir)
    series: dict = {}
    names = sorted({p.name.split("_seed")[0] for p in out_dir.glob("*_seed*_scoring.json")})
    for name in names:
        ood: dict = {}
        for s in SEEDS:
            p = out_dir / f"{name}_seed{s:05d}_scoring.json"
            if p.exists():
                for r, v in T.load_cells(p)[(0, sigma)]["by_res"].items():
                    ood.setdefault(r, []).append(v)
        ind: dict = {}
        if test_dir is not None:
            for s in SEEDS:
                p = Path(test_dir) / name / f"seed_{s:05d}_scoring.json"
                if p.exists():
                    for r, v in T.load_cells(p)[(0, sigma)]["by_res"].items():
                        ind.setdefault(r, []).append(v)
        series[name] = {"ood": ood, "in": ind}
    return series
