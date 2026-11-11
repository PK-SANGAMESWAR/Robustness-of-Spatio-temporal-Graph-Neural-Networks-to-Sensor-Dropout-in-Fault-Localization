"""Summarise arm R6^i (R5 weights, shuffled signature table, inference only) against R5 and apply its pre-registered rule.

Reads ``results/s4_4/{r5,r6i}/pcstgnn/seed_*/scoring_full_grid.json`` (validation split). Writes
``results/s4_4/r6i/summary.json``. Rule (``06_STAGE_E_pcstgnn.md``, R6^i): the model uses the correct physics
content if R5 fused F1 > R6^i fused F1 in all 3 seeds at the H2 and H4 conditions
(H2 = k=10 at sigma in {0, 5e-4, 2e-3, 1e-2}; H4 = k=0 at sigma in {5e-4, 2e-3, 1e-2}; 7 cells x 3 seeds = 21 comparisons).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
KS = (0, 1, 2, 5, 10)
SIGMAS = (0.0, 0.0005, 0.002, 0.01)
H2 = [(10, s) for s in SIGMAS]
H4 = [(0, s) for s in SIGMAS[1:]]


def load(arm: str, seed: int) -> dict:
    return json.loads((REPO / f"results/s4_4/{arm}/pcstgnn/seed_{seed:05d}/scoring_full_grid.json").read_text())["cells"]


def main() -> int:
    r5 = {s: load("r5", s) for s in SEEDS}
    r6 = {s: load("r6i", s) for s in SEEDS}
    table = {}
    net_equal = True
    for k in KS:
        for sg in SIGMAS:
            key = f"k{k}_sigma{sg}"
            row = {"r5_fused": [], "r6i_fused": [], "r6i_network_only": [], "r5_network_only": [], "r6i_psm_only": [], "r5_psm_only": []}
            for s in SEEDS:
                a, b = r5[s][key], r6[s][key]
                row["r5_fused"].append(a["f1_fused_mean"])
                row["r6i_fused"].append(b["f1_fused_mean"])
                row["r5_network_only"].append(a["f1_network_only_mean"])
                row["r6i_network_only"].append(b["f1_network_only_mean"])
                row["r5_psm_only"].append(a["f1_psm_mean"])
                row["r6i_psm_only"].append(b["f1_psm_mean"])
                net_equal &= (a["f1_network_only_mean"] == b["f1_network_only_mean"])
            row["mean"] = {n: float(np.mean(v)) for n, v in row.items() if isinstance(v, list)}
            table[key] = row
    comps = []
    for k, sg in H2 + H4:
        row = table[f"k{k}_sigma{sg}"]
        for i, s in enumerate(SEEDS):
            comps.append({"cell": f"k{k}_sigma{sg}", "seed": s, "r5": row["r5_fused"][i], "r6i": row["r6i_fused"][i],
                          "delta": row["r5_fused"][i] - row["r6i_fused"][i], "r5_higher": bool(row["r5_fused"][i] > row["r6i_fused"][i])})
    n_pass = sum(c["r5_higher"] for c in comps)
    all_cells_higher = sum(table[k]["r5_fused"][i] > table[k]["r6i_fused"][i] for k in table for i in range(3))
    out = {"seeds": SEEDS, "network_only_identical_to_r5": bool(net_equal), "rule_comparisons": len(comps),
           "rule_r5_higher": n_pass, "rule_passed": bool(n_pass == len(comps) == 21),
           "all_20_cells_r5_higher_count": int(all_cells_higher), "all_20_cells_total": 60, "comparisons": comps, "table": table}
    (REPO / "results/s4_4/r6i/summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("network_only_identical_to_r5", "rule_comparisons", "rule_r5_higher", "rule_passed",
                                           "all_20_cells_r5_higher_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
