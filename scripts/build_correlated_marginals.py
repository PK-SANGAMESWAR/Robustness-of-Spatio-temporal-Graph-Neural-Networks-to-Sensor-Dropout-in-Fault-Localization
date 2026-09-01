"""Per-node drop marginals of the contiguous outage policy.

The contiguous policy of ``dropout/correlated.py`` grows a connected block from
a uniformly chosen root.  Uniform *roots* do not give uniform *nodes*: a node
adjacent to a high-degree hub is reachable from many roots and is swept into the
block far more often than a leaf hanging off a long branch.  The measured-only
graph is a tree whose degrees run from 1 to 5, so the effect is large.

This matters for how the contiguous-versus-scattered comparison may be read.
The two policies differ in *two* ways at once — whether the dropped set is
connected, and how often each individual sensor is in it — so the difference
between their curves cannot be attributed to spatial correlation alone.  This
script measures the second difference so the caveat can be stated with a number
instead of a hedge, and writes it next to the curves it qualifies.

It is pure combinatorics over the graph: no model, no data, no GPU.

Usage:
    uv run python scripts/build_correlated_marginals.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fault_localization.data.dataset import WindowGraphDataset
from fault_localization.dropout.correlated import build_adjacency, sample_connected_subset

N_DRAWS = 200_000
K_LEVELS = [1, 2, 5, 10]


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ds = WindowGraphDataset(str(repo / "dataset_fullscale_50"), "test")
    adjacency = build_adjacency(ds.edge_index, 25)
    degrees = [len(a) for a in adjacency]

    out: dict = {
        "note": ("Empirical per-node probability of being dropped under the "
                 "contiguous (connected-block) policy, against the uniform "
                 "policy's k/25. Monte Carlo over connected-subset draws."),
        "n_draws": N_DRAWS,
        "n_nodes": 25,
        "degrees": degrees,
        "levels": {},
    }

    for k in K_LEVELS:
        rng = random.Random(20260817 + k)
        counts = np.zeros(25)
        for _ in range(N_DRAWS):
            for node in sample_connected_subset(adjacency, k, rng):
                counts[node] += 1
        freq = counts / N_DRAWS
        uniform = k / 25.0
        dev = np.abs(freq - uniform)
        out["levels"][str(k)] = {
            "uniform_marginal": uniform,
            "contiguous_marginal_per_node": [round(float(v), 5) for v in freq],
            "min": float(freq.min()), "max": float(freq.max()),
            "max_abs_deviation": float(dev.max()),
            "max_abs_deviation_rel": float(dev.max() / uniform),
            # Total variation between the two marginal profiles, normalized so
            # 0 = identical incidence and 1 = disjoint.
            "total_variation": float(np.abs(freq - uniform).sum() / (2 * k)),
        }
        print(f"k={k:>2}: uniform={uniform:.3f}  contiguous range "
              f"[{freq.min():.3f}, {freq.max():.3f}]  "
              f"max dev {dev.max():.3f} ({dev.max()/uniform*100:.0f}% of uniform)  "
              f"TV={out['levels'][str(k)]['total_variation']:.3f}")

    out_dir = repo / "results" / "dropout_study_correlated"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "contiguous_node_marginals.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    print("\nRead this alongside correlated_curves_*.csv: the contiguous and "
          "scattered policies differ in per-node incidence as well as in "
          "correlation, so their gap is not attributable to correlation alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
