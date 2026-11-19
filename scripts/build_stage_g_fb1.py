"""F-B1 and the S1.1 check: Y-bus, Z-bus transfer sensitivity |Z_pf| / |Z_ff| (``PC-STGNN/03_STAGE_B_operator.md`` S1.1).

    PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 19 uv run python scripts/build_stage_g_fb1.py

CPU only. Compiles the IEEE-123 feeder with OpenDSS, reads the system Y matrix (``Circuit.SystemY``), checks that Y is
symmetric and that Y·V reproduces OpenDSS's own nodal injection currents (``Circuit.YCurrents``) after one solve, then forms
Z = Y^-1 and the ratio |Z_pf| / |Z_ff| per phase (fault bus f, sensor bus p, same phase at both buses; mean over the phases both have).
Used for the physics section and the identifiability discussion only; the residual does not use it.
Writes results/pcstgnn/audit/s1_1_network/{s1_1_network.json,transfer_sensitivity.npy} and results/stage_g/figures/F-B1_*.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import opendssdirect as dss  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.config.constants import FAULT_BUSES, PMU_BUSES  # noqa: E402
from fault_localization.simulation import feeder  # noqa: E402

OUT = REPO / "results/pcstgnn/audit/s1_1_network"
FIG = REPO / "results/stage_g/figures"


def main() -> int:
    feeder.compile_feeder()
    dss.Solution.Solve()
    names = [n.lower() for n in dss.Circuit.YNodeOrder()]
    n = len(names)
    y = np.array(dss.Circuit.SystemY())
    Y = (y[0::2] + 1j * y[1::2]).reshape(n, n)
    v = np.array(dss.Circuit.YNodeVArray())
    V = v[0::2] + 1j * v[1::2]
    i = np.array(dss.Circuit.YCurrents())
    I = i[0::2] + 1j * i[1::2]
    sym = float(np.abs(Y - Y.T).max())
    rel = float(np.linalg.norm(Y @ V - I) / np.linalg.norm(I))
    Z = np.linalg.inv(Y)
    idx = {nm: k for k, nm in enumerate(names)}

    def zratio(f: int, p: int) -> float:
        vals = []
        for ph in (1, 2, 3):
            a, b = idx.get(f"{f}.{ph}"), idx.get(f"{p}.{ph}")
            if a is not None and b is not None:
                vals.append(abs(Z[b, a]) / abs(Z[a, a]))
        return float(np.mean(vals)) if vals else float("nan")

    S = np.array([[zratio(f, p) for p in PMU_BUSES] for f in FAULT_BUSES])
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "transfer_sensitivity.npy", S)
    res = {"n_nodes": n, "y_symmetry_max_abs_diff": sym, "y_v_vs_ycurrents_rel_err": rel, "pass_criterion": "rel err < 1e-3 (0.1 %)",
           "pass": bool(rel < 1e-3), "n_nan_pairs": int(np.isnan(S).sum()), "ratio_min": float(np.nanmin(S)), "ratio_max": float(np.nanmax(S)),
           "definition": "mean over phases present at both buses of |Z[p.ph, f.ph]| / |Z[f.ph, f.ph]|; Z = inv(SystemY) after one solve",
           "note": "check is I = Y·V against OpenDSS YCurrents at all nodes, not one branch current"}
    (OUT / "s1_1_network.json").write_text(json.dumps(res, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    im = ax.imshow(np.ma.masked_invalid(S), cmap="Blues", vmin=0, vmax=1.05, aspect="auto")
    ax.set_xticks(range(len(PMU_BUSES)), [str(b) for b in PMU_BUSES], rotation=90, fontsize=6)
    ax.set_yticks(range(len(FAULT_BUSES)), [str(b) for b in FAULT_BUSES], fontsize=6)
    ax.set_xlabel("sensor bus p", color="#52514e")
    ax.set_ylabel("fault bus f", color="#52514e")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("|Z_pf| / |Z_ff|", fontsize=8)
    ax.set_title("F-B1  Transfer sensitivity of each sensor to each fault bus", fontsize=9, loc="left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "F-B1_transfer_sensitivity.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "F-B1_transfer_sensitivity.pdf", bbox_inches="tight")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
