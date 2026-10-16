"""F-C3 (table D vs empirical confusability scatter, S2.2) and F-C4 (extrapolation contrast,
pilot version, S2.5), from ``results/pcstgnn/audit/s2_2_confusability/`` and
``results/pcstgnn/audit/s2_5_extrapolation/``.

Usage:
    uv run python scripts/build_stage_c_contrast_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "paper/figures"


def build_fc3() -> None:
    in_dir = REPO / "results/pcstgnn/audit/s2_2_confusability"
    data = np.load(in_dir / "fc3_scatter_data.npz")
    summary = json.loads((in_dir / "confusability_summary.json").read_text())
    D, confusion = data["D"], data["confusion"]
    corr = summary["fc3_correlation"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(np.log10(np.clip(D, 1e-3, None)), confusion, s=14, alpha=0.5, edgecolors="none")
    ax.set_xlabel("log10 D(f1,f2;A) at k=0  (table separability)")
    ax.set_ylabel("empirical nearest-neighbour confusion rate")
    ax.set_title(
        f"F-C3 -- table D vs empirical confusability, {len(D)} location pairs\n"
        f"Spearman rho={corr['rho']:.3f} (p={corr['p_value']:.1e})"
    )
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fc3_table_vs_empirical_confusability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/fc3_table_vs_empirical_confusability.png")


def build_fc4() -> None:
    in_dir = REPO / "results/pcstgnn/audit/s2_5_extrapolation"
    result = json.loads((in_dir / "extrapolation_contrast.json").read_text())
    ks = sorted(int(k) for k in result["ood_pilot_f1_by_k"])
    net_id = [result["in_distribution_f1_by_k"][str(k)] for k in ks]
    net_ood = [result["ood_pilot_f1_by_k"][str(k)] for k in ks]
    net_ood_std = [result["ood_pilot_f1_std_by_k"][str(k)] for k in ks]

    psm_path = REPO / "results/pcstgnn/audit/s2_4_psm/psm_full_eval.json"
    psm = json.loads(psm_path.read_text())
    psm_val_by_k = {int(k): v for k, v in psm["validation"]["f1_by_k"].items()}
    psm_ood_by_k = {int(k): v for k, v in psm["ood_pilot"]["f1_by_k"].items()}
    psm_id = [psm_val_by_k[k] for k in ks]
    psm_ood = [psm_ood_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(ks, net_id, marker="o", linestyle="--", color="tab:blue",
             label="RGSAGE-Max (dropout-aware) -- in-distribution (test)")
    ax.errorbar(ks, net_ood, yerr=net_ood_std, marker="o", color="tab:blue",
                label="RGSAGE-Max (dropout-aware) -- OOD-pilot (unseen R)")
    ax.plot(ks, psm_id, marker="s", linestyle="--", color="tab:orange",
             label="PSM -- validation (trained-R)")
    ax.plot(ks, psm_ood, marker="s", color="tab:orange",
             label="PSM -- OOD-pilot (unseen R)")
    ax.set_xlabel("dropout k")
    ax.set_ylabel("macro-F1 (26-class)")
    ax.set_title(
        "F-C4 -- extrapolation contrast (pilot): network vs PSM under unseen R\n"
        f"drop at k=0: network {result['network_drop_k0']:.3f}, PSM {result['psm_drop_k0']:.3f}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fc4_extrapolation_contrast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR}/fc4_extrapolation_contrast.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_fc3()
    build_fc4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
