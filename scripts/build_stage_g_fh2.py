"""F-H2: case studies (S7.2) on the test split, R5 seed 42, k = 0, sigma = 0. CPU only.

Selection rules (fixed before any event was looked at; the first window in sorted ``window_id`` order wins):
  Case A  10 Ohm, last window of the event (w0039), true class != 0, network-only vote wrong, fused vote right;
  Case B  the same filter, network-only vote right, fused vote wrong (the counter-case: fusion makes it worse).
Both use mask offset 0 of the saved predictions (no dropout, no noise: k = 0, sigma = 0). The silent-failure case of the spec is
not built here: G-TF2 has no pre-registration on the test split (``stage_f_gtf2_preregistration.md``).

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_fh2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.config.constants import FAULT_BUSES  # noqa: E402
from fault_localization.data.dataset import WindowGraphDataset  # noqa: E402
from fault_localization.stage_g import scoring as S  # noqa: E402

RUN = "results/s4_4/r5/pcstgnn/seed_00042"
PRED = REPO / "results/stage_g/test/r5/seed_00042_predictions.npz"
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def pick() -> dict:
    with np.load(PRED, allow_pickle=False) as z:
        ids, y, res = z["window_ids"], z["y_true"].astype(int), z["resistance"]
        pf, pn = z["pf__k0_s0_o0"].astype(int), z["pn__k0_s0_o0"].astype(int)
    last = np.array([str(w).endswith("_w0039") for w in ids])
    base = last & np.isclose(res, 10.0) & (y != 0)
    order = np.argsort(ids)
    out = {}
    for case, cond in (("A", (pn != y) & (pf == y)), ("B", (pn == y) & (pf != y))):
        cand = [i for i in order if base[i] and cond[i]]
        out[case] = {"n_candidates": len(cand), "index": int(cand[0]) if cand else None,
                     "window_id": str(ids[cand[0]]) if cand else None}
    return out


def name(c: int) -> str:
    return "none" if c == 0 else f"bus {FAULT_BUSES[c - 1]}"


def main() -> int:
    chosen = pick()
    ds = WindowGraphDataset(str(REPO / "dataset_fullscale_50"), "test")
    pos = {r[0]: i for i, r in enumerate(ds.records)}
    model, _ = S.load_model(REPO / RUN, ds.edge_index, torch.device("cpu"))
    beta = float(model.beta)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    info = {"rules": __doc__.split("Selection rules")[1].split("Both use")[0].strip(), "beta": beta}
    for row, case in enumerate(("A", "B")):
        c = chosen[case]
        if c["window_id"] is None:
            info[case] = c
            continue
        item = ds[pos[c["window_id"]]]
        x = torch.from_numpy(item["x"])[None]
        with torch.no_grad():
            z = model.network_logits(x)
            rho = model._rho(x, sigma0=0.0)[0][0]
            p_net = torch.softmax(z, -1).sum(1)[0] / 25
            p_fused = torch.softmax(z - beta * rho[None, None, :], -1).sum(1)[0] / 25
        y = int(item["y"])
        psm = int(rho.argmin())
        info[case] = {**c, "true_class": y, "true": name(y), "resistance": float(item["metadata"]["resistance"]),
                      "fault_type": item["metadata"]["fault_type"], "psm_class": psm, "psm": name(psm),
                      "net_pred": int(p_net.argmax()), "fused_pred": int(p_fused.argmax()),
                      "rho_true": float(rho[y]), "rho_min": float(rho.min()),
                      "p_net_true": float(p_net[y]), "p_fused_true": float(p_fused[y]),
                      "p_net_top": float(p_net.max()), "p_fused_top": float(p_fused.max()),
                      "rho": rho.tolist(), "p_net": p_net.tolist(), "p_fused": p_fused.tolist()}
        ax = axes[row, 0]
        cols = [BLUE if i == y else "#c3c2b7" for i in range(26)]
        ax.bar(range(26), rho.numpy(), color=cols, width=0.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("candidate class", color=MUTED, fontsize=8)
        ax.set_ylabel("ρ (nats; 0 = best)", color=MUTED, fontsize=8)
        ax.scatter([y], [0], marker="*", s=110, color=BLUE, zorder=4, clip_on=False)
        ax.set_title(f"Case {case}: {c['window_id']}\n{info[case]['fault_type']}, 10 Ω; ★ = true ({name(y)}); PSM argmin = {name(psm)}",
                     fontsize=8, loc="left", color=INK)
        ax = axes[row, 1]
        keep = sorted(set(np.where(np.maximum(p_net.numpy(), p_fused.numpy()) > 0.03)[0].tolist()) | {y})
        xs = np.arange(len(keep))
        ax.bar(xs - 0.2, p_net.numpy()[keep], 0.4, color="#c3c2b7", label="network only")
        ax.bar(xs + 0.2, p_fused.numpy()[keep], 0.4, color=ORANGE if case == "B" else BLUE, label="fused (R5)")
        ax.set_xticks(xs, [("★" if k == y else "") + str(k) for k in keep], fontsize=7)
        ax.set_xlabel("class (★ = true)", color=MUTED, fontsize=8)
        ax.set_ylabel("P(f | X) = vote / 25", color=MUTED, fontsize=8)
        ax.set_title(f"network-only → {name(info[case]['net_pred'])}; fused → {name(info[case]['fused_pred'])}", fontsize=8, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("F-H2  Case studies, R5 seed 42, test split, k = 0, σ = 0 (rule-selected windows)", fontsize=9, x=0.01, y=1.0, ha="left")
    fig.tight_layout()
    fo = REPO / "results/stage_g/figures"
    fig.savefig(fo / "F-H2_case_studies.png", dpi=200, bbox_inches="tight")
    fig.savefig(fo / "F-H2_case_studies.pdf", bbox_inches="tight")
    (fo / "F-H2_data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("rho", "p_net", "p_fused")} if isinstance(v, dict) else v
                      for k, v in info.items() if k != "rules"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
