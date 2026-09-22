"""Summary table + figure for the PIGNN selection screen.

Reads only what is on disk and never re-runs a model, so the figure cannot
drift from the results. Every arm is budget-matched against RGSAGE-Max's own
validation curve at the same epoch, read from its metrics_by_epoch.csv, so no
baseline re-run is involved in the comparison.

Outputs:
    results/pignn_screen/screen_summary.csv
    results/pignn_screen/screen_summary.json
    paper/figures/pignn_screen.png     (two panels: trajectories + final bars)

Usage:
    uv run python scripts/build_pignn_screen_summary.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {
    "p3": ("P3  physics in the LOSS",  "Type A", "#c2410c"),
    "p1": ("P1  physics in the GRAPH", "Type C", "#0369a1"),
    "p2": ("P2  physics in the PRIOR", "Type D", "#7c3aed"),
}
BASE = "results/fullscale_final/rgsage_max/seed_00042/metrics_by_epoch.csv"


def epochs_of(path):
    try:
        return [(int(r["epoch"]), float(r["val_graph_f1"]),
                 float(r.get("train_phys_loss") or "nan"))
                for r in csv.DictReader(open(path))]
    except OSError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="results/pignn_screen")
    ap.add_argument("--figure", default="paper/figures/pignn_screen.png")
    args = ap.parse_args()

    base = epochs_of(BASE)
    rows, curves = [], {}
    for arm, (label, ptype, _) in ARMS.items():
        d = Path(f"results/pignn_{arm}{args.suffix}/pignn/seed_{args.seed:05d}")
        curves[arm] = epochs_of(d / "metrics_by_epoch.csv")
        mj = d / "metrics.json"
        if not mj.exists():
            rows.append({"arm": arm, "label": label, "type": ptype,
                         "status": "MISSING"})
            continue
        m = json.loads(mj.read_text())
        n_ep = len(curves[arm])
        ref = next((f for e, f, _ in base if e == n_ep), None)
        rows.append({
            "arm": arm, "label": label, "type": ptype, "status": "ok",
            "epochs": n_ep,
            "test_graph_f1": round(m["test_graph"]["f1"], 4),
            "best_val_f1": round(m["best_validation_graph_f1"], 4),
            "baseline_val_at_same_epoch": None if ref is None else round(ref, 4),
            "delta_val_vs_baseline": None if ref is None
            else round(m["best_validation_graph_f1"] - ref, 4),
        })

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "screen_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    done = [r for r in rows if r["status"] == "ok"]
    if done:
        with (out / "screen_summary.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(done[0])); w.writeheader(); w.writerows(done)

    print(f"{'arm':4s} {'type':7s} {'best_val':>9s} {'baseline':>9s} {'delta':>8s} {'test_f1':>9s}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['arm']:4s} {r['type']:7s}   -- not finished --"); continue
        print(f"{r['arm']:4s} {r['type']:7s} {r['best_val_f1']:9.4f} "
              f"{r['baseline_val_at_same_epoch'] or float('nan'):9.4f} "
              f"{r['delta_val_vs_baseline'] or float('nan'):+8.4f} {r['test_graph_f1']:9.4f}")

    # ---- figure ----------------------------------------------------------
    if not done:
        print("\nno finished arms yet; figure skipped"); return 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    n_ep = max(len(c) for c in curves.values() if c)
    ax1.plot([e for e, _, _ in base[:n_ep]], [f for _, f, _ in base[:n_ep]],
             color="0.35", lw=2.2, ls="--", label="RGSAGE-Max (no physics)")
    for arm, (label, _, colour) in ARMS.items():
        if curves[arm]:
            ax1.plot([e for e, _, _ in curves[arm]], [f for _, f, _ in curves[arm]],
                     color=colour, lw=2.2, marker="o", ms=3.5, label=label)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation macro-F1")
    ax1.set_title("Budget-matched training trajectories", fontsize=11)
    ax1.grid(alpha=.3); ax1.legend(fontsize=8, loc="lower right")

    names = [r["label"] for r in done]
    vals = [r["best_val_f1"] for r in done]
    refs = [r["baseline_val_at_same_epoch"] for r in done]
    y = range(len(done))
    ax2.barh(list(y), vals, color=[ARMS[r["arm"]][2] for r in done], height=.55)
    for i, ref in enumerate(refs):
        if ref is not None:
            ax2.plot([ref, ref], [i - .32, i + .32], color="0.2", lw=2.4)
    ax2.set_yticks(list(y)); ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlabel("best validation macro-F1")
    ax2.set_title("vs RGSAGE-Max at the same budget (bar = arm, line = baseline)",
                  fontsize=10)
    ax2.grid(alpha=.3, axis="x")
    fig.suptitle("PIGNN selection screen — same backbone, physics injected at three places",
                 fontsize=12.5)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"\nwrote {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
