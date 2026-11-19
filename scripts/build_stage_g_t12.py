"""T12 — computational cost (CPU part now; latency only on a quiet machine).

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_t12.py            # params, FLOPs, training hours
    uv run python scripts/build_stage_g_t12.py --latency --device cuda                   # adds latency; refuses if not quiet

Writes ``results/stage_g/tables/T12_cost.{csv,md}`` and ``T12_cost.json``. Logic: ``fault_localization/stage_g/cost.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/stage_g/tables"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    import torch

    from fault_localization.stage_g import cost as C
    from fault_localization.stage_g import tables as T

    torch.set_num_threads(args.threads)
    if args.latency:
        bad = C.quiet_reasons()
        if bad:
            print("REFUSED (machine not quiet, 12_CHECKLISTS § 0): " + "; ".join(bad), file=sys.stderr)
            return 2
    prev = json.loads((OUT / "T12_cost.json").read_text()) if (OUT / "T12_cost.json").exists() else {}
    rows, raw = [], {}
    for label, dirs in C.COST_MODELS:
        dirs = [d for d in dirs if (REPO / d / "checkpoint_best.pt").exists()]
        if not dirs:
            continue
        model = C.load_any(dirs[0], torch.device("cpu"))
        n_par, f_net, f_res = C.count_params(model), C.backbone_flops(model), C.residual_flops(model)
        uses_res = bool(getattr(model, "use_fusion", False))
        f_inf = f_net + (f_res if uses_res else 0)
        tr = [C.training_hours(REPO / d) for d in dirs]
        hrs = T.mean_std([t["hours"] for t in tr])
        rate = [r for r in (C.scoring_rate_ms(REPO / d) for d in dirs) if r is not None]
        e = {"label": label, "run_dirs": dirs, "n_seeds": len(dirs), "params": n_par, "flops_backbone": f_net,
             "flops_residual": f_res if uses_res else 0, "flops_per_window": f_inf,
             "train_hours": [t["hours"] for t in tr], "best_epoch": [t["best_epoch"] for t in tr],
             "epochs": [t["epochs"] for t in tr], "scoring_ms_per_window_forward": rate}
        if args.latency:
            dev = torch.device(args.device)
            m = model.to(dev)
            e["latency"] = {f"b{b}": C.latency(m, dev, b, repeats=args.repeats) for b in (1, 32)}
            e["latency_device"] = args.device
        elif label in prev.get("models", {}) and "latency" in prev["models"][label]:
            e["latency"], e["latency_device"] = prev["models"][label]["latency"], prev["models"][label]["latency_device"]
        raw[label] = e
        lat = e.get("latency")
        rows.append({"model": label, "n_seeds": len(dirs), "params": f"{n_par:,}",
                     "MFLOPs / window (backbone + residual)": f"{f_inf / 1e6:.1f} ({f_net / 1e6:.1f} + {e['flops_residual'] / 1e6:.1f})",
                     "training GPU-h": T.fmt(hrs[0], hrs[1], 2), "best-val epoch": "/".join(str(b) for b in e["best_epoch"]),
                     "scoring ms / window-forward": T.fmt(*T.mean_std(rate), 3) if rate else "—",
                     "latency b=1 / b=32 (ms)": (f"{lat['b1']['ms_mean']:.2f} / {lat['b32']['ms_mean']:.2f} ({e['latency_device']})"
                                                 if lat else "not measured (machine busy)")})
        del model
    T.write_table(rows, list(rows[0].keys()), OUT / "T12_cost", "T12 — computational cost",
                  ["Parameters: model rebuilt from its config.yaml. R5 − R0ⁿ = 1 parameter (β).",
                   "FLOPs: backbone by torch FlopCounterMode (matrix products, multiply-add = 2); residual = 4 × table entries "
                   "per window (difference, square, weight, sum; physics/residual.py 'direct'). Residual counted only for "
                   "models that use it at inference (fusion on).",
                   "Training GPU-h: sum of train_seconds in metrics_by_epoch.csv (T1000 8 GB), mean ± std over seeds.",
                   "Scoring ms / window-forward: validation full grid (60 forwards × 248,000 windows), wall clock including "
                   "data loading and noise/mask generation; a throughput figure, not a latency.",
                   "Latency: synthetic batch [B, 25, 20, 6], measured only when no other job runs (12_CHECKLISTS § 0)."])
    (OUT / "T12_cost.json").write_text(json.dumps({"host": platform.node() and "local", "torch": torch.__version__,
                                                   "models": raw}, indent=2) + "\n")
    print((OUT / "T12_cost.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
