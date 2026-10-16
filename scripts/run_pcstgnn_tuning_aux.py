"""S4.3e -- noise-augmented lambda tuning with the auxiliary network-only loss (06_STAGE_E_pcstgnn.md).

Pre-registered in commit 272ba01, before this script or any S4.3e run existed.

Fixed: gamma = 1, noise + dropout augmentation on, 15 epochs, seed 42, beta learned,
training.skip_test_eval = true (the test split is never loaded). Base config
config/pcstgnn_auxdiag/gamma1.yaml (the S4.3d run), changing only lambda_1, lambda_2, epochs.

Grid (serial, in this order): lambda_1 in {0, 0.01, 0.0333, 0.1}, lambda_2 = 0.
Extension rules, after the 4 grid runs, in this order:
  1. lambda_1 = 0.1 best -> add lambda_1 = 0.3, lambda_2 = 0.
  2. Add lambda_2 = 0.1 at the best lambda_1.
"Best" = best guard-valid arm under the selection rule; no valid arm -> no extension.

Early stop (runner, training.collapse_abort): at the end of epoch 3, network-only no-fault share
> 0.95 or network-only F1 <= 0.10 -> run stopped, recorded as collapsed, next run started.
Checkpoint rule: best validation fused F1 at k = 0 (runner, unchanged).
Selection rule: best validation fused F1 averaged over k in {0,1,2,5,10} (S4.3 method, 3 mask
seeds). Ties to 4 decimals -> smaller lambda_1, then smaller lambda_2.
Collapse guard: invalid if, at the selected epoch, network-only F1 <= 0.10 or no-fault > 0.95.
No valid arm -> "no selection"; stop.

Any failed run stops the script (exit 1). It is never retried automatically.

Resumable: a run is skipped if its metrics.json exists; scoring is cached in lambda_selection.json.

Usage:
    uv run python scripts/run_pcstgnn_tuning_aux.py
    uv run python scripts/run_pcstgnn_tuning_aux.py --smoke   # 1 epoch on a separate output root
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from fault_localization.experiments.config import load_config, validate_config  # noqa: E402
from run_pcstgnn_tuning import lambda_selection_score  # noqa: E402

SEED = 42
EPOCHS = 15
GAMMA = 1.0
BASE_CONFIG = REPO / "config" / "pcstgnn_auxdiag" / "gamma1.yaml"
CONFIG_DIR = REPO / "config" / "pcstgnn_tuning_aux"
OUT = REPO / "results" / "pcstgnn_tuning_aux"
DATA_ROOT = "dataset_fullscale_50"
LAMBDA1_GRID = [0.0, 0.01, 0.0333, 0.1]
LAMBDA1_EXTENSION = {0.1: 0.3}
LAMBDA2_EXTENSION = 0.1
GUARD = {"epoch": 3, "min_net_f1": 0.10, "max_class0_frac": 0.95}
VAL_RHO_CACHE = REPO / "cache" / "rtilde_val.npy"


def tag(lam1: float, lam2: float) -> str:
    return f"lam1_{str(lam1).replace('.', 'p')}_lam2_{str(lam2).replace('.', 'p')}"


def run_dir(out: Path, lam1: float, lam2: float) -> Path:
    return out / tag(lam1, lam2) / "pcstgnn" / f"seed_{SEED:05d}"


def build_config(lam1: float, lam2: float, epochs: int, out: Path) -> Path:
    cfg = validate_config(load_config(str(BASE_CONFIG)))
    assert cfg["noise_augmentation"]["enabled"] and cfg["dropout_study"]["dropout_aware_training"]
    assert cfg["model"]["use_fusion"] and cfg["model"]["beta_init"] == 0.0
    assert "rho_k0_cache_path" not in cfg["pc_loss"]   # noise-free cache cannot serve noisy windows
    cfg["pc_loss"].update(lambda1=float(lam1), lambda2=float(lam2), aux_net_loss_weight=GAMMA,
                          log_network_only=True)
    cfg["training"]["epochs"] = epochs
    cfg["training"]["skip_test_eval"] = True
    cfg["training"]["collapse_abort"] = dict(GUARD)
    cfg["output_root"] = str(out.relative_to(REPO))
    cfg["benchmark_version"] = tag(lam1, lam2)
    cfg["seed"] = SEED
    cfg_dir = CONFIG_DIR if out == OUT else out / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{tag(lam1, lam2)}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def train(lam1: float, lam2: float, epochs: int, out: Path) -> bool:
    rd = run_dir(out, lam1, lam2)
    if (rd / "metrics.json").exists():
        print(f"[skip] {tag(lam1, lam2)} already complete", flush=True)
        return True
    cmd = ["uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
           "--config", str(build_config(lam1, lam2, epochs, out)), "--seed", str(SEED),
           "--data-root", DATA_ROOT]
    print(f"[run] {tag(lam1, lam2)}: {' '.join(cmd)}", flush=True)
    t0, start = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    rc = subprocess.run(cmd, cwd=REPO).returncode
    wall = {"start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_hours": (time.time() - t0) / 3600, "returncode": rc}
    (out / tag(lam1, lam2) / "wallclock.json").write_text(json.dumps(wall, indent=2))
    if rc != 0:
        print(f"[FAILED] {tag(lam1, lam2)} exited with code {rc}. Stopping (no retry).", flush=True)
        return False
    return True


def epochs_of(rd: Path) -> list[dict]:
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(rd / "metrics_by_epoch.csv"))]


def score(lam1: float, lam2: float, out: Path, summary: dict) -> dict:
    """Selection score + guard for one arm, cached in the summary."""
    key = tag(lam1, lam2)
    if key in summary["arms"]:
        return summary["arms"][key]
    rd = run_dir(out, lam1, lam2)
    metrics = json.loads((rd / "metrics.json").read_text())
    rows = epochs_of(rd)
    best = max(rows, key=lambda r: r["val_graph_f1"])   # first max, as the runner's strict ">"
    collapsed = bool(metrics.get("collapse_abort", {}).get("triggered", False))
    guard_fail = best["val_net_f1"] <= GUARD["min_net_f1"] or \
        best["val_net_class0_frac"] > GUARD["max_class0_frac"]
    arm = {"lambda1": lam1, "lambda2": lam2, "epochs_run": len(rows),
           "early_stopped_collapsed": collapsed, "best_epoch": int(best["epoch"]),
           "fused_f1_k0_checkpoint": best["val_graph_f1"], "net_f1_k0": best["val_net_f1"],
           "net_f1_sigma1e-2": best["val_net_f1_sigma1e-2"],
           "net_class0_frac": best["val_net_class0_frac"], "beta_at_best_epoch": best["beta"],
           "aux_loss_at_best_epoch": best["train_aux"],
           "train_loop_hours": sum(r["train_seconds"] for r in rows) / 3600,
           "guard_valid": not (collapsed or guard_fail)}
    wc = out / key / "wallclock.json"
    if wc.exists():
        arm["wall_hours"] = json.loads(wc.read_text())["wall_hours"]
    sel = lambda_selection_score(rd, DATA_ROOT)
    assert abs(sel["checkpoint_selection_val_f1_at_k0"] - best["val_graph_f1"]) < 1e-12
    arm.update(val_f1_per_k=sel["val_f1_per_k"], val_f1_avg_over_k=sel["val_f1_avg_over_k"],
               beta_final_checkpoint_run=sel["beta_final"])
    summary["arms"][key] = arm
    save(out / "lambda_selection.json", summary)
    print(f"  {key}: best_epoch={arm['best_epoch']} fused_k0={arm['fused_f1_k0_checkpoint']:.4f} "
          f"avg_k={arm['val_f1_avg_over_k']:.4f} net_f1={arm['net_f1_k0']:.4f} "
          f"no-fault={arm['net_class0_frac']:.3f} valid={arm['guard_valid']}", flush=True)
    return arm


def best_valid(summary: dict) -> dict | None:
    valid = [a for a in summary["arms"].values() if a["guard_valid"]]
    if not valid:
        return None
    return max(valid, key=lambda a: (round(a["val_f1_avg_over_k"], 4), -a["lambda1"], -a["lambda2"]))


def decompose(out: Path, summary: dict) -> None:
    """Fusion decomposition at k = 0 on the full validation split, per arm checkpoint_best.

    Method of results/pcstgnn_tuning/fusion_decomposition_val_k0.json: rho(A) from the k = 0
    slice of cache/rtilde_val.npy (clean validation windows, sigma0 = 0; rho has no learned
    parameters), network-only = soft vote of the backbone logits, fused = soft vote of
    z - beta * rho, PSM = argmin rho.
    """
    import torch
    from sklearn.metrics import f1_score
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.models.pc_stgnn import PCSTGNN
    from fault_localization.models.voting import soft_vote
    from fault_localization.physics import residual as R, table as T
    from fault_localization.physics.cache import load_rho_k0_lookup
    from fault_localization.physics.observation import load_norm_stats

    path = out / "fusion_decomposition_val_k0.json"
    res = json.loads(path.read_text())["arms"] if path.exists() else {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WindowGraphDataset(DATA_ROOT, "val")
    tbl = T.load_table("operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    ns = load_norm_stats(DATA_ROOT)
    w2r, rho0, _ = load_rho_k0_lookup(VAL_RHO_CACHE)
    for key, arm in sorted(summary["arms"].items()):
        if key in res:
            continue
        rd = run_dir(out, arm["lambda1"], arm["lambda2"])
        m = yaml.safe_load((rd / "config.yaml").read_text())["model"]
        assert m["sigma0"] == 0.0
        model = PCSTGNN(ds.edge_index, table, ns, use_fusion=True, sigma_eps=m["sigma_eps"],
                        sigma0=m["sigma0"], aggregator="max", num_layers=4).to(dev)
        model.load_state_dict(torch.load(rd / "checkpoint_best.pt", map_location=dev,
                                         weights_only=True)["model"])
        model.eval()
        beta = float(model.beta)
        Y, PN, PP, PF = [], [], [], []
        with torch.no_grad():
            for b in make_dataloader(ds, 32, shuffle=False):
                x = b["x"].to(dev)
                z = model.network_logits(x)
                rho = torch.as_tensor(rho0[[w2r[w] for w in b["window_id"]]], device=dev,
                                      dtype=z.dtype)
                PN.append(soft_vote(z)[1].cpu())
                PF.append(soft_vote(z - beta * rho[:, None, :])[1].cpu())
                PP.append(rho.argmin(-1).cpu())
                Y.append(b["y"])
        y, pn, pp, pf = map(torch.cat, (Y, PN, PP, PF))
        f1 = lambda p: float(f1_score(y, p, average="macro", labels=list(range(26)),  # noqa: E731
                                      zero_division=0))
        mean = lambda t: float(t.float().mean())  # noqa: E731
        r = {"beta": beta, "f1_fused": f1(pf), "f1_network_only": f1(pn), "f1_psm": f1(pp),
             "network_pred_class0": mean(pn == 0)}
        for name, sel in (("fault", y != 0), ("nofault", y == 0)):
            r[name] = {"n_windows": int(sel.sum()),
                       "acc_fused": mean(pf[sel] == y[sel]),
                       "acc_network_only": mean(pn[sel] == y[sel]),
                       "acc_psm": mean(pp[sel] == y[sel]),
                       "fused_eq_psm": mean(pf[sel] == pp[sel]),
                       "fused_eq_network_only": mean(pf[sel] == pn[sel])}
        res[key] = r
        print(f"  decomposition {key}: {json.dumps(r)}", flush=True)
        save(path, {"note": "Full validation split, k=0, clean inputs, checkpoint_best of each "
                    "S4.3e arm. rho(A) from cache/rtilde_val.npy k=0 slice (sigma0=0). "
                    "network_only = soft vote of the backbone logits (physics term removed); "
                    "fused = soft vote of z - beta*rho; psm = argmin rho. acc_* is accuracy "
                    "against the label within each window group.",
                    "n_windows": int(len(y)), "arms": res})


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, lambda_1=0.0333 only, separate output root, no selection")
    args = ap.parse_args()

    if args.smoke:
        out = REPO / "results" / "_smoke_pcstgnn_tuning_aux"
        return 0 if train(0.0333, 0.0, 1, out) else 1

    out = OUT
    summary_path = out / "lambda_selection.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "preregistration_commit": "272ba01", "arms": {},
        "rules": {"checkpoint": "best validation fused macro-F1 at k=0",
                  "selection": "best validation fused macro-F1 averaged over k in {0,1,2,5,10}, "
                               "3 mask seeds, among guard-valid arms; ties -> smaller lambda1, "
                               "then smaller lambda2",
                  "guard": "invalid if net-only F1 <= 0.10 or net-only no-fault share > 0.95 at "
                           "the selected epoch, or early-stopped as collapsed at epoch 3"}}

    for lam1 in LAMBDA1_GRID:
        if not train(lam1, 0.0, EPOCHS, out):
            return 1
        score(lam1, 0.0, out, summary)

    best = best_valid(summary)
    if best is not None and best["lambda1"] in LAMBDA1_EXTENSION and best["lambda2"] == 0.0:
        ext = LAMBDA1_EXTENSION[best["lambda1"]]
        print(f"[extension 1] lambda1={best['lambda1']} is best -> lambda1={ext}", flush=True)
        if not train(ext, 0.0, EPOCHS, out):
            return 1
        score(ext, 0.0, out, summary)
        best = best_valid(summary)
    if best is not None:
        print(f"[extension 2] lambda2={LAMBDA2_EXTENSION} at lambda1={best['lambda1']}", flush=True)
        if not train(best["lambda1"], LAMBDA2_EXTENSION, EPOCHS, out):
            return 1
        score(best["lambda1"], LAMBDA2_EXTENSION, out, summary)
        best = best_valid(summary)

    summary["selected"] = (tag(best["lambda1"], best["lambda2"]) if best else "no selection")
    save(summary_path, summary)
    decompose(out, summary)
    print(f"\nS4.3e selection: {summary['selected']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
