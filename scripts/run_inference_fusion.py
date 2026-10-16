"""S2.6 -- inference-only fusion (``04_STAGE_C_gonogo.md`` S2.6), CPU only, validation split
only, on the existing R0 checkpoint (dropout-aware RGSAGE-Max, ``06_STAGE_E_pcstgnn.md``'s own
R0 definition) -- no retraining.

z'_{n,f} = z_{n,f} - beta * rho_f(A)   (``01_SPEC.md`` Sec.6, same rho broadcast to every node n)

For each condition (clean k=0; k=10; sigma0 in {5e-4, 2e-3, 1e-2} at k=0), one forward pass over
the validation split produces (z [W,25,26], rho [W,26], y [W]) ONCE; beta is then grid-searched
analytically on those cached tensors (no repeated model passes -- this is what keeps a multi-beta
sweep on 248k CPU-bound windows affordable). ``10 Ohm`` is read off the clean-k0 condition's own
cached metadata, not a separate pass.

Seed budget: seed 42 only, matching this project's own established convention for reduced-seed
exploratory/secondary checks (``02_STAGE_A_audit.md`` S0.5: "1 seed for R1, R3, R7, G1, G3,
B2a") -- a 3-seed confirmation is S2.7/pre-registration follow-up work, explicitly out of scope
for this pass.

Usage:
    uv run python scripts/run_inference_fusion.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch
from sklearn.metrics import f1_score

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.masking import apply_sensor_dropout_batch  # noqa: E402
from fault_localization.dropout.noise import apply_measurement_noise_batch  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.physics.residual import flatten_table  # noqa: E402
from run_dropout_study import load_trained_model  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
LABELS26 = list(range(26))
BETA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEED = 42
OUT_DIR = REPO / "results/pcstgnn/audit/s2_6_fusion"
CONDITIONS = [
    {"name": "clean_k0", "k": 0, "sigma0": 0.0},
    {"name": "k10", "k": 10, "sigma0": 0.0},
    {"name": "sigma5e-4", "k": 0, "sigma0": 5e-4},
    {"name": "sigma2e-3", "k": 0, "sigma0": 2e-3},
    {"name": "sigma1e-2", "k": 0, "sigma0": 1e-2},
]


def cache_condition(model, loader, mean, std, table, k, sigma0, device):
    """One forward pass -> (z [W,25,26], rho [W,26], y [W], resistance [W])."""
    zs, rhos, ys, res = [], [], [], []
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x, y = batch["x"].to(device), batch["y"]
            if k > 0:
                x = apply_sensor_dropout_batch(x, k=k, seed=10000 * k + bi)[0]
            if sigma0 > 0:
                x = apply_measurement_noise_batch(x, sigma0, std, seed=bi)
            z = model(x)  # [B,25,26]
            s, a = OBS.observe(x, mean, std, reduce="last")
            s, a = s.double(), a.double()
            rho, _ = R.rho(s, a, table, CANONICAL_SIGMA_EPS, sigma0=sigma0, method="gram")
            zs.append(z.cpu()); rhos.append(rho.float().cpu()); ys.append(y)
            res.extend(float(m["resistance"]) for m in batch["metadata"])
    return (torch.cat(zs), torch.cat(rhos), torch.cat(ys).numpy(), np.array(res))


def f1_for_beta(z, rho, y, beta):
    z_prime = z - beta * rho[:, None, :]
    scores = torch.softmax(z_prime, dim=-1).sum(dim=1)
    pred = scores.argmax(dim=-1).numpy()
    return f1_score(y, pred, average="macro", labels=LABELS26, zero_division=0)


def grid_search_beta(z, rho, y):
    f1s = [f1_for_beta(z, rho, y, b) for b in BETA_GRID]
    best_i = int(np.argmax(f1s))
    return BETA_GRID[best_i], f1s[best_i], f1s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data_root = Path(args.data_root)
    norm_stats = json.loads((data_root / "processed/norm_stats.json").read_text())
    mean, std = np.asarray(norm_stats["mean"]), np.asarray(norm_stats["std"])
    # Same time-budgeted 30,000-window subsample as run_b1_deltav_baseline.py (splits/
    # val_sample.txt, seed 0) -- S2.6 needs a full network forward pass per condition (5
    # conditions), which is the expensive part; the beta grid search on top is nearly free.
    val_ds = WindowGraphDataset(str(data_root), "val_sample")
    loader = make_dataloader(val_ds, batch_size=256, shuffle=False)

    tbl = T.load_table(REPO / "operator")
    table = flatten_table(tbl["mu"], tbl["mu0"])

    seed_dir = REPO / f"results/dropout_aware/rgsage_max/seed_{SEED:05d}"
    model, cfg = load_trained_model(seed_dir, val_ds.edge_index, device)

    checkpoint_path = OUT_DIR / "fusion_checkpoint_partial.json"
    results = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    clean_res = None
    for cond in CONDITIONS:
        if cond["name"] in results:
            print(f"[cache] {cond['name']} already done, skipping", flush=True)
            continue
        print(f"[cache] {cond['name']} ...", flush=True)
        z, rho, y, res = cache_condition(model, loader, mean, std, table,
                                          cond["k"], cond["sigma0"], device)
        if cond["name"] == "clean_k0":
            clean_res = res
        best_beta, best_f1, all_f1s = grid_search_beta(z, rho, y)
        base_f1 = f1_for_beta(z, rho, y, 0.0)
        entry = {
            "n_windows": len(y), "best_beta": best_beta, "best_f1": best_f1,
            "baseline_f1_beta0": base_f1, "delta_f1": best_f1 - base_f1,
            "f1_by_beta": dict(zip(map(str, BETA_GRID), all_f1s)),
        }
        if cond["name"] == "clean_k0":
            sel10 = res == 10.0
            if sel10.sum() > 0:
                b10, f10, _ = grid_search_beta(z[sel10], rho[sel10], y[sel10])
                base10 = f1_for_beta(z[sel10], rho[sel10], y[sel10], 0.0)
                entry["at_10ohm"] = {"n": int(sel10.sum()), "best_beta": b10,
                                      "best_f1": f10, "baseline_f1_beta0": base10,
                                      "delta_f1": f10 - base10}
        results[cond["name"]] = entry
        print(f"  beta*={best_beta}  F1(beta*)={best_f1:.4f}  F1(beta=0)={base_f1:.4f}  "
              f"deltaF1={entry['delta_f1']:+.4f}", flush=True)
        (OUT_DIR / "fusion_checkpoint_partial.json").write_text(json.dumps(results, indent=2))

    summary = {"seed": SEED, "beta_grid": BETA_GRID, "conditions": results}
    (OUT_DIR / "fusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'fusion_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
