"""S2.3 -- B1: DeltaV nearest-signature baseline (``04_STAGE_C_gonogo.md`` S2.3), on the
**validation** split only (matching S2.4's continuation-instruction scope).

Ds = s - v_nominal, v_nominal = the per-channel no-fault mean measured from the training
split's own pre-fault steps (steps 0-39 of every raw_sim event -- every event, fault or not,
starts with 40 fault-free steps, so this needs no separate no-fault-labelled subset).

"Nearest table signature at nominal load; no profiling over tau, R, or L": unlike B2 (PSM),
which minimises over the full (tau,R,L) nuisance space per candidate f, B1 does a single flat
1-nearest-neighbour search over every (f,tau,R) row of the table AT L = 1.0 (nominal, on-grid)
only -- L is not profiled either. Table rows are stored as **deviations from mu0** (Delta_mu =
mu - mu0 at the same L), so the f=0 (no-fault) row is the zero vector by construction and the
classifier needs no separately-fitted threshold, unlike B1a (run_physics_matched_filter.py's
matched filter, which fits tau on validation).

Reuses ``physics.residual.raw_residual`` with sigma_eps=1 (i.e. plain masked Euclidean distance,
no physical calibration -- B1 does no profiling and is handed no sigma_eps by the spec) on the
Delta-table, so B1 -> B2 is a controlled comparison: same distance machinery, B2 additionally
profiles (tau,R,L) and uses the calibrated sigma_eps.

Usage:
    uv run python scripts/run_b1_deltav_baseline.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import FAULT_TYPES  # noqa: E402
from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.masking import apply_sensor_dropout_batch  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402
from fault_localization.physics.grids import NUM_R, NUM_TYPES, l_grid  # noqa: E402
from fault_localization.physics.residual import flatten_table  # noqa: E402

K_LEVELS = [0, 1, 2, 5, 10]
MASK_SEEDS = [0]  # see run_psm_full_eval.py's note -- B1 has no fitted parameters either
LABELS26 = list(range(26))
OUT_DIR = REPO / "results/pcstgnn/audit/s2_3_b1"
L_NOMINAL = 1.0


def audc(f1_by_k: dict, ks: list[int], relative: bool) -> float:
    order = sorted(ks)
    vals = (np.array([f1_by_k[k] / f1_by_k[order[0]] for k in order]) if relative
            else np.array([f1_by_k[k] for k in order]))
    karr = np.array(order, dtype=float)
    return float(np.trapezoid(vals, karr) / (karr[-1] - karr[0]))


def build_delta_table(mu: np.ndarray, mu0: np.ndarray) -> torch.Tensor:
    """[26, NUM_TYPES*NUM_R, 75] table of (mu - mu0) at L=L_NOMINAL, row 0 = zero vector."""
    l_values = l_grid(0.05)  # matches operator/grid.json's l_step
    l_idx = int(np.argmin(np.abs(l_values - L_NOMINAL)))
    assert abs(l_values[l_idx] - L_NOMINAL) < 1e-9, "L=1.0 must sit exactly on the grid"
    mu_sub = mu[:, :, :, l_idx:l_idx + 1, :, :]
    mu0_sub = mu0[l_idx:l_idx + 1, :, :]
    full = flatten_table(mu_sub, mu0_sub)  # [26, NUM_TYPES*NUM_R, 75], row 0 = mu0 broadcast
    delta = full - full[0:1]               # subtract mu0 from every row -> row 0 becomes 0
    return delta


def eval_b1(data_root: Path, delta_table: torch.Tensor, v_nominal: np.ndarray,
            device: torch.device, batch_size: int = 4096) -> dict:
    norm_stats = json.loads((data_root / "processed/norm_stats.json").read_text())
    mean, std = np.asarray(norm_stats["mean"]), np.asarray(norm_stats["std"])
    # Time-budgeted 30,000-window random subsample of the 248,000-window val split
    # (splits/val_sample.txt, seed 0) -- S2.4/PSM already covers the full population;
    # B1 is the secondary classical-method comparison and does not need it repeated.
    val_ds = WindowGraphDataset(str(data_root), "val_sample")
    loader = make_dataloader(val_ds, batch_size, shuffle=False)
    v_nom_t = torch.as_tensor(v_nominal, dtype=torch.float64, device=device).reshape(25, 3)

    f1_by_k, by_r, by_type = {}, {}, {}
    rows = []
    for k in K_LEVELS:
        seeds = [0] if k == 0 else MASK_SEEDS
        seed_f1s = []
        for ms in seeds:
            preds, trues, ress, types = [], [], [], []
            for bi, batch in enumerate(loader):
                x, y = batch["x"].to(device), batch["y"]
                if k > 0:
                    x = apply_sensor_dropout_batch(x, k=k, seed=ms * 10000 + bi)[0]
                s, a = OBS.observe(x, mean, std, reduce="last")
                s, a = s.double(), a.double()
                ds = s - v_nom_t[None]
                r, _hyp = R.raw_residual(ds, a, delta_table, sigma_eps=1.0, sigma0=0.0,
                                          method="gram")
                pred = r.argmin(dim=-1)
                preds.extend(pred.tolist()); trues.extend(y.tolist())
                if ms == seeds[0]:
                    ress.extend(float(m["resistance"]) for m in batch["metadata"])
                    types.extend(m["fault_type"] for m in batch["metadata"])
                if bi % 15 == 0:
                    print(f"  [B1 k={k} ms={ms}] batch {bi}/{len(loader)}", flush=True)
            f1 = f1_score(trues, preds, average="macro", labels=LABELS26, zero_division=0)
            seed_f1s.append(f1)
            rows.append({"k": k, "mask_seed": ms, "f1_macro26": f1})
            if ms == seeds[0] and k == 0:
                trues_a, preds_a = np.array(trues), np.array(preds)
                ress_a, types_a = np.array(ress), np.array(types)
                for rv in sorted(set(ress_a.tolist())):
                    sel = ress_a == rv
                    by_r[rv] = f1_score(trues_a[sel], preds_a[sel], average="macro",
                                         labels=LABELS26, zero_division=0)
                for tv in FAULT_TYPES:
                    sel = types_a == tv
                    if sel.sum() > 0:
                        by_type[tv] = f1_score(trues_a[sel], preds_a[sel], average="macro",
                                                labels=LABELS26, zero_division=0)
        f1_by_k[k] = float(np.mean(seed_f1s))
        print(f"[B1 k={k}] macro-F1={f1_by_k[k]:.4f}", flush=True)
        (OUT_DIR / "b1_checkpoint_partial.json").write_text(
            json.dumps({"f1_by_k_partial": f1_by_k, "raw_rows": rows}, indent=2))

    return {
        "f1_by_k": f1_by_k, "f1_by_resistance_k0": by_r, "f1_by_fault_type_k0": by_type,
        "audc_relative": audc(f1_by_k, K_LEVELS, relative=True),
        "audc_absolute": audc(f1_by_k, K_LEVELS, relative=False),
        "raw_rows": rows,
    }


def compute_v_nominal(data_root: Path) -> np.ndarray:
    """Per-channel (75) no-fault mean over the training split's pre-fault steps (0-39)."""
    stems = set()
    with (data_root / "splits/train.txt").open() as fh:
        for line in fh:
            stems.add(line.strip().rsplit("_w", 1)[0])
    raw_dir = data_root / "raw_sim"
    paths = sorted(p for p in raw_dir.glob("*.npz") if p.stem in stems)
    total = np.zeros((25, 3), dtype=np.float64)  # [node, phase] -- kept unflattened until the end
    count = 0
    for p in paths:
        with np.load(p, allow_pickle=True) as loaded:
            rec = loaded["data"]
        pre = rec[:, :40, :3]              # [25 node, 40 pre-fault step, 3 phase]
        total += pre.sum(axis=1)           # sum over time only, keep node/phase axes intact
        count += pre.shape[1]
    return (total / count).reshape(75)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    t0 = time.time()
    data_root = Path(args.data_root)
    v_nom_path = OUT_DIR / "v_nominal.npy"
    if v_nom_path.exists():
        v_nominal = np.load(v_nom_path)
        print(f"[v_nominal] cached ({time.time()-t0:.1f}s)", flush=True)
    else:
        v_nominal = compute_v_nominal(data_root)
        np.save(v_nom_path, v_nominal)
        print(f"[v_nominal] computed, shape={v_nominal.shape}, "
              f"mean={v_nominal.mean():.4f} ({time.time()-t0:.1f}s)", flush=True)

    tbl = T.load_table(REPO / "operator")
    delta_table = build_delta_table(tbl["mu"], tbl["mu0"])
    print(f"[table] delta_table {tuple(delta_table.shape)}  device={device}", flush=True)

    result = eval_b1(data_root, delta_table, v_nominal, device)

    b1a_path = REPO / "results/physics_baseline/matched_filter.json"
    b1a = json.loads(b1a_path.read_text()) if b1a_path.exists() else None

    summary = {"b1_delta_v": result, "b1a_matched_filter": b1a, "wall_seconds": time.time() - t0}
    (OUT_DIR / "b1_full_eval.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'b1_full_eval.json'} ({summary['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
