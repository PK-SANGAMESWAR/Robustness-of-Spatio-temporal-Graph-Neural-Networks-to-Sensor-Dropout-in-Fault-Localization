"""S2.4 -- B2: PSM (Physics Signature Matching) full evaluation
(``04_STAGE_C_gonogo.md`` S2.4), on the **validation** split and on ``data/ood_pilot/`` only
(per the author's 2026-09-25 continuation instruction -- test and the sealed OOD set stay
untouched). f_hat = argmin_f rho_f(A), zero learned parameters, canonical frozen operator
table + sigma_eps=7.2555e-4 (neither touched here).

Validation-split evaluation goes through ``WindowGraphDataset`` (matching every learned model's
protocol exactly, and giving F1-by-R / F1-by-type / AUDC via ``processed/metadata.csv``).
OOD-pilot has no processed/windowed data (out of scope for this script -- physics does not need
it): predictions come straight from ``physics.observation.observe_raw`` on ``raw_sim``, the same
raw-corpus path ``physics/cache.py`` uses, generating the 40 per-event window labels directly
(steps 0-19 no-fault, 20-39 fault, ``processing/labels.py:79``).

Usage:
    uv run python scripts/run_psm_full_eval.py
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

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES, WINDOW_LENGTH  # noqa: E402
from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.masking import apply_sensor_dropout_batch  # noqa: E402
from fault_localization.physics import observation as OBS  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
K_LEVELS = [0, 1, 2, 5, 10]
# Single mask seed per k, not the usual 3: PSM has zero learned/fitted parameters (no training-
# seed variance to average out the way a network run does), and the "direct" residual method is
# too slow for a 248k-window validation split at 3x the mask-seed cost within this pass's time
# budget (04_STAGE_C_gonogo.md S2.4 has no seed-count requirement of its own). Documented here,
# not hidden: any reported PSM number in this file is a single-mask-seed point, not a mean+-std.
MASK_SEEDS = [0]
LABELS26 = list(range(26))
OUT_DIR = REPO / "results/pcstgnn/audit/s2_4_psm"
RESIDUAL_METHOD = "gram"  # float64 matmul form -- ~800x faster than "direct" at this table size


def audc(f1_by_k: dict[int, float], ks: list[int], relative: bool) -> float:
    order = sorted(ks)
    if relative:
        base = f1_by_k[order[0]]
        vals = np.array([f1_by_k[k] / base for k in order])
    else:
        vals = np.array([f1_by_k[k] for k in order])
    karr = np.array(order, dtype=float)
    return float(np.trapezoid(vals, karr) / (karr[-1] - karr[0]))


def rank_of_true(rho_vals: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0-indexed rank of the true class under ascending rho (0 = correct)."""
    order = rho_vals.argsort(dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    return ranks.gather(1, y[:, None]).squeeze(1)


# ---------------------------------------------------------------------------
# Validation split (WindowGraphDataset)
# ---------------------------------------------------------------------------

def eval_val_split(data_root: Path, table: torch.Tensor, device: torch.device,
                    batch_size: int = 4096, checkpoint_path: Path | None = None) -> dict:
    norm_stats = json.loads((data_root / "processed/norm_stats.json").read_text())
    mean, std = np.asarray(norm_stats["mean"]), np.asarray(norm_stats["std"])
    val_ds = WindowGraphDataset(str(data_root), "val")
    loader = make_dataloader(val_ds, batch_size, shuffle=False)

    rows = []
    f1_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    top3_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    rank_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    by_r: dict[float, list[float]] = {}
    by_type: dict[str, list[float]] = {}

    for k in K_LEVELS:
        seeds = [0] if k == 0 else MASK_SEEDS
        seed_f1s = []
        for ms in seeds:
            preds, trues, ress, types = [], [], [], []
            top3_hits, ranks = [], []
            for bi, batch in enumerate(loader):
                x, y = batch["x"].to(device), batch["y"].to(device)
                if k > 0:
                    x = apply_sensor_dropout_batch(x, k=k, seed=ms * 10000 + bi)[0]
                s, a = OBS.observe(x, mean, std, reduce="last")
                s, a = s.double(), a.double()
                rho, _hyp = R.rho(s, a, table, CANONICAL_SIGMA_EPS, sigma0=0.0, method=RESIDUAL_METHOD)
                pred = R.psm_predict(rho)
                top3 = rho.topk(3, dim=-1, largest=False).indices
                top3_hit = (top3 == y[:, None]).any(dim=-1).float()
                rk = rank_of_true(rho, y)
                preds.extend(pred.tolist()); trues.extend(y.tolist())
                top3_hits.extend(top3_hit.tolist()); ranks.extend(rk.tolist())
                if ms == seeds[0]:
                    res_batch = [float(m["resistance"]) for m in batch["metadata"]]
                    type_batch = [m["fault_type"] for m in batch["metadata"]]
                    ress.extend(res_batch); types.extend(type_batch)
                if bi % 15 == 0:
                    print(f"  [val k={k} ms={ms}] batch {bi}/{len(loader)}", flush=True)
            f1 = f1_score(trues, preds, average="macro", labels=LABELS26, zero_division=0)
            seed_f1s.append(f1)
            top3_by_k[k].append(float(np.mean(top3_hits)))
            rank_by_k[k].append(float(np.mean(ranks)))
            rows.append({"k": k, "mask_seed": ms, "f1_macro26": f1,
                         "top3_acc": float(np.mean(top3_hits)),
                         "mean_rank": float(np.mean(ranks))})
            if ms == seeds[0] and k == 0:
                trues_a = np.array(trues); preds_a = np.array(preds)
                ress_a = np.array(ress); types_a = np.array(types)
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
        print(f"[val k={k}] macro-F1={f1_by_k[k]:.4f} top3={np.mean(top3_by_k[k]):.4f} "
              f"mean_rank={np.mean(rank_by_k[k]):.3f}", flush=True)
        if checkpoint_path is not None:
            partial = {"f1_by_k_partial": f1_by_k, "top3_by_k_partial":
                       {kk: float(np.mean(v)) for kk, v in top3_by_k.items() if v},
                       "raw_rows": rows}
            checkpoint_path.write_text(json.dumps(partial, indent=2, default=str))

    result = {
        "f1_by_k": f1_by_k,
        "top3_by_k": {k: float(np.mean(v)) for k, v in top3_by_k.items()},
        "mean_rank_by_k": {k: float(np.mean(v)) for k, v in rank_by_k.items()},
        "f1_by_resistance_k0": by_r,
        "f1_by_fault_type_k0": by_type,
        "audc_relative": audc(f1_by_k, K_LEVELS, relative=True),
        "audc_absolute": audc(f1_by_k, K_LEVELS, relative=False),
        "raw_rows": rows,
    }
    return result


# ---------------------------------------------------------------------------
# OOD-pilot (raw_sim only, no processed/ dataset)
# ---------------------------------------------------------------------------

def eval_ood_pilot(pilot_dir: Path, table: torch.Tensor, device: torch.device,
                    batch_events: int = 32) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_dir = pilot_dir / "raw_sim"
    paths = sorted(raw_dir.glob("*.npz"))
    print(f"[ood-pilot] {len(paths)} events", flush=True)

    f1_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    top3_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    rank_by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}

    events = []
    for p in paths:
        with np.load(p, allow_pickle=True) as loaded:
            record = loaded["data"]
            meta = json.loads(str(loaded["meta"]))
        events.append((record, FAULT_BUSES.index(meta["fault_bus"]) + 1))

    for k in K_LEVELS:
        seeds = [0] if k == 0 else MASK_SEEDS
        seed_f1s = []
        for ms in seeds:
            preds, trues, top3_hits, ranks = [], [], [], []
            batch_s, batch_a, batch_y = [], [], []
            bi = 0
            for ei, (record, fault_bus_class) in enumerate(events):
                for w in range(40):
                    label = 0 if w < 20 else fault_bus_class
                    s, a = OBS.observe_raw(record, w, WINDOW_LENGTH, reduce="last")
                    batch_s.append(s); batch_a.append(a); batch_y.append(label)
                if len(batch_s) >= batch_events * 40 or ei == len(events) - 1:
                    s_t = torch.from_numpy(np.stack(batch_s).astype(np.float64)).to(device)
                    a_t = torch.from_numpy(np.stack(batch_a).astype(np.float64)).to(device)
                    y_t = torch.tensor(batch_y, dtype=torch.long, device=device)
                    if k > 0:
                        # dropout at the node level directly on availability -- physics only
                        # ever consumes (s,a), so this matches zero_messages' effect exactly.
                        g = torch.Generator().manual_seed(ms * 10000 + bi)
                        for row in range(a_t.shape[0]):
                            drop = torch.randperm(25, generator=g)[:k].to(device)
                            a_t[row, drop, :] = 0.0
                    rho, _ = R.rho(s_t, a_t, table, CANONICAL_SIGMA_EPS, sigma0=0.0, method=RESIDUAL_METHOD)
                    pred = R.psm_predict(rho)
                    top3 = rho.topk(3, dim=-1, largest=False).indices
                    top3_hit = (top3 == y_t[:, None]).any(dim=-1).float()
                    rk = rank_of_true(rho, y_t)
                    preds.extend(pred.tolist()); trues.extend(y_t.tolist())
                    top3_hits.extend(top3_hit.tolist()); ranks.extend(rk.tolist())
                    batch_s, batch_a, batch_y = [], [], []
                    bi += 1
            f1 = f1_score(trues, preds, average="macro", labels=LABELS26, zero_division=0)
            seed_f1s.append(f1)
            top3_by_k[k].append(float(np.mean(top3_hits)))
            rank_by_k[k].append(float(np.mean(ranks)))
        f1_by_k[k] = float(np.mean(seed_f1s))
        print(f"[ood-pilot k={k}] macro-F1={f1_by_k[k]:.4f} top3={np.mean(top3_by_k[k]):.4f}",
              flush=True)

    return {
        "n_events": len(paths),
        "f1_by_k": f1_by_k,
        "top3_by_k": {k: float(np.mean(v)) for k, v in top3_by_k.items()},
        "mean_rank_by_k": {k: float(np.mean(v)) for k, v in rank_by_k.items()},
        "audc_relative": audc(f1_by_k, K_LEVELS, relative=True),
        "audc_absolute": audc(f1_by_k, K_LEVELS, relative=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--ood-pilot-dir", default=str(REPO / "data/ood_pilot"))
    ap.add_argument("--k-levels", type=int, nargs="+", default=None,
                     help="override K_LEVELS, e.g. to resume only the missing k's")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--skip-val", action="store_true",
                     help="reuse the cached val_checkpoint_partial_k*.json f1_by_k instead of "
                          "recomputing (by_r/by_type are lost from that cache and reported null)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.k_levels is not None:
        global K_LEVELS
        K_LEVELS = args.k_levels

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    from fault_localization.physics.residual import flatten_table
    tbl = T.load_table(REPO / "operator")
    table = flatten_table(tbl["mu"], tbl["mu0"])
    print(f"[load] table {tuple(table.shape)} device={device} ({time.time()-t0:.1f}s)", flush=True)

    if args.skip_val:
        cached = json.loads((OUT_DIR / f"val_checkpoint_partial_k{'_'.join(map(str,K_LEVELS))}.json").read_text())
        f1_by_k = {int(k): v for k, v in cached["f1_by_k_partial"].items()}
        val_result = {
            "f1_by_k": f1_by_k,
            "top3_by_k": {int(k): v for k, v in cached.get("top3_by_k_partial", {}).items()},
            "f1_by_resistance_k0": None, "f1_by_fault_type_k0": None,
            "note": "reused from interrupted-run cache; by_r/by_type not preserved by that cache",
            "audc_relative": audc(f1_by_k, K_LEVELS, relative=True),
            "audc_absolute": audc(f1_by_k, K_LEVELS, relative=False),
            "raw_rows": cached.get("raw_rows", []),
        }
        print(f"[val] reused cached f1_by_k={f1_by_k}", flush=True)
    else:
        val_result = eval_val_split(Path(args.data_root), table, device,
                                     checkpoint_path=OUT_DIR / f"val_checkpoint_partial_k{'_'.join(map(str,K_LEVELS))}.json")
    ood_result = (eval_ood_pilot(Path(args.ood_pilot_dir), table, device) if not args.skip_ood
                  else {"skipped": True})

    summary = {
        "sigma_eps": CANONICAL_SIGMA_EPS, "k_levels": K_LEVELS,
        "validation": val_result, "ood_pilot": ood_result,
        "wall_seconds": time.time() - t0,
    }
    (OUT_DIR / "psm_full_eval.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'psm_full_eval.json'} ({summary['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
