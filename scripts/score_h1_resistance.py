"""Gate 4 H1 check: re-score the 6 Gate 4 checkpoints at k=0, sigma=0, by resistance
(PC-STGNN/S4.4_remaining_arms_plan.md Part 4 step 2; `14_PREREGISTRATION.md` H1: "Physics fusion
improves high-impedance localization | R5 vs R0 | 10 Ohm, k=0 | F1 by R | +").

H1 was never evaluated by the original Gate 4 scoring grid: `scoring_full_grid.json` reports F1
pooled over all three training resistances, with no per-R breakdown (confirmed in
`06_STAGE_E_pcstgnn.md`'s "GATE 4 -- corrected record"). This script does **one streaming pass**
over the validation split at k=0, sigma=0 only (no dropout, no noise -- H1's own condition), for
each of the 6 already-trained-and-scored Gate 4 checkpoints, and reports fused F1 (+ for R5, the
network-only/PSM decomposition) pooled and split by resistance (0.1/1.0/10.0 ohm).

Deliberately narrow in scope, not a re-run of `score_s4_4_gate4_run.py::score_run`: that scores
the full 20-cell grid x 3 mask-seed offsets (60 passes/checkpoint), which measured ~7.44 h/run
for R5 and would cost ~26 GPU-h across all 6 checkpoints for information H1 does not need. This
script does 1 pass/checkpoint. No training occurs -- only forward passes on frozen checkpoints.

Writes results/s4_4/<arm>/pcstgnn/seed_<seed:05d>/scoring_h1.json, alongside (not replacing)
the existing scoring_full_grid.json from Gate 4.

Usage:
    uv run python scripts/score_h1_resistance.py                  # all 6 Gate 4 checkpoints
    uv run python scripts/score_h1_resistance.py --max-windows 960 # timing check
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.models.voting import soft_vote
from score_s4_4_gate4_run import (  # noqa: E402 -- reuse, don't reimplement
    DATA_ROOT, RESISTANCES, _load_run_config, f1_by_resistance, macro_f1,
)
from fault_localization.experiments.runner import _model as build_model_from_config

GATE4_PAIRS = [("r0n", 42), ("r0n", 43), ("r0n", 44), ("r5", 42), ("r5", 43), ("r5", 44)]


def score_h1(arm: str, seed: int, max_windows: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("results/s4_4") / arm / "pcstgnn" / f"seed_{seed:05d}"
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    run_config = _load_run_config(run_dir)
    data_root = run_config.get("data_root", DATA_ROOT)
    val_ds = WindowGraphDataset(data_root, "val")
    edge_index = val_ds.edge_index
    window_to_resistance = {wid: float(md["resistance"]) for wid, *_rest, md in val_ds.records}

    model = build_model_from_config(run_config, edge_index).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(state)
    model.eval()
    use_fusion = bool(getattr(model, "use_fusion", False))
    has_table = bool(getattr(model, "phys_table", torch.empty(0)).numel())
    beta = float(model.beta) if use_fusion else None

    loader = make_dataloader(val_ds, 32, shuffle=False)
    pf_all, pn_all, pp_all, y_all, r_all = [], [], [], [], []
    n_seen = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if max_windows is not None and n_seen >= max_windows:
                break
            x = batch["x"]
            wids = batch["window_id"]
            y = batch["y"].tolist()
            if max_windows is not None:
                keep = max_windows - n_seen
                if keep < len(wids):
                    x, wids, y = x[:keep], wids[:keep], y[:keep]
            x = x.to(device)
            n_seen += x.shape[0]
            y_all.extend(y)
            r_all.extend(window_to_resistance[w] for w in wids)

            if use_fusion:
                z = model.network_logits(x)
                rho, _, _, _ = model._rho(x, sigma0=0.0)
                pf_all.extend(soft_vote(z - beta * rho[:, None, :])[1].cpu().tolist())
                pn_all.extend(soft_vote(z)[1].cpu().tolist())
                pp_all.extend(rho.argmin(-1).cpu().tolist())
            elif has_table:
                z = model.network_logits(x)
                rho, _, _, _ = model._rho(x, sigma0=0.0)
                pf = soft_vote(z)[1]
                pf_all.extend(pf.cpu().tolist())
                pn_all.extend(pf.cpu().tolist())
                pp_all.extend(rho.argmin(-1).cpu().tolist())
            else:
                pf_all.extend(soft_vote(model(x))[1].cpu().tolist())
    elapsed_s = time.perf_counter() - t0

    y_true = np.asarray(y_all)
    resistance_arr = np.asarray(r_all)
    pf = np.asarray(pf_all)
    result = {
        "arm": arm, "seed": seed, "use_fusion": use_fusion, "has_table": has_table, "beta": beta,
        "n_windows": n_seen, "k": 0, "sigma": 0.0,
        "f1_fused_pooled": macro_f1(y_true, pf),
        "f1_fused_by_resistance": f1_by_resistance(y_true, pf, resistance_arr),
        "scoring_wallclock_seconds": elapsed_s,
    }
    if has_table:
        pn = np.asarray(pn_all)
        pp = np.asarray(pp_all)
        result["f1_network_only_pooled"] = macro_f1(y_true, pn)
        result["f1_network_only_by_resistance"] = f1_by_resistance(y_true, pn, resistance_arr)
        result["f1_psm_pooled"] = macro_f1(y_true, pp)
        result["f1_psm_by_resistance"] = f1_by_resistance(y_true, pp, resistance_arr)
        result["fused_eq_network_only"] = float(np.mean(pf == pn))
        result["fused_eq_psm"] = float(np.mean(pf == pp))

    if max_windows is None:
        (run_dir / "scoring_h1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()
    for arm, seed in GATE4_PAIRS:
        t0 = time.perf_counter()
        r = score_h1(arm, seed, max_windows=args.max_windows)
        dt = time.perf_counter() - t0
        print(f"{arm} seed {seed}: f1_fused={r['f1_fused_pooled']:.4f} "
              f"f1_by_R={{{', '.join(f'{k}: {v:.4f}' for k, v in r['f1_fused_by_resistance'].items())}}} "
              f"({dt:.1f}s wall)", flush=True)


if __name__ == "__main__":
    main()
