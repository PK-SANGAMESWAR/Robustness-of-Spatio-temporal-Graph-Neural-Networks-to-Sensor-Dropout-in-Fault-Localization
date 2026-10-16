"""S2.5 -- the extrapolation contrast (``04_STAGE_C_gonogo.md`` S2.5), pilot-only, per the
author's 2026-09-25 continuation instruction: CPU only, ``data/ood_pilot/`` only, never
``data/ood_test/``.

Evaluate the existing dropout-aware RGSAGE-Max checkpoints (seeds 42/43/44,
``results/dropout_aware/rgsage_max/seed_000{seed}/``) on the OOD-pilot (R in {5,20} Ohm, never
seen in training) and compare the F1 drop (in-distribution -> OOD) against PSM's own drop from
S2.4 (``results/pcstgnn/audit/s2_4_psm/psm_full_eval.json``).

OOD-pilot has no ``processed/`` dataset (S2.0/S2.4 only touched raw_sim); this script builds one
via the standard pipeline (``processing.build_dataset.build``, deterministic given the same
feeder), then overwrites its ``norm_stats.json`` with the CORPUS's own (never recomputes
normalization from the 1,100-event pilot -- the trained checkpoints were fit against the corpus's
mean/std, and evaluating them under different statistics would not test the network, it would
test a renormalization artifact). A synthetic ``splits/all.txt`` (every pilot window) replaces
the meaningless 70/15/15 split build_dataset always produces for an eval-only corpus.

In-distribution reference: ``results/dropout_study_aware/degradation_curves.csv`` (existing,
3-seed, standard test split) for rgsage_max, NOT re-evaluated here.

Usage:
    uv run python scripts/run_extrapolation_contrast.py
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
import pandas as pd
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.evaluator import evaluate_dropout_curve  # noqa: E402
from run_dropout_study import load_trained_model  # noqa: E402

K_LEVELS = [0, 1, 2, 5, 10]
N_MASK_SEEDS = 1  # time-budgeted, like S2.4's PSM pass -- see that script's note
SEEDS = [42, 43, 44]
OUT_DIR = REPO / "results/pcstgnn/audit/s2_5_extrapolation"


def ensure_pilot_processed(pilot_dir: Path, corpus_root: Path) -> None:
    processed = pilot_dir / "processed"
    if (processed / "norm_stats.json").exists() and (pilot_dir / "splits/all.txt").exists():
        print("[pilot] processed/ already built, reusing", flush=True)
        return
    from fault_localization.processing.build_dataset import build
    build(pilot_dir)

    # Overwrite with the CORPUS's own norm stats -- see module docstring.
    corpus_stats = json.loads((corpus_root / "processed/norm_stats.json").read_text())
    (processed / "norm_stats.json").write_text(json.dumps(corpus_stats))

    # Build an "all" split covering every pilot window, replacing the meaningless
    # 70/15/15 split build_dataset() always produces.
    import csv
    ids = [r["window_id"] for r in csv.DictReader((processed / "labels.csv").open())]
    (pilot_dir / "splits" / "all.txt").write_text("\n".join(ids) + "\n")
    print(f"[pilot] built processed/ ({len(ids)} windows), norm_stats overridden from corpus",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", default=str(REPO / "data/ood_pilot"))
    ap.add_argument("--corpus-root", default=str(REPO / "dataset_fullscale_50"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_dir = Path(args.pilot_dir)
    corpus_root = Path(args.corpus_root)
    ensure_pilot_processed(pilot_dir, corpus_root)

    device = torch.device(args.device)
    pilot_ds = WindowGraphDataset(str(pilot_dir), "all")
    loader = make_dataloader(pilot_ds, batch_size=256, shuffle=False)
    print(f"[pilot] {len(pilot_ds)} windows loaded", flush=True)

    per_seed = {}
    for seed in SEEDS:
        seed_dir = REPO / f"results/dropout_aware/rgsage_max/seed_{seed:05d}"
        model, cfg = load_trained_model(seed_dir, pilot_ds.edge_index, device)
        curve = evaluate_dropout_curve(model, loader, device, dropout_levels=K_LEVELS,
                                        n_mask_seeds=N_MASK_SEEDS, policy="zero_messages",
                                        model_name=f"rgsage_max_seed{seed}")
        per_seed[seed] = {int(k): v["f1_mean"] for k, v in curve.items()}
        print(f"[seed {seed}] OOD-pilot F1 by k: {per_seed[seed]}", flush=True)
        (OUT_DIR / "extrapolation_checkpoint_partial.json").write_text(
            json.dumps(per_seed, indent=2))

    ood_f1_by_k = {k: float(np.mean([per_seed[s][k] for s in SEEDS])) for k in K_LEVELS}
    ood_f1_std_by_k = {k: float(np.std([per_seed[s][k] for s in SEEDS], ddof=1)) for k in K_LEVELS}

    # In-distribution reference (existing 3-seed curve, standard test split)
    id_curve_path = REPO / "results/dropout_study_aware/degradation_curves.csv"
    id_f1_by_k = {}
    if id_curve_path.exists():
        df = pd.read_csv(id_curve_path)
        sub = df[df["model"] == "rgsage_max"] if "model" in df.columns else df
        for k in K_LEVELS:
            row = sub[sub["k"] == k]
            if len(row):
                id_f1_by_k[k] = float(row.iloc[0]["f1_mean"])

    # PSM comparison from S2.4
    psm_path = REPO / "results/pcstgnn/audit/s2_4_psm/psm_full_eval.json"
    psm = json.loads(psm_path.read_text()) if psm_path.exists() else None

    result = {
        "network": "dropout_aware_rgsage_max",
        "seeds": SEEDS, "n_mask_seeds": N_MASK_SEEDS,
        "ood_pilot_f1_by_k": ood_f1_by_k, "ood_pilot_f1_std_by_k": ood_f1_std_by_k,
        "in_distribution_f1_by_k": id_f1_by_k,
        "network_drop_k0": (id_f1_by_k.get(0, float("nan")) - ood_f1_by_k[0]),
    }
    if psm is not None:
        psm_val_k0 = psm["validation"]["f1_by_k"]["0"]
        psm_ood_k0 = psm["ood_pilot"]["f1_by_k"]["0"]
        result["psm_validation_f1_k0"] = psm_val_k0
        result["psm_ood_pilot_f1_k0"] = psm_ood_k0
        result["psm_drop_k0"] = psm_val_k0 - psm_ood_k0
        result["contrast"] = {
            "network_drop_k0": result["network_drop_k0"],
            "psm_drop_k0": result["psm_drop_k0"],
            "psm_more_robust_to_ood_R": bool(result["psm_drop_k0"] < result["network_drop_k0"]),
        }

    (OUT_DIR / "extrapolation_contrast.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_DIR / 'extrapolation_contrast.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
