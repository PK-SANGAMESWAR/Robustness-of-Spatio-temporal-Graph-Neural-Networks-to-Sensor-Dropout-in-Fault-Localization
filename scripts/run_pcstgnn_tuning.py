"""S4.3 -- tuning (06_STAGE_E_pcstgnn.md), run after S4.1 (null test, PASSED) and S4.2
(gradient audit, PASSED 12/12).

Single seed (42), 15-epoch budget (Paper-1 depth study / PIGNN-screen precedent). Sweeps:

    1. lambda_1 in {0.1, 0.3, 1}, lambda_2 = 0, beta learned, use_fusion=True   -- 3 runs
    2. At the best lambda_1, lambda_2 in {0.1, 1}                              -- 2 runs

Total 5 training runs, not 6.  ``06_STAGE_E_pcstgnn.md``'s "Cost. 6 x 3.5h ~= 21 GPU-h" line
does not match its own "Do" bullet's "~=5-6 half-cost runs" or the 3+2 enumeration -- there is
no unaccounted 6th arm anywhere in the doc (a lambda_1=lambda_2=0 recheck is NOT it: that
recovers exactly the S4.1 null test, already run). The best-lambda_1 run's own lambda_2=0 result
IS the baseline the lambda_2 sweep compares against, so it is not retrained. Treating "6" as a
documentation slip (5 x 3.5h ~= 17.5 GPU-h) unless told otherwise.

Two DIFFERENT selection rules, per 06_STAGE_E_pcstgnn.md S4.3 and 12_CHECKLISTS.md SS0/A:
  - Checkpoint selection WITHIN a run: validation soft-voted graph F1 at k=0 (Paper-1 convention,
    unchanged -- this is what runner.py already does every epoch).
  - Lambda selection ACROSS runs: validation soft-voted graph F1 averaged over k in
    {0, 1, 2, 5, 10} (this script's ``lambda_selection_score``), computed on the VALIDATION split
    only -- never test, never OOD.

Branch rule (06_STAGE_E_pcstgnn.md S4.3): if the best lambda_1 lands at a grid boundary (0.1 or
1), extend one step further in that direction (0.03 below 0.1, 3.0 above 1) before finalizing.

No noise augmentation here: S4.3b ("every arm below is trained with measurement-noise
augmentation on") is scoped to the S4.4 ablation table that follows it, not to this sweep -- the
lambda_1/lambda_2 tuning trains the network's own loss terms, not the noise-dependent fusion
regime S4.3b's rationale (Sec S2.0a consequence 4) is about, and the author's own pre-drafted
docs/archive/NEXT.md prompt for this exact sweep does not mention noise either. sigma0 stays 0. Dropout-aware
training (train_k_range) stays ON, matching config/dropout_aware/pcstgnn.yaml and because
lambda_2's held-out term is vacuous (|D|=0 always) without it.

Runs strictly serially: single T1000 8GB GPU, nothing else running on it.

k=0 rho(A) cache (2026-09-30): live rho(A) costs a measured 44.7 ms/batch-of-32 (Gate 3,
`results/pcstgnn/audit/s3_4_residual_speed/speed_benchmark.json`, compute-bound), so every
arm here pays a real ~6.7 h/run tax over the physics-free rate. At k=0 (no synthetic dropout,
~25% of batches under train_k_range=[0,1,2,5]) rho(A) is a fixed function of the window alone,
so those batches read it from `cache/rtilde_train.npy`'s k=0 slice instead of recomputing it
live (`RHO_K0_CACHE_PATH` below) -- verified equivalent to the live path on real batches from
this exact loader, not synthetic ones, before being trusted here: max relative error 1.6e-5
(`scripts/verify_pcstgnn_rho_cache.py`, matching S3.4's own cross-validation tolerance),
window-id lookup by string (never by row position -- the cache's build order does not match
the split file's order). k in {1,2,5} (~75% of batches) are unaffected and stay fully live:
their masks are fresh per (epoch, batch_idx), so no fixed offline cache can serve them. This
brings the per-run rate from ~10.2h down to ~8.5h (15-epoch, physics-loaded).

Usage:
    uv run python scripts/run_pcstgnn_tuning.py                  # the full sweep
    uv run python scripts/run_pcstgnn_tuning.py --smoke           # 1-epoch pipeline check
    uv run python scripts/run_pcstgnn_tuning.py --stage lambda1   # just the lambda_1 sweep
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.experiments.config import load_config, validate_config  # noqa: E402

SEED = 42
EPOCHS = 15
MODEL = "pcstgnn"
BASE_CONFIG = "config/dropout_aware/pcstgnn.yaml"
CANONICAL_SIGMA_EPS = 0.0007255549105131452
LAMBDA1_GRID = [0.1, 0.3, 1.0]
LAMBDA2_GRID = [0.1, 1.0]
RHO_K0_CACHE_PATH = str(REPO / "cache" / "rtilde_train.npy")
CONFIG_OUT_DIR = REPO / "config" / "pcstgnn_tuning"
SUMMARY_PATH = REPO / "results" / "pcstgnn_tuning" / "lambda_selection.json"
FIGURE_PATH = REPO / "paper" / "figures" / "fe1_lambda1_sweep.png"
K_LEVELS = [0, 1, 2, 5, 10]
# PSM's validation macro-F1 at k = 0,1,2,5,10 (0.772/0.771/0.769/0.763/0.749, 04_STAGE_C S2.4,
# window-level 26-class, single mask seed), averaged -- the physics-only reference line on F-E1.
PSM_VAL_F1_MEAN_OVER_K = (0.772 + 0.771 + 0.769 + 0.763 + 0.749) / 5
N_MASK_SEEDS = 3


def _tag(lam1: float, lam2: float) -> str:
    def fmt(v):
        return str(v).replace(".", "p")
    return f"lam1_{fmt(lam1)}_lam2_{fmt(lam2)}"


def build_config(lam1: float, lam2: float, epochs: int) -> Path:
    """Self-contained config for one (lambda_1, lambda_2) arm, fully resolved.

    Mirrors ``run_dropout_aware_sweep.make_dropout_aware_config``: fully resolve base + model
    config, then bake in the frozen recipe so the arm cannot silently drift from S4.1's.
    """
    cfg = validate_config(load_config(BASE_CONFIG))
    cfg["training"]["epochs"] = epochs
    cfg["model"]["use_fusion"] = True
    cfg["model"]["beta_init"] = 0.0
    cfg["model"]["sigma_eps"] = CANONICAL_SIGMA_EPS
    cfg["model"]["sigma0"] = 0.0
    cfg["pc_loss"] = {"enabled": True, "lambda1": float(lam1), "lambda2": float(lam2),
                      "rho_k0_cache_path": RHO_K0_CACHE_PATH}

    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_OUT_DIR / f"{_tag(lam1, lam2)}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def run_dir_for(lam1: float, lam2: float, output_root: str) -> Path:
    return Path(output_root) / f"pcstgnn_tuning_{_tag(lam1, lam2)}" / MODEL / f"seed_{SEED:05d}"


def run_one(lam1: float, lam2: float, *, epochs: int, data_root: str, output_root: str) -> bool:
    """Train one arm. Resumable: skipped if its metrics.json already exists."""
    run_dir = run_dir_for(lam1, lam2, output_root)
    if (run_dir / "metrics.json").exists():
        print(f"[skip] lambda1={lam1} lambda2={lam2} -- already complete at {run_dir}", flush=True)
        return True

    config_path = build_config(lam1, lam2, epochs)
    benchmark_version = f"pcstgnn_tuning_{_tag(lam1, lam2)}"
    cmd = [
        "uv", "run", "python", "-m", "fault_localization.experiments.cli", "train",
        "--config", str(config_path),
        "--seed", str(SEED),
        "--data-root", data_root,
        "--benchmark-version", benchmark_version,
    ]
    print(f"[run] lambda1={lam1} lambda2={lam2}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAILED] lambda1={lam1} lambda2={lam2} exited with code {result.returncode}",
              flush=True)
        return False
    return True


def lambda_selection_score(run_dir: Path, data_root: str) -> dict:
    """Validation soft-voted graph F1 averaged over k in {0,1,2,5,10} -- the lambda-selection
    metric, DISTINCT from the within-run checkpoint-selection metric (val F1 at k=0, unchanged).
    Computed on the VALIDATION split only, from the run's own checkpoint_best.pt.
    """
    from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
    from fault_localization.dropout.evaluator import evaluate_dropout_curve
    from fault_localization.models.pc_stgnn import PCSTGNN
    from fault_localization.physics import residual as R
    from fault_localization.physics import table as T
    from fault_localization.physics.observation import load_norm_stats

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    val_ds = WindowGraphDataset(data_root, "val")
    val_loader = make_dataloader(val_ds, config["training"]["batch_size"], shuffle=False)

    m = config["model"]
    tbl = T.load_table(m.get("operator_dir", "operator"))
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    norm_stats = load_norm_stats(data_root)
    model = PCSTGNN(
        val_ds.edge_index, table, norm_stats,
        use_fusion=m.get("use_fusion", False),
        beta_init=m.get("beta_init", 0.0),
        sigma_eps=m.get("sigma_eps", 1e-3),
        sigma0=m.get("sigma0", 0.0),
        aggregator=m.get("aggregator", "max"),
        gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
        num_layers=config.get("gnn_layers", 1),
        temporal_hidden_dim=m.get("temporal_hidden_dim", 128),
        num_classes=26,
        dropout=m.get("dropout", 0.35),
    ).to(device)
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    curve = evaluate_dropout_curve(
        model, val_loader, device,
        dropout_levels=K_LEVELS, n_mask_seeds=N_MASK_SEEDS,
        policy=config.get("dropout_study", {}).get("policy", "zero_messages"),
        f1_average=config["metrics"]["f1_average"],
        model_name="pcstgnn_tuning",
    )
    per_k = {str(k): curve[k]["f1_mean"] for k in K_LEVELS}
    avg_over_k = sum(per_k.values()) / len(per_k)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    return {
        "val_f1_per_k": per_k,
        "val_f1_avg_over_k": avg_over_k,
        "checkpoint_selection_val_f1_at_k0": metrics["best_validation_graph_f1"],
        "beta_final": metrics.get("beta_final"),
    }


def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return {"arms": {}, "note": (
        "checkpoint_selection_val_f1_at_k0 = within-run selection rule (val F1 at k=0, "
        "Paper-1 convention). val_f1_avg_over_k = the lambda-selection rule for THIS sweep "
        "(mean val F1 over k in {0,1,2,5,10}). These are two different rules; do not conflate."
    )}


def save_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_arm(lam1: float, lam2: float, *, data_root: str, output_root: str) -> dict:
    run_dir = run_dir_for(lam1, lam2, output_root)
    summary = load_summary()
    key = _tag(lam1, lam2)
    if key in summary["arms"]:
        return summary["arms"][key]
    score = lambda_selection_score(run_dir, data_root)
    score["lambda1"] = lam1
    score["lambda2"] = lam2
    summary["arms"][key] = score
    save_summary(summary)
    return score


def extend_boundary(best_lam1: float) -> float | None:
    if best_lam1 == min(LAMBDA1_GRID):
        return round(best_lam1 / 3.0, 4)   # one step further down: 0.1 -> 0.03
    if best_lam1 == max(LAMBDA1_GRID):
        return best_lam1 * 3.0             # one step further up: 1 -> 3.0
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="dataset_fullscale_50")
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true", help="1-epoch pipeline check, distinct tag")
    ap.add_argument("--stage", choices=["lambda1", "lambda2", "both"], default="both")
    args = ap.parse_args()

    epochs = 1 if args.smoke else args.epochs
    data_root = args.data_root
    output_root = args.output_root

    print(f"S4.3 tuning sweep -- seed={SEED} epochs={epochs} data_root={data_root} "
          f"(serial, single GPU)", flush=True)

    lam1_grid = list(LAMBDA1_GRID)
    scored: dict[float, dict] = {}

    if args.stage in ("lambda1", "both"):
        for lam1 in lam1_grid:
            if not run_one(lam1, 0.0, epochs=epochs, data_root=data_root, output_root=output_root):
                return 1
            scored[lam1] = evaluate_arm(lam1, 0.0, data_root=data_root, output_root=output_root)
            print(f"  lambda1={lam1}: val_f1_avg_over_k={scored[lam1]['val_f1_avg_over_k']:.4f} "
                  f"(k0 checkpoint sel={scored[lam1]['checkpoint_selection_val_f1_at_k0']:.4f}, "
                  f"beta={scored[lam1]['beta_final']:.4f})", flush=True)

        if not args.smoke:
            best_lam1 = max(scored, key=lambda k: scored[k]["val_f1_avg_over_k"])
            ext = extend_boundary(best_lam1)
            if ext is not None:
                print(f"[branch] best lambda1={best_lam1} at grid boundary -- extending to "
                      f"{ext}", flush=True)
                if not run_one(ext, 0.0, epochs=epochs, data_root=data_root,
                               output_root=output_root):
                    return 1
                scored[ext] = evaluate_arm(ext, 0.0, data_root=data_root, output_root=output_root)
                print(f"  lambda1={ext}: val_f1_avg_over_k="
                      f"{scored[ext]['val_f1_avg_over_k']:.4f}", flush=True)
                lam1_grid.append(ext)
    else:
        # lambda2-only stage assumes a prior lambda1 sweep is already recorded.
        summary = load_summary()
        for key, v in summary["arms"].items():
            if v["lambda2"] == 0.0:
                scored[v["lambda1"]] = v

    if not scored:
        print("[error] no lambda1 arms scored -- cannot proceed to lambda2 stage", flush=True)
        return 1

    best_lam1 = max(scored, key=lambda k: scored[k]["val_f1_avg_over_k"])
    print(f"\n=== best lambda1 so far: {best_lam1} "
          f"(val_f1_avg_over_k={scored[best_lam1]['val_f1_avg_over_k']:.4f}) ===\n", flush=True)

    lam2_scored = {0.0: scored[best_lam1]}
    if args.stage in ("lambda2", "both"):
        for lam2 in LAMBDA2_GRID:
            if not run_one(best_lam1, lam2, epochs=epochs, data_root=data_root,
                           output_root=output_root):
                return 1
            lam2_scored[lam2] = evaluate_arm(best_lam1, lam2, data_root=data_root,
                                             output_root=output_root)
            print(f"  lambda1={best_lam1} lambda2={lam2}: "
                  f"val_f1_avg_over_k={lam2_scored[lam2]['val_f1_avg_over_k']:.4f} "
                  f"beta={lam2_scored[lam2]['beta_final']:.4f}", flush=True)

    best_lam2 = max(lam2_scored, key=lambda k: lam2_scored[k]["val_f1_avg_over_k"])

    print("\n=== S4.3 sweep complete ===")
    print(f"selected lambda1 = {best_lam1}")
    print(f"selected lambda2 = {best_lam2}")
    print(f"selected arm beta_final = {lam2_scored[best_lam2]['beta_final']}")
    print(f"selected arm val_f1_avg_over_k (lambda-selection rule) = "
          f"{lam2_scored[best_lam2]['val_f1_avg_over_k']:.4f}")
    print(f"selected arm checkpoint-selection val F1 at k=0 (Paper-1 rule) = "
          f"{lam2_scored[best_lam2]['checkpoint_selection_val_f1_at_k0']:.4f}")

    if not args.smoke and args.stage == "both":
        try:
            build_figure(lam1_grid, scored)
        except Exception as e:  # figure is a nice-to-have, never blocks the numeric result
            print(f"[warn] F-E1 figure not built: {e}", flush=True)

    return 0


def build_figure(lam1_grid: list[float], scored: dict[float, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = sorted(lam1_grid)
    ys = [scored[x]["val_f1_avg_over_k"] for x in xs]
    best = max(xs, key=lambda x: scored[x]["val_f1_avg_over_k"])
    extension = [x for x in xs if x not in LAMBDA1_GRID]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "-", color="#2b6cb0", zorder=1)
    for x, y in zip(xs, ys):
        hollow = x in extension
        ax.scatter([x], [y], s=60, zorder=2, color="white" if hollow else "#2b6cb0",
                   edgecolors="#2b6cb0", linewidths=1.6)
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.axhline(PSM_VAL_F1_MEAN_OVER_K, color="#888888", ls="--", lw=1)
    ax.text(xs[-1], PSM_VAL_F1_MEAN_OVER_K + 0.002, "PSM, physics only (S2.4)", ha="right",
            va="bottom", fontsize=8.5, color="#555555")
    ax.annotate("selected", (best, scored[best]["val_f1_avg_over_k"]), textcoords="offset points",
                xytext=(18, -18), fontsize=9, color="#B5541C",
                arrowprops=dict(arrowstyle="->", color="#B5541C"))
    lo = min(min(ys), PSM_VAL_F1_MEAN_OVER_K)
    ax.set_ylim(lo - 0.01, max(ys) + 0.012)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_1$  ($\lambda_2 = 0$, $\beta$ learned)")
    ax.set_ylabel("Validation macro-F1\n(mean over k ∈ {0,1,2,5,10})")
    ax.set_title(r"F-E1: validation F1 vs. $\lambda_1$ (S4.3, seed 42, 15 epochs, noise-free)",
                 fontsize=10)
    if extension:
        ax.text(0.98, 0.97, "hollow = boundary-extension arm", transform=ax.transAxes,
                fontsize=8, color="#555555", ha="right", va="top")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)
    print(f"wrote {FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
