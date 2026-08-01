"""End-to-end orchestrator for the SDR-STGNN sensor-dropout contribution.

Runs the whole contribution on ONE GPU, in sequence (single T1000 — no parallel
GPU jobs), each step resumable/crash-isolated:

  1. Train SDR-STGNN            (gated aggregation + dropout-aware) → results/sdr_stgnn/
  2. Train dropout-aware RGSAGE-Max (mask-aware, ungated baseline)  → results/dropout_aware/
  3. Degradation study — 5 standard baselines (inference)          → results/dropout_study/
  4. Degradation study — dropout-aware RGSAGE-Max (inference)       → results/dropout_study_aware/
  5. Degradation study — SDR-STGNN (inference)                      → results/dropout_study_sdr/
  6. Build ALL result figures (degradation / ablation / retention / AUDC / heatmap
     + the standard-vs-dropout-aware overlay with the SDR curve).

Steps 1-2 are the only training (each ~one 50-run model on the frozen recipe);
3-6 are fast. Standard baselines (results/fullscale_final) must already exist.
Corpus-agnostic: pass --data-root dataset_fullscale_100 to run on the 100-run corpus.

Usage (real; GPU must be free):
    nohup uv run python scripts/run_sdr_contribution.py \
        > results/sdr_contribution.log 2>&1 &

Skip training (only (re)build curves + figures from existing checkpoints):
    uv run python scripts/run_sdr_contribution.py --eval-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = ["uv", "run", "python"]


def step(desc: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 78}\n[STEP] {desc}\n{'=' * 78}\n  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO)
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"[FAILED] {desc} (rc={r.returncode}, {dt:.1f} min) — stopping.", flush=True)
        sys.exit(r.returncode)
    print(f"[ok] {desc} — {dt:.1f} min", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--baselines-dir", default="results/fullscale_final",
                   help="Existing standard-trained baselines (for the context curves).")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip steps 1-2 (training); only run degradation studies + figures.")
    p.add_argument("--k-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    p.add_argument("--n-mask-seeds", type=int, default=3)
    args = p.parse_args()

    dr = args.data_root
    kl = [str(k) for k in args.k_levels]
    total0 = time.time()

    # 1-2. Training (the only GPU-heavy steps).
    if not args.eval_only:
        step("Train SDR-STGNN (gated + dropout-aware)",
             PY + ["scripts/run_sdr_stgnn.py", "--data-root", dr])
        step("Train dropout-aware RGSAGE-Max (mask-aware, ungated baseline)",
             PY + ["scripts/run_dropout_aware_sweep.py", "--models", "rgsage_max",
                   "--data-root", dr])

    # 3-5. Degradation studies (inference only).
    study = PY + ["scripts/run_dropout_study.py", "--data-root", dr,
                  "--k-levels", *kl, "--n-mask-seeds", str(args.n_mask_seeds)]
    step("Degradation study — 5 standard baselines",
         study + ["--results-dir", args.baselines_dir,
                  "--output-dir", "results/dropout_study"])
    step("Degradation study — dropout-aware RGSAGE-Max",
         study + ["--results-dir", "results/dropout_aware", "--models", "rgsage_max",
                  "--output-dir", "results/dropout_study_aware"])
    step("Degradation study — SDR-STGNN",
         study + ["--results-dir", "results/sdr_stgnn", "--models", "sdr_stgnn",
                  "--output-dir", "results/dropout_study_sdr"])

    # 6. Figures.
    step("Build SDR result figures (degradation / ablation / retention / AUDC / heatmap)",
         PY + ["scripts/build_sdr_result_figures.py",
               "--baselines", "results/dropout_study/degradation_curves.csv",
               "--aware", "results/dropout_study_aware/degradation_curves.csv",
               "--sdr", "results/dropout_study_sdr/degradation_curves.csv",
               "--out-dir", "paper/figures"])
    step("Build standard-vs-dropout-aware overlay (with SDR curve)",
         PY + ["scripts/build_dropout_comparison.py",
               "--standard", "results/dropout_study/degradation_curves.csv",
               "--dropout-aware", "results/dropout_study_aware/degradation_curves.csv",
               "--sdr", "results/dropout_study_sdr/degradation_curves.csv",
               "--output-dir", "results/dropout_study"])

    print(f"\n{'=' * 78}\nSDR contribution complete in {(time.time()-total0)/60:.1f} min.\n"
          f"Figures → paper/figures/sdr_*.png ; summary → paper/figures/sdr_robustness_summary.csv\n"
          f"{'=' * 78}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
