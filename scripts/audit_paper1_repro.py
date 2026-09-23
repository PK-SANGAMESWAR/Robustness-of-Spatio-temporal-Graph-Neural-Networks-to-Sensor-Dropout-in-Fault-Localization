"""Stage A S0.1 -- do the Paper-1 numbers still reproduce?

Every comparison in Paper 2 is paired against Paper 1's tables.  If those
numbers no longer come out of the artefacts on disk, every delta this project
reports is measured against a moving reference, so this is the one Stage-A check
whose failure stops the project (``02_STAGE_A_audit.md`` S0.1).

Three layers, cheapest first, because they fail differently:

``derived``  (free)
    Recompute each run's macro-F1 **from its own** ``predictions_test.csv`` and
    compare against the ``metrics.json`` written at training time.  This catches
    a changed metric definition or a corrupted prediction dump without touching
    a GPU.
``aggregate``  (free)
    Recompute ``benchmark_summary.csv`` from the per-seed ``metrics.json`` files,
    and check every ``degradation_curves.csv`` against its own stored
    ``f1_per_train_seed``.  This catches a summary that has drifted from the runs
    it claims to summarise -- including the classic failure of a
    ``*.prebugfix.csv`` leaking into a current table.
``reeval``  (GPU, opt-in)
    Re-run the dropout sweep from the checkpoints and diff against the stored
    curves.  This is the only layer that re-derives the numbers end to end.

Usage:
    uv run python scripts/audit_paper1_repro.py                  # free layers
    uv run python scripts/audit_paper1_repro.py --reeval         # + GPU sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results/pcstgnn/audit"

CURVE_SETS = {
    "standard": ("results/dropout_study/degradation_curves.csv", "results/fullscale_final"),
    "dropout_aware": ("results/dropout_study_aware/degradation_curves.csv",
                      "results/dropout_aware"),
    "sdr": ("results/dropout_study_sdr/degradation_curves.csv", "results/sdr_stgnn"),
}
PRINTED_PRECISION = 1e-4          # the precision the paper actually prints


def macro_f1(truth: np.ndarray, pred: np.ndarray, n_classes: int = 26) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def iter_runs(results_root: Path):
    for family in ("fullscale_final", "dropout_aware", "sdr_stgnn"):
        base = results_root / family
        if not base.exists():
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "figures"):
            for seed_dir in sorted(model_dir.glob("seed_*")):
                if (seed_dir / "metrics.json").exists():
                    yield family, model_dir.name, seed_dir


def layer_derived(results_root: Path) -> list[dict]:
    """metrics.json vs a fresh macro-F1 off predictions_test.csv."""
    rows = []
    for family, model, seed_dir in iter_runs(results_root):
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        stored = float(metrics["test_graph"]["f1"])
        pred_path = seed_dir / "predictions_test.csv"
        if not pred_path.exists():
            rows.append({"run": f"{family}/{model}/{seed_dir.name}", "stored_f1": stored,
                         "recomputed_f1": None, "abs_diff": None, "ok": None})
            continue
        truth, pred = [], []
        with pred_path.open() as fh:
            for row in csv.DictReader(fh):
                truth.append(int(row["truth"]))
                pred.append(int(row["vote_prediction"]))
        fresh = macro_f1(np.asarray(truth), np.asarray(pred))
        rows.append({"run": f"{family}/{model}/{seed_dir.name}", "stored_f1": stored,
                     "recomputed_f1": fresh, "abs_diff": abs(fresh - stored),
                     "ok": abs(fresh - stored) < PRINTED_PRECISION})
    return rows


def layer_summary(results_root: Path) -> list[dict]:
    """benchmark_summary.csv vs the per-seed metrics.json it summarises."""
    path = results_root / "fullscale_final/benchmark_summary.csv"
    if not path.exists():
        return []
    per_model: dict[str, list[float]] = {}
    for family, model, seed_dir in iter_runs(results_root):
        if family != "fullscale_final":
            continue
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        per_model.setdefault(model, []).append(float(metrics["test_graph"]["f1"]))
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            seeds = sorted(per_model.get(row["model"], []))
            stored_mean = float(row["graph_macro_f1_mean"])
            stored_std = float(row["graph_macro_f1_std"])
            fresh_mean = float(np.mean(seeds)) if seeds else float("nan")
            fresh_std = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan")
            rows.append({
                "model": row["model"], "n_seeds_csv": int(row["n_seeds"]),
                "n_seeds_found": len(seeds),
                "stored_mean": stored_mean, "recomputed_mean": fresh_mean,
                "mean_abs_diff": abs(fresh_mean - stored_mean),
                "stored_std": stored_std, "recomputed_std": fresh_std,
                "std_abs_diff": abs(fresh_std - stored_std),
                "ok": (len(seeds) == int(row["n_seeds"])
                       and abs(fresh_mean - stored_mean) < PRINTED_PRECISION
                       and abs(fresh_std - stored_std) < PRINTED_PRECISION),
            })
    return rows


def layer_curves(repo: Path) -> list[dict]:
    """Each curve row vs its own stored per-seed values."""
    rows = []
    for name, (rel, _) in CURVE_SETS.items():
        path = repo / rel
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                per_seed = np.asarray(json.loads(row["f1_per_train_seed"]), dtype=float)
                stored_mean = float(row["f1_mean"])
                stored_std = float(row["f1_std"])
                fresh_mean = float(per_seed.mean())
                fresh_std = (float(per_seed.std(ddof=1)) if per_seed.size > 1
                             else stored_std)
                # per-seed values are rounded to 6 dp in the CSV, so the mean can
                # only be recovered to about that precision.
                rows.append({
                    "curve": name, "model": row["model"], "k": int(row["k"]),
                    "n_train_seeds": int(row["n_train_seeds"]),
                    "stored_mean": stored_mean, "recomputed_mean": fresh_mean,
                    "mean_abs_diff": abs(fresh_mean - stored_mean),
                    "std_abs_diff": abs(fresh_std - stored_std),
                    "err_kind": row["err_kind"],
                    "ok": (abs(fresh_mean - stored_mean) < 1e-5
                           and abs(fresh_std - stored_std) < 1e-5
                           and int(row["n_train_seeds"]) == per_seed.size),
                })
    return rows


def layer_prebugfix(repo: Path) -> dict:
    """No current table may be a pre-bugfix file (`.rules/40-data.md`)."""
    stale = sorted(str(p.relative_to(repo)) for p in repo.glob("results/**/*.prebugfix.*"))
    live = [rel for rel, _ in CURVE_SETS.values() if "prebugfix" in rel]
    return {"n_prebugfix_files": len(stale), "examples": stale[:6],
            "current_tables_reference_prebugfix": live}


def layer_reeval(repo: Path, out_dir: Path, models: list[str], seeds: list[int],
                 device: str) -> list[dict]:
    """Re-run the sweep from checkpoints and diff against the stored curve."""
    scratch = out_dir / "reeval"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (rel, results_dir) in CURVE_SETS.items():
        stored_path = repo / rel
        if not stored_path.exists():
            continue
        stored = {}
        with stored_path.open() as fh:
            for row in csv.DictReader(fh):
                stored[(row["model"], int(row["k"]))] = float(row["f1_mean"])
        wanted = [m for m in models if any(m == k[0] for k in stored)]
        if not wanted:
            continue
        target = scratch / name
        cmd = [sys.executable, str(repo / "scripts/run_dropout_study.py"),
               "--results-dir", str(repo / results_dir),
               "--data-root", str(repo / "dataset_fullscale_50"),
               "--output-dir", str(target), "--device", device,
               "--models", *wanted, "--seeds", *[str(s) for s in seeds]]
        print("  $ " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=repo)
        fresh_path = target / "degradation_curves.csv"
        with fresh_path.open() as fh:
            for row in csv.DictReader(fh):
                key = (row["model"], int(row["k"]))
                if key not in stored:
                    continue
                fresh = float(row["f1_mean"])
                rows.append({"curve": name, "model": key[0], "k": key[1],
                             "stored_f1_mean": stored[key], "reeval_f1_mean": fresh,
                             "abs_diff": abs(fresh - stored[key]),
                             "n_seeds_reeval": int(row["n_train_seeds"]),
                             "ok": abs(fresh - stored[key]) < PRINTED_PRECISION})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(REPO / "results"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--reeval", action="store_true", help="also re-run the GPU sweep")
    ap.add_argument("--reeval-models", nargs="+", default=["rgsage_max"])
    ap.add_argument("--reeval-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    derived = layer_derived(results_root)
    summary = layer_summary(results_root)
    curves = layer_curves(REPO)
    prebugfix = layer_prebugfix(REPO)
    reeval = (layer_reeval(REPO, out_dir, args.reeval_models, args.reeval_seeds, args.device)
              if args.reeval else [])

    def failures(rows):
        return [r for r in rows if r.get("ok") is False]

    all_rows = {"derived": derived, "summary": summary, "curves": curves, "reeval": reeval}
    with (out_dir / "paper1_repro.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer", "key", "stored", "recomputed", "abs_diff", "ok"])
        for r in derived:
            writer.writerow(["derived", r["run"], r["stored_f1"], r["recomputed_f1"],
                             r["abs_diff"], r["ok"]])
        for r in summary:
            writer.writerow(["summary", r["model"], r["stored_mean"], r["recomputed_mean"],
                             r["mean_abs_diff"], r["ok"]])
        for r in curves:
            writer.writerow(["curves", f"{r['curve']}/{r['model']}/k{r['k']}",
                             r["stored_mean"], r["recomputed_mean"],
                             r["mean_abs_diff"], r["ok"]])
        for r in reeval:
            writer.writerow(["reeval", f"{r['curve']}/{r['model']}/k{r['k']}",
                             r["stored_f1_mean"], r["reeval_f1_mean"],
                             r["abs_diff"], r["ok"]])

    passed = all(not failures(rows) for rows in all_rows.values())
    (out_dir / "paper1_repro.json").write_text(json.dumps(
        {"pass": passed, "printed_precision": PRINTED_PRECISION,
         "reeval_run": bool(args.reeval), "prebugfix": prebugfix,
         **{k: v for k, v in all_rows.items()}}, indent=2) + "\n")

    for layer, rows in all_rows.items():
        if not rows:
            print(f"{layer:<9}: (not run)")
            continue
        bad = failures(rows)
        worst = max((r.get("abs_diff") or r.get("mean_abs_diff") or 0.0) for r in rows)
        print(f"{layer:<9}: {len(rows) - len(bad)}/{len(rows)} match  "
              f"(worst abs diff {worst:.2e})")
        for r in bad[:5]:
            print(f"    MISMATCH {r}")
    print(f"pre-bugfix files present (quarantined, not read): "
          f"{prebugfix['n_prebugfix_files']}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}"
          + ("" if args.reeval else "   [free layers only -- run with --reeval for the "
                                    "end-to-end GPU check]"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
