"""Stage G result tables (``PC-STGNN/10_FIGURES_TABLES.md``): T5, T7, T10, T11, T13 from the non-sealed splits.

CPU only. Reads scoring JSON files and saved predictions; never opens the sealed set and never writes into
``results/s4_4`` or ``results/stage_g/{val,test}``. Every number in a table is copied from a file on disk; each row
carries its source path, split, seed count and evaluation protocol (``12_CHECKLISTS.md`` § 0).

Conventions (``11_METRICS.md``):
- seed spread: mean and sample std (ddof = 1) over seeds 42/43/44; a 1-seed row has ``n_seeds = 1`` and no std;
- AUDC (relative): trapezoid of F1(k)/F1(0) over k in {0, 1, 2, 5, 10}, divided by k_max - k_min (Paper 1 Eq. 8),
  computed per training seed and then averaged; AUDC (absolute): the same integral on raw F1(k);
- paired two-sided t-test over seeds (df = 2) and Holm step-down, as ``scripts/summarize_stage_g.py``;
- event-level paired cluster bootstrap: ``fault_localization.stage_g.gate6.analyze_cell`` (10,000 resamples,
  ``RandomState(20261102)``), reused unchanged for the test-split pairs of ``14_PREREGISTRATION.md`` § 3 as amended
  (R5 vs R0n, R5 vs R6i, R5 vs R2).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
K_LEVELS = (0, 1, 2, 5, 10)
SIGMAS = (0.0, 0.0005, 0.002, 0.01)
TRIVIAL_NOFAULT_MACRO_F1 = (2 * 0.5 * 1.0 / 1.5) / 26        # 0.0256, 11_METRICS.md § 1


# ------------------------------------------------------------------------------------------
# Reading scoring files (two formats: the Gate-4 scorer and the Stage G scorer)
# ------------------------------------------------------------------------------------------


def parse_cell(name: str) -> tuple[int, float]:
    """``"k10_sigma0.002"`` / ``"k0_sigma0"`` / ``"k5"`` -> ``(10, 0.002)`` / ``(0, 0.0)`` / ``(5, 0.0)``."""
    if "_sigma" in name:
        k, s = name.split("_sigma")
        return int(k[1:]), float(s)
    return int(name[1:]), 0.0


def load_cells(path: Path) -> dict:
    """``{(k, sigma): {"f1", "by_res", "by_group", "net", "psm"}}``; absent fields are ``None``.

    Gate-4 scorer files (``results/s4_4/*/scoring_full_grid.json``) carry ``f1_fused_mean``; Stage G scorer files
    carry ``f1_mean``. Both are the fused soft-voted 26-class macro-F1, mean over the mask/noise offsets.
    """
    d = json.loads(Path(path).read_text())["cells"]
    out = {}
    for name, v in d.items():
        f1 = v["f1_mean"] if "f1_mean" in v else v["f1_fused_mean"]
        out[parse_cell(name)] = {
            "f1": float(f1),
            "by_res": {float(r): float(x) for r, x in (v.get("f1_by_resistance") or {}).items()},
            "by_group": dict(v.get("f1_by_group") or {}),
            "net": v.get("f1_network_only_mean"),
            "psm": v.get("f1_psm_mean"),
        }
    return out


def s4_4_path(arm: str, seed: int) -> Path:
    return REPO / f"results/s4_4/{arm}/pcstgnn/seed_{seed:05d}/scoring_full_grid.json"


def stage_g_path(split: str, name: str, seed: int) -> Path:
    return REPO / f"results/stage_g/{split}/{name}/seed_{seed:05d}_scoring.json"


def available_seeds(path_of, seeds=SEEDS) -> list[int]:
    return [s for s in seeds if Path(path_of(s)).exists()]


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------


def mean_std(values) -> tuple[float, float | None]:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        raise ValueError("no values")
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else None)


def normalized_trapezoid(ks, values) -> float:
    """Paper 1 Eq. 8: trapezoid over k divided by (k_max - k_min). Same as ``scripts/build_absolute_audc.py``."""
    ks, values = np.asarray(ks, float), np.asarray(values, float)
    order = np.argsort(ks)
    ks, values = ks[order], values[order]
    return float(np.trapezoid(values, ks) / (ks[-1] - ks[0]))


def audc(ks, f1s, relative: bool) -> float:
    f1s = np.asarray(f1s, float)
    ks = np.asarray(ks, float)
    if relative:
        f0 = f1s[ks == 0]
        if f0.size != 1 or f0[0] <= 0:
            raise ValueError("relative AUDC needs one positive F1 at k = 0")
        f1s = f1s / f0[0]
    return normalized_trapezoid(ks, f1s)


def paired(a, b) -> dict:
    """Paired two-sided t-test over seeds of ``a - b`` (same order of seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if d.size < 2:
        p = None
    elif np.ptp(d) == 0:
        p = 0.0 if d.mean() != 0 else 1.0
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(d.mean()), "n_positive": int((d > 0).sum()),
            "n": int(d.size), "p": p, "per_seed_delta": [float(x) for x in d]}


def holm(ps) -> list[float]:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return [float(x) for x in adj]


# ------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------


def fmt(mean: float | None, std: float | None = None, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    return f"{mean:.{nd}f}" if std is None else f"{mean:.{nd}f} ± {std:.{nd}f}"


def write_table(rows: list[dict], columns: list[str], out_base: Path, title: str, notes: list[str]) -> None:
    """Write ``<out_base>.csv`` (all keys) and ``<out_base>.md`` (``columns`` only, with a title and notes)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out_base.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    lines += [""] + [f"- {n}" for n in notes] + [""]
    out_base.with_suffix(".md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------------------
# T5 — main results, test split, random dropout, sigma = 0
# ------------------------------------------------------------------------------------------

#: PC-STGNN models scored by the Stage G scorer on the test split (``results/stage_g/test/<name>``)
T5_MODELS = (("R0ⁿ (noise-aug., no physics)", "r0n"), ("R5 (PC-STGNN)", "r5"), ("R2 (learned decoder)", "r2"),
             ("R6ⁱ (shuffled table, inference)", "r6i"), ("B2ᵃ (regressed R̂; epoch-1 checkpoint)", "b2a"))
#: Paper-1 rows (Paper-1 evaluator; same split, same metric; mask streams differ for k > 0)
PAPER1_CURVES = {"standard": "results/dropout_study/degradation_curves.csv",
                 "dropout-aware": "results/dropout_study_aware/degradation_curves.csv"}
PAPER1_NAMES = {"gru": "GRU", "rgcn": "RGCN", "rgsage_mean": "RGSAGE-Mean", "rgsage_max": "RGSAGE-Max", "rgatv2": "RGATv2"}


def curve_row(label: str, per_seed: dict[int, dict[int, float]], source: str, protocol: str) -> dict:
    """One T5 row from ``{seed: {k: F1}}``. AUDC only when all five k exist for every seed."""
    seeds = sorted(per_seed)
    row = {"model": label, "n_seeds": len(seeds), "protocol": protocol, "source": source}
    for k in K_LEVELS:
        vals = [per_seed[s][k] for s in seeds if k in per_seed[s]]
        m, sd = mean_std(vals) if len(vals) == len(seeds) else (None, None)
        row[f"f1_k{k}"] = m
        row[f"f1_k{k}_std"] = sd
        row[f"k={k}"] = fmt(m, sd)
    full = all(all(k in per_seed[s] for k in K_LEVELS) for s in seeds)
    if full:
        rel = [audc(K_LEVELS, [per_seed[s][k] for k in K_LEVELS], True) for s in seeds]
        ab = [audc(K_LEVELS, [per_seed[s][k] for k in K_LEVELS], False) for s in seeds]
        ret = [per_seed[s][10] / per_seed[s][0] for s in seeds]
        for name, vals in (("audc_rel", rel), ("audc_abs", ab), ("ret10", ret)):
            m, sd = mean_std(vals)
            row[name], row[f"{name}_std"] = m, sd
        row["AUDC rel"] = fmt(row["audc_rel"], row["audc_rel_std"])
        row["AUDC abs"] = fmt(row["audc_abs"], row["audc_abs_std"])
        row["Ret@10"] = f"{100 * row['ret10']:.1f} %"
    else:
        row["AUDC rel"] = row["AUDC abs"] = row["Ret@10"] = "—"
    return row


def stage_g_curve(name: str, split: str = "test", sigma: float = 0.0) -> dict[int, dict[int, float]]:
    out = {}
    for s in available_seeds(lambda s: stage_g_path(split, name, s)):
        cells = load_cells(stage_g_path(split, name, s))
        out[s] = {k: v["f1"] for (k, sg), v in cells.items() if sg == sigma}
    return out


def paper1_curve(csv_path: Path, model: str) -> dict[int, dict[int, float]]:
    per_seed: dict[int, dict[int, float]] = {}
    with Path(csv_path).open() as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model:
                continue
            seeds = json.loads(r["seeds"])
            vals = json.loads(r["f1_per_train_seed"])
            for s, v in zip(seeds, vals):
                per_seed.setdefault(int(s), {})[int(r["k"])] = float(v)
    return per_seed


def build_t5() -> list[dict]:
    rows = []
    for label, name in T5_MODELS:
        cur = stage_g_curve(name)
        if cur:
            rows.append(curve_row(label, cur, f"results/stage_g/test/{name}/seed_*_scoring.json",
                                  "Stage G scorer, test, random dropout, sigma=0, 3 offsets"))
    # PSM (B2) read from the R5 scorer: physics-only prediction, no learned weight
    psm = {}
    for s in available_seeds(lambda s: stage_g_path("test", "r5", s)):
        cells = load_cells(stage_g_path("test", "r5", s))
        psm[s] = {k: v["psm"] for (k, sg), v in cells.items() if sg == 0.0 and v["psm"] is not None}
    if psm:
        vals = {k: [psm[s][k] for s in psm] for k in K_LEVELS}
        same = all(np.ptp(v) < 1e-12 for v in vals.values())
        one = {42: psm[min(psm)]} if same else psm
        rows.append(curve_row("PSM (B2, zero parameters)", one, "results/stage_g/test/r5/seed_*_scoring.json (f1_psm_mean)",
                              "Stage G scorer, test, sigma=0; identical in the 3 R5 files" if same else
                              "Stage G scorer, test, sigma=0; differs across R5 files"))
    for regime, rel in PAPER1_CURVES.items():
        for model, pretty in PAPER1_NAMES.items():
            cur = paper1_curve(REPO / rel, model)
            if cur:
                tag = " = R0" if (regime == "dropout-aware" and model == "rgsage_max") else ""
                rows.append(curve_row(f"{pretty} ({regime}){tag}", cur, rel,
                                      "Paper-1 evaluator, test, zero_messages; other mask stream for k > 0"))
    return rows


# ------------------------------------------------------------------------------------------
# T7 — core ablation, validation split
# ------------------------------------------------------------------------------------------

T7_ARMS = (("R0ⁿ", "r0n"), ("R1 (type head)", "r1"), ("R2 (learned decoder)", "r2"), ("R3 (λ₁ only)", "r3"),
           ("R4 (λ₂ only)", "r4"), ("R5 (PC-STGNN)", "r5"), ("R5-γ0 (no aux. loss)", "r5_gamma0"),
           ("R6ⁱ (shuffled table, inference)", "r6i"), ("R8 (no noise aug.)", "r8"))
T7_CELLS = ((0, 0.0, None, "k0 σ0 (H6)"), (0, 0.0, 10.0, "10 Ω k0 σ0 (H1)"), (10, 0.0, None, "k10 σ0 (H2)"),
            (0, 0.01, None, "k0 σ1e-2 (H4)"), (10, 0.01, None, "k10 σ1e-2"))


def _cell_value(cells: dict, k: int, sigma: float, res: float | None):
    v = cells.get((k, sigma))
    if v is None:
        return None
    if res is None:
        return v["f1"]
    return v["by_res"].get(res)


def build_t7() -> list[dict]:
    rows = []
    for label, arm in T7_ARMS:
        seeds = available_seeds(lambda s: s4_4_path(arm, s))
        if not seeds:
            continue
        cells = {s: load_cells(s4_4_path(arm, s)) for s in seeds}
        for s in seeds:
            # Gate-4 files (R0n, R5) have no per-R split; the H1 re-score (k = 0, sigma = 0) holds it
            h1 = s4_4_path(arm, s).with_name("scoring_h1.json")
            if h1.exists() and not cells[s][(0, 0.0)]["by_res"]:
                d = json.loads(h1.read_text())
                if d["k"] == 0 and d["sigma"] == 0.0 and abs(d["f1_fused_pooled"] - cells[s][(0, 0.0)]["f1"]) < 1e-9:
                    cells[s][(0, 0.0)]["by_res"] = {float(r): float(x) for r, x in d["f1_fused_by_resistance"].items()}
        row = {"arm": label, "n_seeds": len(seeds), "source": f"results/s4_4/{arm}/pcstgnn/seed_*/scoring_full_grid.json"
                                                              " (+ scoring_h1.json for 10 Ω where present)"}
        for k, sg, res, name in T7_CELLS:
            vals = [_cell_value(cells[s], k, sg, res) for s in seeds]
            if any(v is None for v in vals):
                row[name] = "—"
                continue
            m, sd = mean_std(vals)
            row[name] = fmt(m, sd)
            row[f"{name} mean"] = m
        rows.append(row)
    b = REPO / "results/stage_g/val/b2a/seed_00042_scoring.json"
    if b.exists():
        cells = load_cells(b)
        row = {"arm": "B2ᵃ (regressed R̂; epoch-1 checkpoint)", "n_seeds": 1, "source": str(b.relative_to(REPO))}
        for k, sg, res, name in T7_CELLS:
            v = _cell_value(cells, k, sg, res)
            row[name] = fmt(v)
            if v is not None:
                row[f"{name} mean"] = v
        rows.append(row)
    # R0 (Paper 1 anchor): only the training-loop validation F1 at k = 0, sigma = 0 exists
    r0 = []
    for s in SEEDS:
        p = REPO / f"results/dropout_aware/rgsage_max/seed_{s:05d}/metrics.json"
        if p.exists():
            r0.append(float(json.loads(p.read_text())["best_validation_graph_f1"]))
    if r0:
        m, sd = mean_std(r0)
        row = {"arm": "R0 (Paper 1, no noise aug.)", "n_seeds": len(r0),
               "source": "results/dropout_aware/rgsage_max/seed_*/metrics.json (best_validation_graph_f1)"}
        for _k, _s, _r, name in T7_CELLS:
            row[name] = "not scored"
        row["k0 σ0 (H6)"] = fmt(m, sd)
        row["k0 σ0 (H6) mean"] = m
        rows.insert(0, row)
        r0n = next((r for r in rows if r["arm"] == "R0ⁿ"), None)
        if r0n is not None and len(r0) == 3:
            r0n_vals = [load_cells(s4_4_path("r0n", s))[(0, 0.0)]["f1"] for s in SEEDS]
            pr = paired(r0n_vals, r0)
            rows.insert(2, {"arm": "R0ⁿ − R0 (noise augmentation alone)", "n_seeds": 3,
                            "k0 σ0 (H6)": f"{pr['delta']:+.3f} ({pr['n_positive']}/3 seeds > 0, p={pr['p']:.3f})",
                            "source": "difference of the two rows above, paired by seed"})
    return rows


def r0n_metric_consistency() -> list[dict]:
    """``12_CHECKLISTS.md`` § 0: the training-loop validation F1 of R0n must equal the scorer's k0 sigma0 value."""
    out = []
    for s in SEEDS:
        m = json.loads((REPO / f"results/s4_4/r0n/pcstgnn/seed_{s:05d}/metrics.json").read_text())["best_validation_graph_f1"]
        c = load_cells(s4_4_path("r0n", s))[(0, 0.0)]["f1"]
        out.append({"seed": s, "metrics_json_val_f1": m, "scorer_val_k0_sigma0": c, "abs_diff": abs(m - c)})
    return out


# ------------------------------------------------------------------------------------------
# T10 — noise and loss policies, test split, R5 vs R0n
# ------------------------------------------------------------------------------------------


def build_t10() -> list[dict]:
    rows = []
    r5 = {s: load_cells(stage_g_path("test", "r5", s)) for s in SEEDS}
    r0n = {s: load_cells(stage_g_path("test", "r0n", s)) for s in SEEDS}
    for k in (0, 10):
        for sg in SIGMAS:
            pr = paired([r5[s][(k, sg)]["f1"] for s in SEEDS], [r0n[s][(k, sg)]["f1"] for s in SEEDS])
            rows.append({"condition": f"noise σ={sg:g}", "k": k, **_paired_cols(pr), "source": "results/stage_g/test/{r5,r0n}"})
    for policy in ("random", "persistent", "contiguous", "targeted"):
        a_name, b_name = ("r5", "r0n") if policy == "random" else (f"r5_{policy}", f"r0n_{policy}")
        a = {s: load_cells(stage_g_path("test", a_name, s)) for s in SEEDS}
        b = {s: load_cells(stage_g_path("test", b_name, s)) for s in SEEDS}
        for k in (1, 2, 5, 10):
            pr = paired([a[s][(k, 0.0)]["f1"] for s in SEEDS], [b[s][(k, 0.0)]["f1"] for s in SEEDS])
            rows.append({"condition": f"policy {policy}", "k": k, **_paired_cols(pr),
                         "source": f"results/stage_g/test/{{{a_name},{b_name}}}"})
    return rows


def _paired_cols(pr: dict) -> dict:
    return {"R5": fmt(pr["mean_a"]), "R0ⁿ": fmt(pr["mean_b"]), "ΔF1": f"{pr['delta']:+.3f}",
            "seeds > 0": f"{pr['n_positive']}/{pr['n']}", "p (raw)": ("—" if pr["p"] is None else f"{pr['p']:.4f}"),
            "delta": pr["delta"], "p": pr["p"], "per_seed_delta": pr["per_seed_delta"]}


# ------------------------------------------------------------------------------------------
# T11 — model mismatch, validation split (from results/stage_g/analysis/summary.json)
# ------------------------------------------------------------------------------------------

MISMATCH_LABELS = {"tap_p1": "tap +1 step (all regulators)", "tap_m1": "tap −1 step (all regulators)",
                   "imp5": "line impedance ±5 % (per line)"}


def build_t11(summary_path: Path | None = None) -> list[dict]:
    summary = json.loads(Path(summary_path or REPO / "results/stage_g/analysis/summary.json").read_text())
    rows = []
    for table, label in MISMATCH_LABELS.items():
        mm = summary["mismatch_phase1_val"][table]
        pc = mm["per_cell"]
        row = {"table": label, "R5 > R0ⁿ (of 21)": f"{mm['r5_higher']}/{mm['rule_comparisons']}",
               "graceful (rule)": "yes" if mm["graceful"] else "no"}
        for cell in ("k10_sigma0.0", "k0_sigma0.01"):
            c = pc[cell]
            row[f"{cell}: R5 table / R5 nominal / R0ⁿ"] = (f"{c['r5_on_table_mean']:.3f} / {c['r5_nominal_mean']:.3f} / "
                                                           f"{c['r0n_mean']:.3f}")
        row["source"] = "results/stage_g/analysis/summary.json (mismatch_phase1_val)"
        rows.append(row)
    rt = summary["retrain_imp5_val_seed42"]
    for cell in ("k0_sigma0.0", "k10_sigma0.0"):
        if cell in rt:
            c = rt[cell]
            rows.append({"table": f"retrain on ±5 % table, seed 42, {cell}",
                         "R5 > R0ⁿ (of 21)": "1 seed: no rule", "graceful (rule)": "—",
                         "k10_sigma0.0: R5 table / R5 nominal / R0ⁿ": "", "k0_sigma0.01: R5 table / R5 nominal / R0ⁿ": "",
                         "retrain / nominal weights on ±5 % / nominal on nominal / R0ⁿ":
                             f"{c['retrain_on_imp5']:.3f} / {c['r5_nominal_weights_on_imp5']:.3f} / "
                             f"{c['r5_nominal_on_nominal']:.3f} / {c['r0n']:.3f}",
                         "source": "results/stage_g/analysis/summary.json (retrain_imp5_val_seed42)"})
    return rows


# ------------------------------------------------------------------------------------------
# T13 — statistics: paired t over seeds + event-level cluster bootstrap (test split)
# ------------------------------------------------------------------------------------------

#: (hypothesis, k, sigma, subset): subset "pooled" or "R=10" (the 10-ohm events)
T13_CELLS = (("H1", 0, 0.0, "R=10"), ("H2", 10, 0.0, "pooled"), ("H2", 10, 0.0005, "pooled"), ("H2", 10, 0.002, "pooled"),
             ("H2", 10, 0.01, "pooled"), ("H4", 0, 0.0005, "pooled"), ("H4", 0, 0.002, "pooled"), ("H4", 0, 0.01, "pooled"),
             ("H6", 0, 0.0, "pooled"))
#: pre-registered bootstrap pairs (14 § 3, amended): reference model name in results/stage_g/test
T13_PAIRS = (("R5 vs R0ⁿ", "r0n"), ("R5 vs R6ⁱ", "r6i"), ("R5 vs R2", "r2"))


def bootstrap_pair(ref: str, cells, n_boot: int = 10000, test_dir: Path | None = None) -> dict:
    """Event-level paired cluster bootstrap of R5 − ``ref`` on the test split, with ``gate6.analyze_cell``.

    ``analyze_cell`` names its two arms ``r5`` and ``r0n``; the reference model's runs are passed under the key
    ``r0n``. Subsets: all events (``pooled``) and the events at each resistance (``R=<r>``).
    """
    from fault_localization.stage_g import gate6 as G

    base = Path(test_dir or REPO / "results/stage_g/test")
    runs = {"r5": {s: G.Run(base / "r5" / f"seed_{s:05d}_predictions.npz") for s in SEEDS},
            "r0n": {s: G.Run(base / ref / f"seed_{s:05d}_predictions.npz") for s in SEEDS}}
    ref_run = runs["r5"][SEEDS[0]]
    for m, by_seed in runs.items():
        for s, r in by_seed.items():
            if not np.array_equal(r.data["window_ids"], ref_run.data["window_ids"]):
                raise ValueError(f"{m} seed {s}: windows differ from R5 seed {SEEDS[0]}")
    ev_names, ev_idx, res, _g = G.events_and_strata(ref_run)
    subsets = {"pooled": np.ones(len(ev_names), dtype=bool)}
    for r in sorted(set(np.round(res, 6).tolist())):
        subsets[f"R={r:g}"] = np.isclose(res, r)
    out = {}
    for k, sg in cells:
        out[(k, sg)] = G.analyze_cell(runs, k, sg, ev_idx, len(ev_names), subsets, list(SEEDS), n_boot)
    return {"n_events": int(len(ev_names)), "n_windows": int(len(ev_idx)), "cells": out}


def saved_cells(npz_path: Path) -> set[tuple[int, float]]:
    """The (k, sigma) cells stored in a predictions file."""
    with np.load(npz_path, allow_pickle=False) as z:
        return {(int(k), float(s)) for k, s in json.loads(str(z["cells"]))["cells"]}


def build_t13(n_boot: int = 10000, pairs=T13_PAIRS, boot: dict | None = None,
              test_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Rows for T13 and the raw bootstrap output (``boot`` may be passed in to skip the computation)."""
    base = Path(test_dir or REPO / "results/stage_g/test")
    boot = dict(boot or {})
    rows = []
    for pair, ref in pairs:
        have = saved_cells(base / ref / f"seed_{SEEDS[0]:05d}_predictions.npz")
        cells_needed = [c for c in sorted({(k, sg) for _h, k, sg, _sub in T13_CELLS}) if c in have]
        if pair not in boot:
            boot[pair] = bootstrap_pair(ref, cells_needed, n_boot=n_boot, test_dir=base)
        b = boot[pair]
        for hyp, k, sg, sub in T13_CELLS:
            if (k, sg) not in b["cells"] or (hyp == "H6" and ref != "r0n"):
                continue
            e = b["cells"][(k, sg)][sub]
            pr = paired(e["f1"]["r5"], e["f1"]["r0n"])
            label = hyp if ref == "r0n" else ("A1" if ref == "r6i" else "A2")
            row = {"pair": pair, "hypothesis": label, "family": "{H1–H4}" if ref == "r0n" else "{A1, A2}",
                   "cell": f"k={k}, σ={sg:g}, {sub}", "R5": fmt(pr["mean_a"]), "reference": fmt(pr["mean_b"]),
                   "ΔF1": f"{pr['delta']:+.3f}", "seeds > 0": f"{pr['n_positive']}/3",
                   "p (t, raw)": f"{pr['p']:.4f}", "p": pr["p"],
                   "95 % CI (event bootstrap)": f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]" if "ci95" in e else "—",
                   "ci95": e.get("ci95"), "n_events": e["n_events"], "delta_per_seed": e["delta_per_seed"]}
            if hyp == "H6":
                # H6 (14 § 1): clean accuracy preserved = |ΔF1| within 1 seed-std (of the reference); not a sign rule
                sd = float(np.std(e["f1"]["r0n"], ddof=1))
                row["family"] = "none (control)"
                row["supported (3/3 and CI excl. 0)"] = (f"rule is |Δ| ≤ 1 seed-std ({sd:.3f}): "
                                                         f"{'yes' if abs(pr['delta']) <= sd else 'no'}"
                                                         f"{' (R5 higher)' if pr['delta'] > 0 else ''}")
            else:
                row["supported (3/3 and CI excl. 0)"] = ("yes" if (e["sign_all_seeds_positive"] and e.get("ci_lower_above_zero"))
                                                         else "no")
            rows.append(row)
    for fam in ("{H1–H4}", "{A1, A2}"):
        fam_rows = [r for r in rows if r["family"] == fam]
        if fam_rows:
            for r, a in zip(fam_rows, holm([r["p"] for r in fam_rows])):
                r["p Holm (cells chosen post hoc)"] = f"{a:.4f}"
                r["p_holm"] = a
    return rows, boot


# ------------------------------------------------------------------------------------------
# T6 — baseline ladder (two blocks, one per split: the sources were scored on different splits)
# ------------------------------------------------------------------------------------------


def _row6(block, rung, name, split, k0, k10, audc_abs, n_seeds, params, protocol, source):
    return {"block": block, "rung": rung, "model": name, "split": split, "n_seeds": n_seeds, "learned parameters": params,
            "k=0": fmt(*k0) if isinstance(k0, tuple) else fmt(k0), "k=10": fmt(*k10) if isinstance(k10, tuple) else fmt(k10),
            "AUDC abs": "—" if audc_abs is None else fmt(audc_abs), "protocol": protocol, "source": source}


def build_t6() -> list[dict]:
    """The ladder B1ᵃ → B1 → B2 → classical → B3 → B4 → B5 with the split of every number stated."""
    rows = []
    val_proto = "26-class window macro-F1, validation split, zero_messages, 1 mask seed for B1/B1ᵃ/PSM (the audit scripts)"
    mf = json.loads((REPO / "results/physics_baseline/matched_filter.json").read_text())
    b = json.loads((REPO / "results/pcstgnn/audit/s2_3_b1/b1_full_eval.json").read_text())
    p = json.loads((REPO / "results/pcstgnn/audit/s2_4_psm/psm_full_eval.json").read_text())["validation"]
    mfc = mf["curve"]
    rows.append(_row6("validation", "B1ᵃ", "Rank-1 sag-template matched filter", "val", mfc["sigma0.0_k0"]["f1_mean"],
                      mfc["sigma0.0_k10"]["f1_mean"], None, 1, 0, val_proto, "results/physics_baseline/matched_filter.json"))
    rows.append(_row6("validation", "B1", "ΔV nearest signature (no profiling)", "val", b["b1_delta_v"]["f1_by_k"]["0"],
                      b["b1_delta_v"]["f1_by_k"]["10"], b["b1_delta_v"]["audc_absolute"], 1, 0, val_proto,
                      "results/pcstgnn/audit/s2_3_b1/b1_full_eval.json"))
    rows.append(_row6("validation", "B2", "PSM (profiled τ, R, L; argmin ρ)", "val", p["f1_by_k"]["0"], p["f1_by_k"]["10"], None, 1, 0,
                      val_proto, "results/pcstgnn/audit/s2_4_psm/psm_full_eval.json"))
    for label, rung, arm in (("R0ⁿ (noise-aug., no physics)", "B3′", "r0n"), ("R5 (PC-STGNN)", "B5", "r5")):
        v = {s: load_cells(s4_4_path(arm, s)) for s in SEEDS}
        rows.append(_row6("validation", rung, label, "val", mean_std([v[s][(0, 0.0)]["f1"] for s in SEEDS]),
                          mean_std([v[s][(10, 0.0)]["f1"] for s in SEEDS]), None, 3,
                          "95,258" if arm == "r0n" else "95,259",
                          "26-class window macro-F1, validation, Gate-4 scorer, 3 offsets, σ=0", f"results/s4_4/{arm}/pcstgnn/seed_*/scoring_full_grid.json"))
    test_proto = "26-class window macro-F1, test split, zero_messages"
    cl = {}
    with (REPO / "results/classical_baseline/classical_curves.csv").open() as fh:
        for r in csv.DictReader(fh):
            cl.setdefault(r["model"], {})[int(r["k"])] = float(r["f1_mean"])
    for m, label in (("logreg", "Logistic regression (summary features)"), ("random_forest", "Random forest (summary features)")):
        rows.append(_row6("test", "B2ᵇ", label, "test", cl[m][0], cl[m][10], audc(K_LEVELS, [cl[m][k] for k in K_LEVELS], False), 1,
                          "n/a (classical)", test_proto + ", 1 fit, 3 mask seeds", "results/classical_baseline/classical_curves.csv"))
    psm = {s: {k: v["psm"] for (k, sg), v in load_cells(stage_g_path("test", "r5", s)).items() if sg == 0.0} for s in SEEDS}
    rows.append(_row6("test", "B2", "PSM (profiled τ, R, L; argmin ρ)", "test", psm[42][0], psm[42][10],
                      audc(K_LEVELS, [psm[42][k] for k in K_LEVELS], False), 1, 0,
                      "26-class window macro-F1, test, Stage G scorer, σ=0 (read from the R5 scorer files; identical in all 3)",
                      "results/stage_g/test/r5/seed_*_scoring.json (f1_psm_mean)"))
    for model, rung, label in (("rgsage_max", "B3", "RGSAGE-Max, standard (Paper 1)"),):
        c = paper1_curve(REPO / PAPER1_CURVES["standard"], model)
        rows.append(_row6("test", rung, label, "test", mean_std([c[s][0] for s in SEEDS]), mean_std([c[s][10] for s in SEEDS]),
                          mean_std([audc(K_LEVELS, [c[s][k] for k in K_LEVELS], False) for s in SEEDS])[0], 3, "95,258",
                          test_proto + ", Paper-1 evaluator", PAPER1_CURVES["standard"]))
    c = paper1_curve(REPO / PAPER1_CURVES["dropout-aware"], "rgsage_max")
    rows.append(_row6("test", "B4 = R0", "RGSAGE-Max, dropout-aware (Paper 1)", "test", mean_std([c[s][0] for s in SEEDS]),
                      mean_std([c[s][10] for s in SEEDS]), mean_std([audc(K_LEVELS, [c[s][k] for k in K_LEVELS], False) for s in SEEDS])[0],
                      3, "95,258", test_proto + ", Paper-1 evaluator", PAPER1_CURVES["dropout-aware"]))
    for label, rung, arm in (("R0ⁿ (noise-aug., no physics)", "B3′", "r0n"), ("R5 (PC-STGNN)", "B5", "r5")):
        cur = stage_g_curve(arm)
        rows.append(_row6("test", rung, label, "test", mean_std([cur[s][0] for s in SEEDS]), mean_std([cur[s][10] for s in SEEDS]),
                          mean_std([audc(K_LEVELS, [cur[s][k] for k in K_LEVELS], False) for s in SEEDS])[0], 3,
                          "95,258" if arm == "r0n" else "95,259", test_proto + ", Stage G scorer, 3 offsets, σ=0",
                          f"results/stage_g/test/{arm}/seed_*_scoring.json"))
    return rows


# ------------------------------------------------------------------------------------------
# T9 — silent failures: detection AUROC (S5.1/S5.2) and the G-TF2 gate (validation half B, sigma0 = 1e-2)
# ------------------------------------------------------------------------------------------

DETECTION_FILES = {0.0: "results/pcstgnn/audit/s5_1_silent_detection/silent_detection.json",
                   0.01: "results/stage_f/silent_detection_noise/sigma_0.01/silent_detection.json"}


def build_t9_detection() -> list[dict]:
    """PSM-refit residual AUROC per corruption condition: LOO and in-sample, all windows, at sigma0 = 0 and 1e-2."""
    per: dict = {}
    for sig, rel in DETECTION_FILES.items():
        rows = json.loads((REPO / rel).read_text())["rows"]
        for r in rows:
            if r["refit"] == "psm" and r["stratum"] == "all":
                per.setdefault(r["condition"], {}).setdefault(sig, {})[r["score"]] = r
    out = []
    for cond, by_sig in per.items():
        row = {"condition": cond, "corruption": next(iter(by_sig[0.0].values()))["corruption"],
               "sensors corrupted (s)": next(iter(by_sig[0.0].values()))["n_sensors"]}
        for sig in (0.0, 0.01):
            if sig in by_sig and {"loo", "ins"} <= set(by_sig[sig]):
                row[f"AUROC LOO σ0={sig:g}"] = f"{by_sig[sig]['loo']['auroc']:.4f}"
                row[f"AUROC in-sample σ0={sig:g}"] = f"{by_sig[sig]['ins']['auroc']:.4f}"
                row[f"TPR@FPR0.05 LOO σ0={sig:g}"] = f"{by_sig[sig]['loo']['tpr_at_fpr05']:.3f}"
        out.append(row)
    return out


def build_t9_gate(arm_dirs: tuple = ("gtf2", "gtf2r"), subdir: str = "core") -> list[dict]:
    """G-TF2 (and the residual-only arm where present): F1 without gate / with gate / oracle gate, per condition.

    Default = validation half B (``results/stage_f/{gtf2,gtf2r}/core``). ``build_t9_gate_test`` reads the
    test-split run (``results/stage_f/gtf2_test``, pre-registered in ``PC-STGNN/stage_f_gtf2_test_preregistration.md``).
    """
    rows = []
    for arm_dir in arm_dirs:
        p = REPO / (f"results/stage_f/{arm_dir}/{subdir}/summary.json" if subdir else f"results/stage_f/{arm_dir}/summary.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        sig = d["sigma"]
        if not isinstance(sig, dict) or "0.01" not in sig or "conditions" not in sig["0.01"]:
            continue
        gk = d["arm"]                                          # the gated value is stored under the arm's own key ("gtf2", "resid05")
        for cond, c in sig["0.01"]["conditions"].items():
            n, g, o = np.array(c["nogate"]), np.array(c[gk]), np.array(c.get("oracle", [np.nan] * 3))
            pr = paired(g, n)
            rows.append({"arm": d["arm"], "condition": cond, "no gate": fmt(*mean_std(n)), "gate": fmt(*mean_std(g)),
                         "oracle gate": fmt(*mean_std(o)) if not np.isnan(o).any() else "—", "gate − no gate": f"{pr['delta']:+.4f}",
                         "seeds > 0": f"{pr['n_positive']}/3", "p (raw t)": f"{pr['p']:.4f}",
                         "recovered of gap to oracle": fmt(c["recovered_mean_over_seeds"]) if "recovered_mean_over_seeds" in c else "—",
                         "FPR clean sensors": fmt(float(np.mean(c["realized_fpr_clean_sensors_of_corrupted_windows"])))
                         if "realized_fpr_clean_sensors_of_corrupted_windows" in c else "—",
                         "role": d.get("decision_rule_gtf2_sigma1e-2", {}).get(cond.rsplit("_s", 1)[0], {}).get("role", ""),
                         "delta_per_seed": [float(x) for x in (g - n)]})
        cl = sig["0.01"].get("clean")
        if cl:
            n, g = np.array(cl["nogate"]), np.array(cl[gk])
            rows.append({"arm": d["arm"], "condition": "clean windows (no corruption)", "no gate": fmt(*mean_std(n)),
                         "gate": fmt(*mean_std(g)), "oracle gate": "—", "gate − no gate": f"{(g - n).mean():+.4f}",
                         "seeds > 0": f"{int((g > n).sum())}/3", "p (raw t)": "—", "recovered of gap to oracle": "—",
                         "FPR clean sensors": fmt(float(np.mean(cl["realized_fpr_clean_windows"]))), "role": "cost check",
                         "delta_per_seed": [float(x) for x in (g - n)]})
    return rows


def build_t9_gate_test() -> list[dict]:
    """G-TF2 on the test split (247,000 windows, thresholds frozen from validation half A). Same columns as T9b."""
    return build_t9_gate(("gtf2_test",), "")
