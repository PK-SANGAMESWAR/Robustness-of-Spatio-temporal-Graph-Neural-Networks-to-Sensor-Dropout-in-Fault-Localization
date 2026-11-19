"""Stage B tables and figures that Stage G still owed: T2, T3, T4, F-B3, F-B4 (error-histogram variant), F-B6.

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/build_stage_g_stage_b.py

CPU only, no sealed data. Every number is read from a file at build time:
  results/pcstgnn/audit/{s1_4_7,s1_5_rgrid_diagnostic,s1_7_discretization,timing.json,window_census.json,build_path.json}
  operator/{grid.json,build_log.csv,mu.npy}, fault_localization/config/constants.py, scripts/generate_ood_sets.py (SET_SPECS)
Writes results/stage_g/tables/{T2,T3,T4}_*.{md,csv} and results/stage_g/figures/F-B{3,4,6}_*.{png,pdf}, plus stage_b_data.json.
F-B1 has its own script (``build_stage_g_fb1.py``); F-B5 reads the output of ``run_fb5_floor_by_R.py``.
F-B4 as written in 10_FIGURES_TABLES.md needs L* against the stored multiplier; the saved file keeps only the grid-step error, so the figure shows that.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.config import constants as C  # noqa: E402

AUD = REPO / "results/pcstgnn/audit"
TAB = REPO / "results/stage_g/tables"
FIG = REPO / "results/stage_g/figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def jl(p: Path):
    return json.loads(p.read_text())


def write_table(name: str, header: list[str], rows: list[list], title: str, note: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md += ["", note, ""]
    (TAB / f"{name}.md").write_text("\n".join(md))


def style(ax, xl, yl):
    ax.set_xlabel(xl, color=MUTED)
    ax.set_ylabel(yl, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def sealed_spec() -> dict:
    """SET_SPECS from generate_ood_sets.py, read as text-free import of the dict only (no data access)."""
    spec = importlib.util.spec_from_file_location("gen_ood", REPO / "scripts/generate_ood_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SET_SPECS


def t2(data: dict) -> None:
    cen = jl(AUD / "window_census.json")["census"]
    sp = sealed_spec()
    n_ood = {k: len(C.FAULT_BUSES) * len(C.FAULT_TYPES) * len(v["resistances"]) * len(v["run_ids"]) for k, v in sp.items()}
    rows = [
        ["Training corpus", "25 fault buses × 11 types × R ∈ " + str(C.FAULT_RESISTANCES) + " Ω = " + str(C.NUM_SCENARIOS) + " scenarios",
         f"load multiplier {C.LOAD_MULTIPLIER_LOW} to {C.LOAD_MULTIPLIER_HIGH}; 25 μPMU buses", "dataset_fullscale_50 (frozen)"],
        ["Test split windows", f"{cen['n_windows']:,} ({cen['n_fault_windows']:,} fault, {cen['n_nofault_windows']:,} no-fault)",
         f"{cen['zero_post_inception']:,} fault windows with 0 post-inception samples", "audit/window_census.json"],
        ["OOD pilot (throwaway)", f"R ∈ {sp['pilot']['resistances']} Ω, {len(sp['pilot']['run_ids'])} runs per cell, {n_ood['pilot']:,} events",
         "Gate-2 set; numbers never reported", "generate_ood_sets.py SET_SPECS"],
        ["OOD sealed set", f"R ∈ {sp['test']['resistances']} Ω, {len(sp['test']['run_ids'])} runs per cell, {n_ood['test']:,} events",
         "opened once (14_PREREGISTRATION § 6)", "generate_ood_sets.py SET_SPECS"],
    ]
    write_table("T2_dataset_ood_config", ["Set", "Configuration", "Note", "Source"], rows,
                "T2 Dataset and OOD set configuration",
                "Counts are products of the constants and of SET_SPECS; no sealed event was read to build this table.")
    data["T2"] = {"n_ood_events": n_ood}


def t3_t4(data: dict) -> None:
    g = jl(REPO / "operator/grid.json")
    tm = jl(AUD / "timing.json")
    s = jl(AUD / "s1_4_7/s1_4_7_summary.json")
    mu = np.load(REPO / "operator/mu.npy", mmap_mode="r")
    t = tm["tables"]["0.05"]
    rows = [
        ["Axes (μ[f,τ,R,L,i,p])", " × ".join(str(x) for x in mu.shape)],
        ["Fault buses / types", f"{len(g['fault_buses'])} / {len(g['fault_types'])}"],
        ["R grid", f"{len(g['r_grid'])} points, {g['r_grid'][0]} to {g['r_grid'][-1]} Ω"],
        ["L grid", f"{len(g['l_grid'])} points, step {g['l_step']}, {g['l_grid'][0]} to {g['l_grid'][-1]}"],
        ["Solves (OpenDSS)", f"{g['n_solves']:,}; non-converged {g['n_non_converged']}; NaN {g['n_nan']}"],
        ["Solve mode / reuse", f"{g['solve_mode']} / {g['reuse_limit']}"],
        ["Table file size", f"{mu.nbytes / 1e6:.1f} MB (float32 mu.npy)"],
        ["Build cost (estimate)", f"{t['serial_minutes']:.2f} min serial, {t['parallel_minutes']:.2f} min on {tm['workers_assumed']} workers"],
    ]
    write_table("T3_signature_table_config", ["Item", "Value"], rows, "T3 Signature table configuration",
                "Sources: operator/grid.json, results/pcstgnn/audit/timing.json. Build cost is the audit's estimate from per-solve timing, not a stopwatch of the real build.")
    b, ld = s["S1.4_bias"], s["S1.5_load"]
    rows4 = [["all", b["n_events"], f"{b['median_abs_e_max_channel']:.2e}", f"{b['p95_abs_e_max_channel']:.2e}", f"{b['max_abs_e']:.2e}"]]
    for r, v in b["by_R"].items():
        rows4.append([f"R = {r} Ω", v["n"], f"{v['median']:.2e}", f"{v['p95']:.2e}", f"{v['max']:.2e}"])
    for ft, v in b["by_type"].items():
        rows4.append([ft, v["n"], f"{v['median']:.2e}", f"{v['p95']:.2e}", f"{v['max']:.2e}"])
    rows4 += [["L* within 1 step", ld["n_events"], f"{ld['l_star_within_1_step_frac']:.4f}", "", ""],
              ["L* exact", ld["n_events"], f"{ld['l_star_exact_frac']:.4f}", "", ""],
              ["τ* correct", ld["n_events"], f"{ld['tau_star_correct_frac']:.4f}", "", ""],
              ["R* within 1 step", ld["n_events"], f"{ld['r_star_within_1_step_frac']:.4f}", "", ""]]
    write_table("T4_fidelity_summary", ["Subset", "n events", "median |e| (max channel, p.u.) or fraction", "p95 (p.u.)", "max (p.u.)"], rows4,
                "T4 Operator fidelity on the training events",
                "Source: results/pcstgnn/audit/s1_4_7/s1_4_7_summary.json. The |e| rows are the table-versus-corpus voltage gap; the audit note calls S1.4 a construction check, not an independent fidelity test. Fraction rows are in column 3.")
    data["T3_T4"] = {"n_solves": g["n_solves"], "l_within_1": ld["l_star_within_1_step_frac"]}


def f_b3(data: dict) -> None:
    b = jl(AUD / "s1_4_7/s1_4_7_summary.json")["S1.4_bias"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, key, order in ((axes[0], "by_R", list(b["by_R"])), (axes[1], "by_type", list(b["by_type"]))):
        xs = np.arange(len(order))
        med = [b[key][o]["median"] for o in order]
        p95 = [b[key][o]["p95"] for o in order]
        mx = [b[key][o]["max"] for o in order]
        ax.bar(xs - 0.27, med, 0.26, color=BLUE, label="median")
        ax.bar(xs, p95, 0.26, color=GREEN, label="p95")
        ax.bar(xs + 0.27, mx, 0.26, color=ORANGE, label="max")
        ax.set_xticks(xs, [f"{o} Ω" if key == "by_R" else o for o in order], fontsize=7, rotation=0 if key == "by_R" else 60)
        style(ax, "fault resistance" if key == "by_R" else "fault type", "|e| (p.u.)" if key == "by_R" else "")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("F-B3  Table-versus-corpus voltage gap |e|, 28,875 training events", fontsize=9, color=INK, x=0.01, y=1.04, ha="left")
    save(fig, "F-B3_fidelity_by_R_type")


def f_b4(data: dict) -> None:
    z = np.load(AUD / "s1_4_7/load_errors_by_event.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    d = {}
    for ax, key, col, lab in ((axes[0], "l_err", BLUE, "L* error (grid steps)"), (axes[1], "r_err", ORANGE, "R* error (grid steps)")):
        v = z[key]
        vals, cnt = np.unique(v, return_counts=True)
        ax.bar(vals, cnt / v.size, color=col, width=0.8)
        style(ax, lab, "fraction of events")
        ax.set_yscale("log")
        d[key] = {int(a): float(c / v.size) for a, c in zip(vals, cnt)}
    fig.suptitle("F-B4 (variant)  Profile-fit error in grid steps; L* is on the stored-multiplier grid", fontsize=9, color=INK, x=0.01, y=1.04, ha="left")
    save(fig, "F-B4_profile_error_hist")
    data["F-B4"] = d


def f_b6(data: dict) -> None:
    rg = jl(AUD / "s1_5_rgrid_diagnostic/rgrid_diagnostic.json")
    dis = jl(AUD / "s1_7_discretization/s1_7_discretization.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    names = ["current", "hybrid", "dense25"]
    n = [rg[k]["n_r_grid_points"] for k in names]
    ax = axes[0]
    ax.plot(n, [rg[k]["l_star_within_1_step_frac"] for k in names], color=BLUE, marker="o", label="L* within 1 step")
    ax.plot(n, [rg[k]["r_star_exact_frac"] for k in names], color=ORANGE, marker="o", label="R* exact")
    ax.axhline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
    style(ax, "R grid points", "fraction of 28,875 events")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("S1.5 diagnostic, three R grids", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    pts = [dis["halved_9pt"]["n_r_grid_points"], dis["full_18pt"]["n_r_grid_points"]]
    f1 = [dis["halved_9pt"]["macro_f1_25class"], dis["full_18pt"]["macro_f1_25class"]]
    ax.plot(pts, f1, color=BLUE, marker="o")
    for x, y in zip(pts, f1):
        ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 7), textcoords="offset points", fontsize=8, ha="center", color=INK)
    ax.set_xlim(6, 21)
    ax.set_ylim(min(f1) - 0.004, max(f1) + 0.004)
    style(ax, "R grid points", "PSM macro-F1 (train)")
    ax.set_title(f"S1.7 discretization (σ_ε = {dis['sigma_eps']:.2e})", fontsize=9, color=INK, loc="left")
    fig.suptitle("F-B6  Effect of the R-grid density", fontsize=9, color=INK, x=0.01, y=1.04, ha="left")
    save(fig, "F-B6_rgrid_density")
    data["F-B6"] = {"s1_5": {k: {kk: rg[k][kk] for kk in ("n_r_grid_points", "l_star_within_1_step_frac", "r_star_exact_frac")} for k in names},
                    "s1_7": {"points": pts, "f1": f1}}


def f_b2(data: dict) -> None:
    d = jl(REPO / "results/pcstgnn/operator/lgrid_post_rgrid_fix/lgrid_summary.json")
    ie = d["interpolation_error"]
    steps = [e["l_step"] for e in ie]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for key, col, lab in (("max", ORANGE, "max"), ("p99", GREEN, "p99"), ("p50", BLUE, "median")):
        ax.plot(steps, [e[key] for e in ie], color=col, marker="o", label=lab)
    sig = sorted({m["sigma_eps"] for m in d["margin"] if not m["is_discretisation_sigma"]})
    for sg in sig:
        ax.axhline(sg, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate(f"σ_ε = {sg:g}", (max(steps), sg), xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED, ha="right")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks(steps, [str(x) for x in steps])
    style(ax, "L-grid step (finer to the right)", "error vs 0.0125 grid (p.u.)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("F-B2  L-axis linear interpolation error (max is of tap-step size)", fontsize=8, color=INK, loc="left")
    save(fig, "F-B2_lgrid_interpolation")
    data["F-B2"] = {"source": "results/pcstgnn/operator/lgrid_post_rgrid_fix/lgrid_summary.json",
                    "steps": steps, "max": [e["max"] for e in ie], "p99": [e["p99"] for e in ie], "p50": [e["p50"] for e in ie],
                    "sigma_lines": sig}


def f_b5(data: dict) -> None:
    d = jl(AUD / "s1_6_floor_by_R/floor_by_R.json")
    rs = list(d["by_R"])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.arange(len(rs))
    med = np.array([d["by_R"][r]["median"] for r in rs])
    lo = np.array([d["by_R"][r]["p10"] for r in rs])
    hi = np.array([d["by_R"][r]["p90"] for r in rs])
    ax.axhspan(0.3, 3.0, color=GRID, alpha=0.7, zorder=0)
    ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="o", color=BLUE, capsize=4, markersize=6, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{r} Ω" for r in rs])
    ax.set_xlim(-0.5, len(rs) - 0.5)
    style(ax, "fault resistance", "r-bar at the true hypothesis")
    ax.set_title(f"F-B5  median, p10 to p90; band = target [0.3, 3]; σ_ε = {d['sigma_eps']:.3g}", fontsize=8, color=INK, loc="left")
    save(fig, "F-B5_floor_by_R")
    data["F-B5"] = d


def main() -> int:
    data: dict = {}
    t2(data)
    t3_t4(data)
    f_b2(data)
    f_b5(data)
    f_b3(data)
    f_b4(data)
    f_b6(data)
    (FIG / "stage_b_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("ok", sorted(p.name for p in TAB.glob("T[234]_*")), sorted(p.name for p in FIG.glob("F-B*")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
