"""Stage G analysis of the finished queue (validation mismatch study, B2a, test-split S3 and S4 policies). CPU only.

Reads the scoring JSON files under ``results/stage_g/`` and ``results/s4_4/{r0n,r5}``. Never opens the sealed set.
Writes ``results/stage_g/analysis/summary.json``. Paired t-tests over seeds 42/43/44 (n = 3); Holm over the 7 H2/H4 cells
is a post hoc family. Trends are trends.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SG = REPO / "results/stage_g"
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.0005, 0.002, 0.01)
H2 = [(10, s) for s in SIGMAS]
H4 = [(0, s) for s in SIGMAS[1:]]
RULE_CELLS = H2 + H4


def _cells(path: Path, key: str) -> dict:
    d = json.loads(path.read_text())["cells"]
    out = {}
    for name, v in d.items():
        k, s = name.split("_sigma")
        out[(int(k[1:]), float(s))] = v[key]
    return out


def fused(path):            # Gate-4 scorer files (R0n, R5) use f1_fused_mean; Stage G scorer uses f1_mean
    d = json.loads(path.read_text())["cells"]
    key = "f1_fused_mean" if "f1_fused_mean" in next(iter(d.values())) else "f1_mean"
    return _cells(path, key)


def s4(arm, seed):
    return fused(REPO / f"results/s4_4/{arm}/pcstgnn/seed_{seed:05d}/scoring_full_grid.json")


def sg(split, name, seed):
    return fused(SG / split / name / f"seed_{seed:05d}_scoring.json")


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    p = float(stats.ttest_rel(a, b).pvalue) if np.ptp(d) > 0 or d.mean() != 0 else 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(d.mean()), "n_positive": int((d > 0).sum()), "p": p,
            "per_seed_delta": [float(x) for x in d]}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return [float(x) for x in adj]


def main() -> int:
    out = {}
    r0n_val = {s: s4("r0n", s) for s in SEEDS}
    r5_val = {s: s4("r5", s) for s in SEEDS}
    # ---- mismatch Phase 1 (validation)
    mm = {}
    for table in ("tap_p1", "tap_m1", "imp5"):
        sc = {s: sg("val", f"r5_{table}", s) for s in SEEDS}
        comps = [{"cell": f"k{k}_sigma{sg_}", "seed": s, "r5_on_table": sc[s][(k, sg_)], "r0n": r0n_val[s][(k, sg_)],
                  "r5_nominal": r5_val[s][(k, sg_)], "r5_higher": bool(sc[s][(k, sg_)] > r0n_val[s][(k, sg_)])}
                 for k, sg_ in RULE_CELLS for s in SEEDS]
        n = sum(c["r5_higher"] for c in comps)
        per_cell = {f"k{k}_sigma{x}": {"r5_on_table_mean": float(np.mean([sc[s][(k, x)] for s in SEEDS])),
                                       "r0n_mean": float(np.mean([r0n_val[s][(k, x)] for s in SEEDS])),
                                       "r5_nominal_mean": float(np.mean([r5_val[s][(k, x)] for s in SEEDS]))} for k, x in RULE_CELLS}
        mm[table] = {"rule_comparisons": len(comps), "r5_higher": n, "graceful": bool(n == len(comps)), "per_cell": per_cell, "comparisons": comps}
    out["mismatch_phase1_val"] = mm
    # ---- retrain on imp5 table (seed 42, val) and B2a (seed 42)
    s42 = 42
    rt = sg("val", "retrain_imp5", s42)
    ri = sg("val", "r5_imp5", s42)
    out["retrain_imp5_val_seed42"] = {f"k{k}_sigma{x}": {"retrain_on_imp5": rt[(k, x)], "r5_nominal_weights_on_imp5": ri[(k, x)],
                                                       "r5_nominal_on_nominal": r5_val[s42][(k, x)], "r0n": r0n_val[s42][(k, x)]} for k, x in sorted(ri)}
    b_val = sg("val", "b2a", s42)
    b_test = sg("test", "b2a", s42)
    r5_test = {s: sg("test", "r5", s) for s in SEEDS}
    r0n_test = {s: sg("test", "r0n", s) for s in SEEDS}
    out["b2a_seed42"] = {"val": {f"k{k}_sigma{x}": {"b2a": b_val[(k, x)], "r5": r5_val[s42][(k, x)], "r0n": r0n_val[s42][(k, x)]} for k, x in sorted(b_val)},
                         "test": {f"k{k}_sigma{x}": {"b2a": b_test[(k, x)], "r5": r5_test[s42][(k, x)], "r0n": r0n_test[s42][(k, x)]} for k, x in sorted(b_test)}}
    # ---- S3 test split, R5 vs R0n (paired over seeds), all 20 cells
    s3 = {}
    for cell in sorted(r5_test[42]):
        s3[f"k{cell[0]}_sigma{cell[1]}"] = paired([r5_test[s][cell] for s in SEEDS], [r0n_test[s][cell] for s in SEEDS])
    rule = [f"k{k}_sigma{x}" for k, x in RULE_CELLS]
    adj = holm([s3[c]["p"] for c in rule])
    for c, a in zip(rule, adj):
        s3[c]["p_holm_7cell_post_hoc"] = a
    out["s3_test_r5_vs_r0n"] = s3
    others = {}
    for arm in ("r2", "r6i"):
        t = {s: sg("test", arm, s) for s in SEEDS}
        others[arm] = {f"k{c[0]}_sigma{c[1]}": paired([t[s][c] for s in SEEDS], [r5_test[s][c] for s in SEEDS]) for c in sorted(t[42])}
    out["s3_test_vs_r5_delta_is_arm_minus_r5"] = others
    # ---- S4 policies (test), R5 vs R0n
    pol = {}
    for policy in ("persistent", "contiguous", "targeted"):
        a = {s: sg("test", f"r5_{policy}", s) for s in SEEDS}
        b = {s: sg("test", f"r0n_{policy}", s) for s in SEEDS}
        pol[policy] = {f"k{c[0]}": paired([a[s][c] for s in SEEDS], [b[s][c] for s in SEEDS]) for c in sorted(a[42])}
    out["s4_policies_test_r5_vs_r0n"] = pol
    (SG / "analysis").mkdir(exist_ok=True)
    (SG / "analysis/summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("mismatch graceful:", {t: f"{v['r5_higher']}/{v['rule_comparisons']}" for t, v in mm.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
