"""Layer 0 metrics, O19 step 2 (author 2026-10-30), `val` only, seed 42, k in {0, 1, 2, 5, 10}. `test` is not touched.

Held-out store = primary (O20), standard store = secondary. Post-fault windows (w >= 21) primary, all windows secondary.
The decision rule and the gate-threshold method are fixed in reports/layer0/DECISION_RULE.md (committed before this ran).
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.layer0
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from gridsense.config import TRACK
from gridsense.data.splits import cal_split, split_events
from gridsense.eval import metrics as M
from gridsense.eval.layer0_knn import knn_path
from gridsense.eval.s2_k0 import LAT, boot_ci, split_arrays

SMOKE = os.environ.get("L0_SMOKE") == "1"        # smoke test: fewer k, fewer resamples, output to scratch (never a result)
KS = (0, 1) if SMOKE else (0, 1, 2, 5, 10)
N_BOOT, SEED_BOOT, KNN = (20 if SMOKE else 1000), 0, 10
TARGETS = (0.95, 0.98)
OUT = TRACK / "reports" / "layer0"


class Data:
    """`val` windows without w0020, cal-A / cal-B by run, the k = 0 isotonic map (S2: fitted on cal-A, all unmasked windows)."""

    def __init__(self) -> None:
        ids, keep, votes0, y, runs, w = split_arrays("val")
        self.sel = np.flatnonzero(keep)
        self.y, self.runs, self.w = y[self.sel], runs[self.sel], w[self.sel]
        self.post = self.w >= 21
        a_runs, b_runs = (set(r) for r in cal_split(split_events("val"), seed=42))
        self.in_a, self.in_b = np.isin(self.runs, list(a_runs)), np.isin(self.runs, list(b_runs))
        v = votes0[self.sel]
        top1, correct = v.max(1) / 25.0, v.argmax(1) == self.y
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(top1[self.in_a], correct[self.in_a])
        saved = np.load(TRACK / "eval" / "calibration_k0_val.npz")["isotonic"][self.sel]
        self.iso_matches_s2 = bool(np.allclose(self.iso.predict(top1), saved, atol=1e-6))
        self.n_boot_runs = len(np.unique(self.runs[self.in_b]))


class Cell:
    """One (k, mask offset): STGNN outputs and neighbour features of the selected windows."""

    def __init__(self, d: Data, k: int, o: int):
        votes = np.load(LAT / "val" / f"votes_k{k}_o{o}.npy")[d.sel]
        part = np.sort(votes, axis=1)
        self.k, self.o = k, o
        self.top1, self.margin = part[:, -1] / 25.0, (part[:, -1] - part[:, -2]) / 25.0
        self.pred = votes.argmax(1)
        self.correct = self.pred == d.y
        self.p = d.iso.predict(self.top1)
        z = np.load(knn_path(k, o))
        assert (z["rows"] == d.sel).all()
        self.knn = {s: {"agree": z[f"{s}_agree_{KNN}"], "nbrerr": z[f"{s}_nbrerr_{KNN}"], "nnsim": z[f"{s}_nnsim"],
                        "vote": {kk: z[f"{s}_vote_{kk}"] for kk in (5, 10, 20)}} for s in ("heldout", "standard")}
        self.votes_argmax = self.pred


def _counts(d: Data, rng: np.random.Generator) -> np.ndarray:
    r = d.n_boot_runs
    c = np.zeros((N_BOOT, r), dtype=np.int16)
    for i in range(N_BOOT):
        c[i] = np.bincount(rng.integers(0, r, r), minlength=r)
    return c


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _fit_combo(c: Cell, store: str, fit_mask: np.ndarray) -> np.ndarray:
    X = np.c_[c.p, c.knn[store]["agree"]]
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit_mask], c.correct[fit_mask])
    return lr.predict_proba(X)[:, 1]


def decision(d: Data, cells: dict[int, Cell], store: str, counts: np.ndarray, post_only: bool = True) -> dict:
    """AUROC(error detection) of (b) logistic[p_iso, agreement] minus (a) p_iso, on cal-B; combination fitted on cal-A (DECISION_RULE.md § 2)."""
    sel = d.post if post_only else np.ones_like(d.post)
    use, fit = d.in_b & sel, d.in_a & sel
    _, inv = np.unique(d.runs[use], return_inverse=True)
    ks = sorted(cells)
    ga, gb = {}, {}
    for k in ks:
        c = cells[k]
        pc = _fit_combo(c, store, fit)
        ga[k] = M.TieGroups(-c.p[use], ~c.correct[use])
        gb[k] = M.TieGroups(-pc[use], ~c.correct[use])
    # pooled over k: one logistic fit on pooled cal-A, one AUROC on the pooled cal-B windows
    Xf = np.concatenate([np.c_[cells[k].p, cells[k].knn[store]["agree"]][fit] for k in ks])
    yf = np.concatenate([cells[k].correct[fit] for k in ks])
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xf, yf)
    pool_a = M.TieGroups(-np.concatenate([cells[k].p[use] for k in ks]), ~np.concatenate([cells[k].correct[use] for k in ks]))
    pool_b = M.TieGroups(-np.concatenate([lr.predict_proba(np.c_[cells[k].p, cells[k].knn[store]["agree"]])[use][:, 1] for k in ks]),
                         ~np.concatenate([cells[k].correct[use] for k in ks]))
    point_a = {k: ga[k].auroc() for k in ks}
    point_b = {k: gb[k].auroc() for k in ks}
    boot = np.empty((N_BOOT, len(ks) + 1))
    pool_boot = np.empty(N_BOOT)
    for r in range(N_BOOT):
        w = counts[r][inv].astype(float)
        diffs = [gb[k].auroc(w) - ga[k].auroc(w) for k in ks]
        boot[r, :-1], boot[r, -1] = diffs, np.mean(diffs)
        pool_boot[r] = pool_b.auroc(np.tile(w, len(ks))) - pool_a.auroc(np.tile(w, len(ks)))
    out = {"store": store, "windows": "post_fault" if post_only else "all", "n_cal_b_windows": int(use.sum()), "n_cal_b_runs": int(len(np.unique(d.runs[use]))),
           "logistic_coef_pooled": [float(x) for x in lr.coef_[0]], "per_k": {}}
    for i, k in enumerate(ks):
        out["per_k"][k] = {"auroc_a_iso": point_a[k], "auroc_b_combo": point_b[k], "diff": point_b[k] - point_a[k], "diff_ci": _ci(boot[:, i])}
    dm = float(np.mean([point_b[k] - point_a[k] for k in ks]))
    out["primary_mean_over_k"] = {"diff": dm, "ci": _ci(boot[:, -1]), "ci_lower_above_zero": bool(np.percentile(boot[:, -1], 2.5) > 0)}
    out["pooled_over_k"] = {"auroc_a": pool_a.auroc(), "auroc_b": pool_b.auroc(), "diff": pool_b.auroc() - pool_a.auroc(), "ci": _ci(pool_boot),
                            "ci_lower_above_zero": bool(np.percentile(pool_boot, 2.5) > 0)}
    return out


def signals(c: Cell, d: Data, fit_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Error scores (higher = more likely a wrong prediction)."""
    s = {"raw_vote_share": -c.top1, "isotonic_prob": -c.p, "margin": -c.margin}
    for st in ("heldout", "standard"):
        s[f"knn_agreement_{st}"] = -c.knn[st]["agree"]
        s[f"neighbour_error_rate_{st}"] = c.knn[st]["nbrerr"]
        s[f"nearest_similarity_{st}"] = -c.knn[st]["nnsim"]
        s[f"combo_prob+agreement_{st}"] = -_fit_combo(c, st, fit_mask)
    return s


def per_k_metrics(d: Data, cells: dict[int, Cell], counts: np.ndarray) -> dict:
    out = {}
    _, inv_post = np.unique(d.runs[d.in_b & d.post], return_inverse=True)
    for k, c in cells.items():
        r: dict = {"stgnn": {"acc_val_all": float(c.correct.mean()), "acc_val_post": float(c.correct[d.post].mean()),
                             "macro_f1_val_all": float(f1_score(d.y, c.pred, labels=list(range(26)), average="macro", zero_division=0)),
                             "acc_cal_b_post": float(c.correct[d.in_b & d.post].mean())}}
        for name, m in (("post", d.in_b & d.post), ("all", d.in_b)):
            r[f"calibration_cal_b_{name}"] = {
                "ece_iso": M.ece_equal_mass(c.p[m], c.correct[m]), "brier_iso": M.brier(c.p[m], c.correct[m]),
                "ece_raw_share": M.ece_equal_mass(c.top1[m], c.correct[m]), "brier_raw_share": M.brier(c.top1[m], c.correct[m]),
                "mean_p_iso": float(c.p[m].mean()), "accuracy": float(c.correct[m].mean())}
            r[f"selective_cal_b_{name}"] = {
                "aurc_iso": M.aurc(c.p[m], c.correct[m]), "aurc_raw_share": M.aurc(c.top1[m], c.correct[m]),
                "aurc_margin": M.aurc(c.margin[m], c.correct[m]),
                "risk_at_coverage_0.5_iso": M.risk_at_coverage(c.p[m], c.correct[m], 0.5)[0],
                "risk_at_coverage_0.8_iso": M.risk_at_coverage(c.p[m], c.correct[m], 0.8)[0]}
        r["knn_vote_val"] = {s: {str(kk): {"post": float((c.knn[s]["vote"][kk] == d.y)[d.post].mean()), "all": float((c.knn[s]["vote"][kk] == d.y).mean())}
                                 for kk in (5, 10, 20)} for s in ("heldout", "standard")}
        ok = c.knn["heldout"]["vote"][10] == d.y
        r["knn_vote_val"]["heldout_10_post_ci"] = boot_ci(ok[d.post], d.runs[d.post])
        fit = d.in_a & d.post
        sg = signals(c, d, fit)
        ed = {}
        for name, sc in sg.items():
            m = d.in_b & d.post
            tg = M.TieGroups(sc[m], ~c.correct[m])
            boot = np.array([tg.auroc(counts[i][inv_post].astype(float)) for i in range(N_BOOT)])
            ma = d.in_b
            ed[name] = {"auroc_post": tg.auroc(), "auroc_post_ci": _ci(boot), "auprc_post": M.auprc(sc[m], ~c.correct[m]),
                        "auroc_all": M.auroc(sc[ma], ~c.correct[ma]), "auprc_all": M.auprc(sc[ma], ~c.correct[ma])}
        r["error_detection_cal_b"] = ed
        out[k] = r
        print(f"  per-k metrics k={k} done", flush=True)
    return out


def gates(d: Data, cells: dict[int, Cell]) -> dict:
    """O6: thresholds on cal-B post-fault windows; one global set vs per-k sets; out-of-selection check on cal-A."""
    ks = sorted(cells)
    B, A = d.in_b & d.post, d.in_a & d.post
    variants = {"p_only": lambda c: None, "p+margin": lambda c: c.margin,
                "p+agreement_heldout": lambda c: c.knn["heldout"]["agree"], "p+agreement_standard": lambda c: c.knn["standard"]["agree"]}
    res: dict = {}
    for vname, getf in variants.items():
        res[vname] = {}
        for target in TARGETS:
            P = np.concatenate([cells[k].p[B] for k in ks])
            F = None if vname == "p_only" else np.concatenate([getf(cells[k])[B] for k in ks])
            C = np.concatenate([cells[k].correct[B] for k in ks])
            one = M.best_gate(P, F, C, target)
            entry: dict = {"single": one, "per_k": {}}
            if one["feasible"]:
                entry["single"]["by_k"] = {}
                for k in ks:
                    c = cells[k]
                    f = None if F is None else getf(c)
                    for name, m in (("cal_b", B), ("cal_a", A)):
                        acc = M.apply_gate(c.p[m], None if f is None else f[m], one["tau_p"], one["tau_2"])
                        entry["single"]["by_k"].setdefault(str(k), {})[name] = M.gate_stats(acc, c.correct[m])
                for name, m in (("cal_b", B), ("cal_a", A)):
                    acc = np.concatenate([M.apply_gate(cells[k].p[m], None if getf(cells[k]) is None else getf(cells[k])[m], one["tau_p"], one["tau_2"]) for k in ks])
                    cor = np.concatenate([cells[k].correct[m] for k in ks])
                    entry["single"][f"pooled_{name}"] = M.gate_stats(acc, cor)
            covs = []
            for k in ks:
                c = cells[k]
                f = None if vname == "p_only" else getf(c)
                g = M.best_gate(c.p[B], None if f is None else f[B], c.correct[B], target)
                if g["feasible"]:
                    for name, m in (("cal_b", B), ("cal_a", A)):
                        acc = M.apply_gate(c.p[m], None if f is None else f[m], g["tau_p"], g["tau_2"])
                        g[name] = M.gate_stats(acc, c.correct[m])
                    covs.append(g["coverage"])
                else:
                    covs.append(0.0)
                entry["per_k"][str(k)] = g
            entry["per_k_mean_coverage_cal_b"] = float(np.mean(covs))
            res[vname][str(target)] = entry
    return res


def posthoc(d: Data, cells: dict[int, Cell], counts: np.ndarray) -> dict:
    """POST HOC (not pre-registered; written after the decision was seen): does kNN agreement add beyond probability + margin?

    Per k: logistic fits on cal-A post-fault of [p, margin] and [p, margin, agreement] (held-out store); paired run-bootstrap on cal-B.
    Also the 3-feature gate (p, margin, agreement) at the O6 targets, per k, selected on cal-B, checked on cal-A.
    """
    use, fit = d.in_b & d.post, d.in_a & d.post
    _, inv = np.unique(d.runs[use], return_inverse=True)
    g2, g3 = {}, {}
    for k, c in cells.items():
        a = c.knn["heldout"]["agree"]
        X2, X3 = np.c_[c.p, c.margin], np.c_[c.p, c.margin, a]
        for X, g in ((X2, g2), (X3, g3)):
            pc = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit], c.correct[fit]).predict_proba(X)[:, 1]
            g[k] = M.TieGroups(-pc[use], ~c.correct[use])
    ks = sorted(cells)
    point = {k: g3[k].auroc() - g2[k].auroc() for k in ks}
    boot = np.empty((N_BOOT, len(ks) + 1))
    for r in range(N_BOOT):
        w = counts[r][inv].astype(float)
        dd = [g3[k].auroc(w) - g2[k].auroc(w) for k in ks]
        boot[r, :-1], boot[r, -1] = dd, np.mean(dd)
    out = {"note": "post hoc, not pre-registered", "per_k": {k: {"auroc_p_margin": g2[k].auroc(), "auroc_p_margin_agreement": g3[k].auroc(), "diff": point[k],
                                                                "diff_ci": _ci(boot[:, i])} for i, k in enumerate(ks)},
           "mean_over_k": {"diff": float(np.mean(list(point.values()))), "ci": _ci(boot[:, -1])}, "gate3": {}}
    B, A = d.in_b & d.post, d.in_a & d.post
    for target in TARGETS:
        rows = {}
        for k in ks:
            c = cells[k]
            g = M.best_gate3(c.p[B], c.margin[B], c.knn["heldout"]["agree"][B], c.correct[B], target)
            if g["feasible"]:
                acc = lambda m: (c.p[m] >= g["tau_p"]) & (c.margin[m] >= g["tau_2"]) & (c.knn["heldout"]["agree"][m] >= g["tau_3"])
                g["cal_b"], g["cal_a"] = M.gate_stats(acc(B), c.correct[B]), M.gate_stats(acc(A), c.correct[A])
            rows[str(k)] = g
        out["gate3"][str(target)] = {"per_k": rows, "mean_coverage_cal_b": float(np.mean([r["coverage"] if r["feasible"] else 0 for r in rows.values()]))}
    return out


def main() -> None:
    t0 = time.time()
    d = Data()
    print(f"val windows {len(d.sel)}, cal-A runs/windows, cal-B runs {d.n_boot_runs}; isotonic map equals S2: {d.iso_matches_s2}", flush=True)
    if not d.iso_matches_s2:
        sys.exit("isotonic map differs from the S2 map: stop and report")
    rng = np.random.default_rng(SEED_BOOT)
    counts = _counts(d, rng)
    cells = {k: Cell(d, k, 0) for k in KS}
    res: dict = {"setup": {"n_val_windows_unmasked": int(len(d.sel)), "n_post_fault": int(d.post.sum()), "iso_matches_s2": d.iso_matches_s2,
                           "n_boot": N_BOOT, "boot_seed": SEED_BOOT, "knn_k": KNN, "ks": list(KS), "offset": 0}}
    res["decision"] = {"primary": decision(d, cells, "heldout", counts)}
    print(f"primary decision statistic: {res['decision']['primary']['primary_mean_over_k']}", flush=True)
    res["decision"]["secondary"] = {"standard_store": decision(d, cells, "standard", counts), "all_windows_heldout": decision(d, cells, "heldout", counts, post_only=False)}
    for o in (() if SMOKE else (1, 2)):
        res["decision"]["secondary"][f"offset{o}_heldout"] = decision(d, {k: Cell(d, k, o if k else 0) for k in KS}, "heldout", counts)
    res["per_k"] = per_k_metrics(d, cells, counts)
    res["gates"] = gates(d, cells)
    res["posthoc"] = posthoc(d, cells, counts)
    OUT.mkdir(parents=True, exist_ok=True)
    path = (TRACK / "eval" / "layer0_smoke.json") if SMOKE else (OUT / "layer0_val_seed42.json")
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"written in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
