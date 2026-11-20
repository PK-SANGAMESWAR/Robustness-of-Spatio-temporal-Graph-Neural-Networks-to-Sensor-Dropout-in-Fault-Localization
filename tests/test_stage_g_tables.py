"""Stage G tables (``fault_localization/stage_g/tables.py``): statistics helpers, file readers, AUDC, T13 bootstrap.

Synthetic data for the logic; the real-file tests read the existing results and skip if they are absent.
No sealed-set file is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from fault_localization.stage_g import tables as T

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


# ------------------------------------------------------------------------- helpers


def test_parse_cell_formats():
    assert T.parse_cell("k10_sigma0.002") == (10, 0.002)
    assert T.parse_cell("k0_sigma0") == (0, 0.0)
    assert T.parse_cell("k0_sigma0.0") == (0, 0.0)
    assert T.parse_cell("k5") == (5, 0.0)


def test_mean_std_uses_sample_std():
    m, s = T.mean_std([0.277607, 0.343488, 0.280076])
    assert m == pytest.approx(0.3003903, abs=1e-6)
    assert s == pytest.approx(0.0373443, abs=1e-6)          # Paper-1 degradation_curves.csv, gru k=0 (ddof = 1)
    assert T.mean_std([0.5]) == (0.5, None)


def test_audc_matches_paper1_eq8():
    ks = [0, 1, 2, 5, 10]
    f1 = [0.8, 0.8, 0.8, 0.8, 0.8]
    assert T.audc(ks, f1, relative=True) == pytest.approx(1.0)
    assert T.audc(ks, f1, relative=False) == pytest.approx(0.8)
    f1 = [1.0, 0.9, 0.8, 0.5, 0.0]
    hand = ((1 + 0.9) / 2 + (0.9 + 0.8) / 2 + 3 * (0.8 + 0.5) / 2 + 5 * (0.5 + 0) / 2) / 10
    assert T.audc(ks, f1, relative=False) == pytest.approx(hand)
    assert T.audc(ks[::-1], f1[::-1], relative=False) == pytest.approx(hand)       # order does not matter


def test_audc_relative_needs_k0():
    with pytest.raises(ValueError):
        T.audc([1, 2], [0.5, 0.4], relative=True)


def test_paired_matches_scipy_and_counts_signs():
    a, b = [0.9, 0.85, 0.88], [0.8, 0.84, 0.70]
    pr = T.paired(a, b)
    assert pr["p"] == pytest.approx(stats.ttest_rel(a, b).pvalue)
    assert pr["n_positive"] == 3 and pr["n"] == 3
    assert pr["delta"] == pytest.approx(np.mean(np.subtract(a, b)))
    assert T.paired([0.5], [0.4])["p"] is None


def test_holm_known_values():
    adj = T.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert T.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_fmt():
    assert T.fmt(0.12345) == "0.123"
    assert T.fmt(0.12345, 0.01) == "0.123 ± 0.010"
    assert T.fmt(None) == "—"


# ------------------------------------------------------------------------- readers


def _cells_json(path: Path, key: str, vals: dict):
    cells = {f"k{k}_sigma{s}": {key: v, "f1_by_resistance": {"10.0": v - 0.1}} for (k, s), v in vals.items()}
    path.write_text(json.dumps({"cells": cells}))


def test_load_cells_reads_both_scorer_formats(tmp_path):
    _cells_json(tmp_path / "a.json", "f1_fused_mean", {(0, 0.0): 0.8, (10, 0.01): 0.6})
    _cells_json(tmp_path / "b.json", "f1_mean", {(0, 0): 0.7})
    a, b = T.load_cells(tmp_path / "a.json"), T.load_cells(tmp_path / "b.json")
    assert a[(0, 0.0)]["f1"] == 0.8 and a[(10, 0.01)]["by_res"][10.0] == pytest.approx(0.5)
    assert b[(0, 0.0)]["f1"] == 0.7 and b[(0, 0.0)]["psm"] is None


def test_curve_row_full_and_partial():
    per_seed = {s: {0: 0.9, 1: 0.88, 2: 0.86, 5: 0.8, 10: 0.7 + 0.01 * i} for i, s in enumerate(SEEDS)}
    row = T.curve_row("x", per_seed, "src", "proto")
    assert row["n_seeds"] == 3 and row["f1_k0"] == pytest.approx(0.9) and row["f1_k0_std"] == pytest.approx(0.0)
    assert row["ret10"] == pytest.approx(np.mean([0.7 / 0.9, 0.71 / 0.9, 0.72 / 0.9]))
    part = T.curve_row("y", {s: {0: 0.9, 10: 0.7} for s in SEEDS}, "src", "proto")
    assert part["AUDC rel"] == "—" and part["k=1"] == "—" and part["f1_k10"] == pytest.approx(0.7)


# ------------------------------------------------------------------------- T13 bootstrap on synthetic predictions

CELLS = [(0, 0.0), (10, 0.0), (0, 0.0005), (0, 0.002), (0, 0.01), (10, 0.0005), (10, 0.002), (10, 0.01)]
N_EV = 48


def _write(out: Path, name: str, seed: int, acc: float, rng_seed: int):
    rng = np.random.RandomState(1)
    ev = np.repeat(np.arange(N_EV), 40)
    cls = rng.randint(1, 26, N_EV)
    w = np.tile(np.arange(40), N_EV)
    y = np.where(w >= 20, cls[ev], 0).astype(np.int8)
    res = np.array([0.1, 1.0, 10.0])[np.arange(N_EV) % 3][ev].astype(np.float32)
    rng = np.random.RandomState(rng_seed)
    arrays = {"window_ids": np.array([f"e{e:03d}_w{i % 40:02d}" for i, e in enumerate(ev)]),
              "event_id": np.array([f"e{e:03d}" for e in ev]), "y_true": y, "resistance": res,
              "fault_type": np.array(["AG"] * len(y)), "cells": np.array(json.dumps({"cells": CELLS, "n_offsets": 3}))}
    for k, s in CELLS:
        for o in range(3):
            wrong = rng.randint(0, 26, len(y)).astype(np.int8)
            arrays[f"pf__k{k}_s{s:g}_o{o}"] = np.where(rng.rand(len(y)) < acc, y, wrong).astype(np.int8)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"seed_{seed:05d}_predictions.npz", **arrays)


@pytest.fixture()
def synth_test_dir(tmp_path):
    for i, s in enumerate(SEEDS):
        _write(tmp_path, "r5", s, 0.95, 10 + s)
        _write(tmp_path, "r0n", s, 0.60, 20 + s)
        _write(tmp_path, "r6i", s, 0.10, 30 + s)
        _write(tmp_path, "r2", s, 0.95, 40 + s)            # R2 = R5 accuracy: A2 must not be "supported"
    return tmp_path


def test_t13_supported_only_with_sign_and_ci(synth_test_dir):
    rows, boot = T.build_t13(n_boot=200, test_dir=synth_test_dir)
    by = {(r["pair"], r["cell"]): r for r in rows}
    h2 = by[("R5 vs R0ⁿ", "k=10, σ=0, pooled")]
    assert h2["supported (3/3 and CI excl. 0)"] == "yes" and h2["seeds > 0"] == "3/3"
    assert by[("R5 vs R6ⁱ", "k=10, σ=0, pooled")]["supported (3/3 and CI excl. 0)"] == "yes"
    a2 = [r for r in rows if r["pair"] == "R5 vs R2"]
    assert a2 and not all(r["supported (3/3 and CI excl. 0)"] == "yes" for r in a2)
    h6 = by[("R5 vs R0ⁿ", "k=0, σ=0, pooled")]
    assert h6["family"] == "none (control)" and h6["supported (3/3 and CI excl. 0)"].startswith("rule is |Δ|")
    h1 = by[("R5 vs R0ⁿ", "k=0, σ=0, R=10")]
    assert h1["n_events"] == N_EV // 3
    # Holm within {H1-H4} only, monotone and >= raw p
    fam = [r for r in rows if r["family"] == "{H1–H4}"]
    assert len(fam) == 8 and all(r["p_holm"] >= r["p"] - 1e-12 for r in fam)
    assert not any("p_holm" in r for r in rows if r["family"] == "none (control)")
    # the same resamples are reused: a second call with boot passed in gives identical rows
    rows2, _ = T.build_t13(n_boot=200, test_dir=synth_test_dir, boot=boot)
    assert [r["95 % CI (event bootstrap)"] for r in rows] == [r["95 % CI (event bootstrap)"] for r in rows2]


def test_bootstrap_pair_refuses_misaligned_windows(synth_test_dir):
    p = synth_test_dir / "r0n" / "seed_00043_predictions.npz"
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["window_ids"] = arrays["window_ids"][::-1]
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="windows differ"):
        T.bootstrap_pair("r0n", [(0, 0.0)], n_boot=10, test_dir=synth_test_dir)


def test_write_table(tmp_path):
    rows = [{"a": 1, "b": "x", "extra": [1, 2]}, {"a": 2, "b": "y"}]
    T.write_table(rows, ["a", "b"], tmp_path / "t", "Title", ["note one"])
    md = (tmp_path / "t.md").read_text()
    assert md.startswith("# Title") and "| 1 | x |" in md and "- note one" in md
    assert (tmp_path / "t.csv").read_text().splitlines()[0] == "a,b,extra"


# ------------------------------------------------------------------------- real files (skip if absent)

needs_results = pytest.mark.skipif(not (REPO / "results/stage_g/test/r5/seed_00042_scoring.json").exists(),
                                   reason="Stage G test-split results not present")


@needs_results
def test_real_r0n_metric_consistency():
    """12_CHECKLISTS § 0: R0n training-loop val F1 equals the scorer's k0 sigma0 value."""
    for r in T.r0n_metric_consistency():
        assert r["abs_diff"] < 1e-9


@needs_results
def test_real_t5_rows():
    rows = {r["model"]: r for r in T.build_t5()}
    r0 = rows["RGSAGE-Max (dropout-aware) = R0"]
    assert r0["f1_k0"] == pytest.approx(0.8387763741604594, abs=1e-6)     # results/RESULTS.md § 3: 0.839
    assert r0["audc_rel"] == pytest.approx(0.913, abs=0.001)               # results/RESULTS.md § 3: AUDC 0.913
    assert rows["R5 (PC-STGNN)"]["n_seeds"] == 3 and rows["B2ᵃ (regressed R̂; epoch-1 checkpoint)"]["n_seeds"] == 1
    assert rows["R6ⁱ (shuffled table, inference)"]["AUDC rel"] == "—"      # only k = 0 and 10 scored


@needs_results
def test_real_t10_matches_summary():
    summary = json.loads((REPO / "results/stage_g/analysis/summary.json").read_text())
    rows = T.build_t10()
    r = next(x for x in rows if x["condition"] == "noise σ=0" and x["k"] == 10)
    assert r["delta"] == pytest.approx(summary["s3_test_r5_vs_r0n"]["k10_sigma0.0"]["delta"], abs=1e-12)
    t = next(x for x in rows if x["condition"] == "policy targeted" and x["k"] == 10)
    assert t["delta"] == pytest.approx(summary["s4_policies_test_r5_vs_r0n"]["targeted"]["k10"]["delta"], abs=1e-12)


@needs_results
def test_real_t6_values_and_blocks():
    rows = T.build_t6()
    by = {(r["block"], r["model"]): r for r in rows}
    assert {r["block"] for r in rows} == {"validation", "test"}
    assert by[("validation", "Rank-1 sag-template matched filter")]["k=0"] == "0.162"      # CLAUDE.md / 10_FIGURES: 0.162
    assert by[("test", "Logistic regression (summary features)")]["k=0"] == "0.713"         # results/RESULTS.md § 8
    assert by[("test", "Random forest (summary features)")]["k=0"] == "0.633"
    assert by[("test", "RGSAGE-Max, dropout-aware (Paper 1)")]["k=0"].startswith("0.839")
    assert all(r["protocol"] and r["source"] and r["split"] in ("val", "test") for r in rows)   # 12_CHECKLISTS § 0: scope on every row


@pytest.mark.skipif(not (REPO / "results/stage_f/gtf2/core/summary.json").exists(), reason="Stage F results absent")
def test_real_t9_matches_stage_f_record():
    det = {r["condition"]: r for r in T.build_t9_detection()}
    assert det["bias_2pct_s1"]["AUROC LOO σ0=0"] == "0.9991"                 # 07_STAGE_F: S5.1 gate table
    assert det["bias_2pct_s1"]["AUROC LOO σ0=0.01"] == "0.8537"
    gate = [r for r in T.build_t9_gate() if r["arm"] == "gtf2"]
    stuck = [r for r in gate if r["condition"].startswith("stuck")]
    assert len(stuck) == 3 and all(r["seeds > 0"] == "3/3" for r in stuck)    # 9 of 9 seed-by-s pairs positive
    clean = next(r for r in gate if r["condition"].startswith("clean"))
    assert clean["gate − no gate"] == "-0.0009"                               # mean of -0.0006/-0.0010/-0.0011


def test_T9c_gate_test_matches_summary_json():
    """T9c (test split) rows equal results/stage_f/gtf2_test/summary.json; the decision-rule roles are carried over."""
    import json
    p = T.REPO / "results/stage_f/gtf2_test/summary.json"
    if not p.exists():
        import pytest
        pytest.skip("pending: results/stage_f/gtf2_test/summary.json")
    d = json.loads(p.read_text())
    rows = {r["condition"]: r for r in T.build_t9_gate_test()}
    c = d["sigma"]["0.01"]["conditions"]["stuck_s2"]
    r = rows["stuck_s2"]
    assert r["role"] == "primary"
    assert abs(float(r["gate − no gate"]) - np.mean(np.array(c["gtf2"]) - np.array(c["nogate"]))) < 5e-5
    assert r["seeds > 0"] == "3/3"
    assert rows["gauss_0.02_s1"]["role"] == "reported only"
    assert all(rows[f"stuck_s{s}"]["seeds > 0"] == "3/3" for s in (1, 2, 3))
