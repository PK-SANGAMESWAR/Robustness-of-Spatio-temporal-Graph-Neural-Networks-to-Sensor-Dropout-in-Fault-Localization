"""Gate 6 analysis: the fast macro-F1, the bootstrap, the sign rule and the pass logic. Synthetic data, then the pilot smoke files.

No number from any pilot file is checked or reported; the pilot test only checks the structure.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from fault_localization.stage_g import gate6 as G

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
CELLS = [(0, 0.0), (0, 0.0005), (0, 0.002), (0, 0.01), (5, 0.0), (10, 0.0)]
N_EV = 60
TYPES = np.array(["AG", "AB", "ABCG", "BC", "CAG", "ABC"])
RS = np.array([3.0, 5.0, 20.0, 30.0])


def _geometry():
    rng = np.random.RandomState(1)
    ev = np.repeat(np.arange(N_EV), 40)
    fault_class = rng.randint(1, 26, N_EV)
    w = np.tile(np.arange(40), N_EV)
    y = np.where(w >= 20, fault_class[ev], 0).astype(np.int8)
    res = RS[np.arange(N_EV) % 4][ev].astype(np.float32)
    ftype = TYPES[np.arange(N_EV) % 6][ev]
    return ev, y, res, ftype


def _write_run(out: Path, name: str, seed: int, acc: float, rng_seed: int, geometry=None):
    ev, y, res, ftype = geometry or _geometry()
    rng = np.random.RandomState(rng_seed)
    arrays = {"window_ids": np.array([f"ev{e:04d}_w{i % 40:04d}" for i, e in enumerate(ev)]),
              "event_id": np.array([f"ev{e:04d}" for e in ev]), "y_true": y, "resistance": res, "fault_type": ftype,
              "cells": np.array(json.dumps({"cells": CELLS, "n_offsets": 3}))}
    for k, s in CELLS:
        for o in range(3):
            wrong = rng.randint(0, 26, len(y)).astype(np.int8)
            arrays[f"pf__k{k}_s{s:g}_o{o}"] = np.where(rng.rand(len(y)) < acc, y, wrong).astype(np.int8)
    np.savez_compressed(out / f"{name}_seed{seed:05d}_predictions.npz", **arrays)


def _make_dir(tmp_path, acc5, acc0n, acc0=0.5):
    for seed in SEEDS:
        i = SEEDS.index(seed)
        _write_run(tmp_path, "r5", seed, acc5[i], 10 + seed)
        _write_run(tmp_path, "r0n", seed, acc0n[i], 20 + seed)
        _write_run(tmp_path, "r0", seed, acc0, 30 + seed)
    return tmp_path


# ------------------------------------------------------------------------- the fast F1
def test_macro_f1_from_confusion_equals_sklearn_including_empty_classes():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 20, 5000)                     # classes 20..25 never occur in the truth
    p = np.where(rng.rand(5000) < 0.6, y, rng.randint(0, 24, 5000))
    c = G.event_confusions(y, p, np.zeros(5000, dtype=int), 1)
    assert G.macro_f1_from_confusion(c.sum(axis=0)) == pytest.approx(f1_score(y, p, average="macro", labels=list(range(26)), zero_division=0))
    # batch dimension
    assert G.macro_f1_from_confusion(np.stack([c.sum(0), c.sum(0)])).shape == (2,)


def test_event_confusions_count_every_window_once():
    y = np.array([0, 1, 1, 2, 2, 2])
    p = np.array([0, 1, 2, 2, 2, 0])
    ev = np.array([0, 0, 1, 1, 2, 2])
    c = G.event_confusions(y, p, ev, 3)
    assert c.shape == (3, 676) and c.sum() == 6 and c[0].reshape(26, 26)[1, 1] == 1


# ------------------------------------------------------------------------- the rule
def test_clear_gain_in_all_seeds_passes_gate6(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75, 0.76, 0.74], acc0n=[0.60, 0.61, 0.62])
    out = G.run_analysis(d, n_boot=10000)
    g = out["gate6"]
    assert g["evaluable"] and g["passed"]
    assert set(g["per_sigma"]) == {"0.0005", "0.002", "0.01"}
    assert all(v["sign_all_seeds_positive"] and v["ci95"][0] > 0 for v in g["per_sigma"].values())
    assert out["n_events"] == N_EV and out["n_windows"] == 40 * N_EV and "r0" in out["models"]


def test_one_negative_seed_fails_the_sign_rule_even_if_the_mean_gain_has_a_positive_interval(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.85, 0.52, 0.86], acc0n=[0.60, 0.60, 0.60])
    out = G.run_analysis(d, n_boot=10000)
    for v in out["gate6"]["per_sigma"].values():
        assert v["delta_per_seed"][1] < 0 and not v["sign_all_seeds_positive"] and not v["passes"]
        assert v["ci95"][0] > 0                                # the interval alone would have passed
    assert not out["gate6"]["passed"]


def test_no_difference_fails_because_the_interval_includes_zero(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.6, 0.6, 0.6], acc0n=[0.6, 0.6, 0.6])
    out = G.run_analysis(d, n_boot=10000)
    assert not out["gate6"]["passed"]
    assert any(not v["ci_lower_above_zero"] for v in out["gate6"]["per_sigma"].values())


def test_rule_needs_all_three_sigma(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75, 0.76, 0.74], acc0n=[0.60, 0.61, 0.62])
    out = G.run_analysis(d, n_boot=10000)
    assert out["gate6"]["passed"]
    out["gate6"]["per_sigma"]["0.01"]["passes"] = False        # same data, one sigma failing
    assert not all(v["passes"] for v in out["gate6"]["per_sigma"].values())


def test_not_evaluable_with_fewer_resamples_or_seeds(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75] * 3, acc0n=[0.6] * 3)
    assert not G.run_analysis(d, n_boot=300)["gate6"]["evaluable"]
    assert not G.run_analysis(d, seeds=(42,), n_boot=10000)["gate6"]["evaluable"]


# ------------------------------------------------------------------------- consistency with direct computation
def test_point_estimates_equal_sklearn_and_bootstrap_is_reproducible(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75, 0.76, 0.74], acc0n=[0.60, 0.61, 0.62])
    a = G.run_analysis(d, n_boot=400)
    b = G.run_analysis(d, n_boot=400)
    assert a["cells"]["k0_sigma0.01"]["pooled"]["ci95"] == b["cells"]["k0_sigma0.01"]["pooled"]["ci95"]
    ev, y, res, ftype = _geometry()
    r5 = G.Run(d / "r5_seed00042_predictions.npz")
    r0n = G.Run(d / "r0n_seed00042_predictions.npz")
    f5 = np.mean([f1_score(y, r5.pred(0, 0.01, o), average="macro", labels=list(range(26)), zero_division=0) for o in range(3)])
    f0 = np.mean([f1_score(y, r0n.pred(0, 0.01, o), average="macro", labels=list(range(26)), zero_division=0) for o in range(3)])
    pooled = a["cells"]["k0_sigma0.01"]["pooled"]
    assert pooled["f1"]["r5"][0] == pytest.approx(f5) and pooled["f1"]["r0n"][0] == pytest.approx(f0)
    assert pooled["delta_per_seed"][0] == pytest.approx(f5 - f0)
    # per resistance: the stratum is the events with that R
    m = np.isclose(res, 20.0)
    f5_r = np.mean([f1_score(y[m], r5.pred(0, 0.01, o)[m], average="macro", labels=list(range(26)), zero_division=0) for o in range(3)])
    assert a["cells"]["k0_sigma0.01"]["R=20"]["f1"]["r5"][0] == pytest.approx(f5_r)


def test_strata_resistances_and_grounded_groups(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75] * 3, acc0n=[0.6] * 3)
    out = G.run_analysis(d, n_boot=200)
    cell = out["cells"]["k0_sigma0.002"]
    assert {"pooled", "R=3", "R=5", "R=20", "R=30", "grounded", "ungrounded"} <= set(cell)
    assert cell["R=3"]["n_events"] + cell["R=5"]["n_events"] + cell["R=20"]["n_events"] + cell["R=30"]["n_events"] == N_EV
    assert cell["grounded"]["n_events"] + cell["ungrounded"]["n_events"] == N_EV
    # types AG ABCG CAG are grounded (3 of 6 types), AB BC ABC ungrounded
    assert cell["grounded"]["n_events"] == N_EV // 2
    assert set(out["cells"]) == {"k0_sigma0.0005", "k0_sigma0.002", "k0_sigma0.01", "k0_sigma0", "k5_sigma0", "k10_sigma0"}


def test_different_windows_in_two_runs_are_refused(tmp_path):
    d = _make_dir(tmp_path, acc5=[0.75] * 3, acc0n=[0.6] * 3)
    ev, y, res, ftype = _geometry()
    _write_run(d, "r0n", 43, 0.6, 99, geometry=(ev[::-1].copy(), y, res, ftype))
    with pytest.raises(ValueError, match="windows differ"):
        G.run_analysis(d, n_boot=50)


# ------------------------------------------------------------------------- pilot smoke files: structure only
SMOKE = REPO / "results/stage_g/ood_pilot_smoke"


@pytest.mark.skipif(not (SMOKE / "COMPLETED.json").exists(), reason="pilot smoke run not present")
def test_pilot_smoke_predictions_give_a_structure_and_the_cli_prints_no_number(tmp_path):
    d = tmp_path / "smoke"
    shutil.copytree(SMOKE, d, ignore=shutil.ignore_patterns("eval_root"))
    out = G.run_analysis(d, seeds=(42,), n_boot=50)
    assert not out["gate6"]["evaluable"] and "pooled" in out["cells"]["k0_sigma0.01"]
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_gate6_analysis.py"), "--dir", str(d), "--seeds", "42",
                        "--n-boot", "50"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "structure only" in r.stdout
    assert "delta" not in r.stdout and "ci95" not in r.stdout
    written = json.loads((d / "gate6_analysis_SMOKE_NOT_REPORTABLE.json").read_text())
    assert written["reportable"] is False and not (d / "gate6_analysis.json").exists()


def test_cli_refuses_a_directory_without_completed_json(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_gate6_analysis.py"), "--dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2 and "COMPLETED" in r.stderr
