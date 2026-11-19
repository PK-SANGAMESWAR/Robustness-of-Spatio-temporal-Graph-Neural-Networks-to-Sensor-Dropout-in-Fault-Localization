"""T8 / F-G3 helpers on a synthetic run directory. No sealed file is read; the real run is not needed."""
from __future__ import annotations

import json

import pytest

from fault_localization.stage_g import gate6 as G
from fault_localization.stage_g import sealed_tables as S
from tests.test_gate6 import SEEDS, _make_dir


@pytest.fixture()
def run_dir(tmp_path):
    _make_dir(tmp_path, acc5=[0.9, 0.9, 0.9], acc0n=[0.6, 0.6, 0.6])
    (tmp_path / "STARTED.json").write_text(json.dumps({"smoke": False}))
    return tmp_path


def _analysis(run_dir, reportable=True):
    out = G.run_analysis(run_dir, n_boot=60)
    out["reportable"] = reportable
    (run_dir / "gate6_analysis.json").write_text(json.dumps(out))
    return out


def test_refuses_unfinished_smoke_and_unmarked_runs(run_dir):
    with pytest.raises(FileNotFoundError, match="COMPLETED"):
        S.check_reportable(run_dir)
    (run_dir / "COMPLETED.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="gate6_analysis"):
        S.check_reportable(run_dir)
    _analysis(run_dir, reportable=False)
    with pytest.raises(ValueError, match="not marked reportable"):
        S.check_reportable(run_dir)
    (run_dir / "STARTED.json").write_text(json.dumps({"smoke": True}))
    with pytest.raises(ValueError, match="smoke"):
        S.check_reportable(run_dir)


def test_t8_rows_and_roles(run_dir):
    (run_dir / "COMPLETED.json").write_text("{}")
    _analysis(run_dir)
    rows = S.build_t8(S.check_reportable(run_dir))
    primary = [r for r in rows if r["role"].startswith("primary")]
    assert {r["cell"] for r in primary} == {"k=0, σ=0.0005", "k=0, σ=0.002", "k=0, σ=0.01"} and all(r["subset"] == "pooled" for r in primary)
    p = primary[0]
    assert p["seeds > 0"] == "3/3" and p["delta"] > 0.2 and p["ci95"][0] > 0
    assert {"R=3", "R=30", "grounded", "ungrounded"} <= {r["subset"] for r in rows}
    assert all(r["R0"] != "—" for r in rows)                      # r0 files exist in the fixture


def test_gate6_lines_copy_the_verdict(run_dir):
    a = _analysis(run_dir)                                  # 60 resamples: Gate 6 must say "not evaluable", never "passed"
    lines = S.gate6_lines(a)
    assert a["gate6"]["evaluable"] is False and a["gate6"]["passed"] is False
    assert "evaluable: False; passed: False" in lines[0]
    assert sum("→ pass" in l for l in lines) == sum(v["passes"] for v in a["gate6"]["per_sigma"].values()) == 3
    a["gate6"]["per_sigma"]["0.01"]["passes"] = False      # the lines copy the file, they do not recompute
    assert sum("→ pass" in l for l in S.gate6_lines(a)) == 2


def test_resistance_series_reads_scoring_files(tmp_path):
    for s in SEEDS:
        for name, v in (("r5", 0.9), ("r0n", 0.5)):
            cells = {"k0_sigma0": {"f1_mean": v, "f1_by_resistance": {"3.0": v, "30.0": v - 0.1}}}
            (tmp_path / f"{name}_seed{s:05d}_scoring.json").write_text(json.dumps({"cells": cells}))
    ser = S.f1_by_resistance_series(tmp_path)
    assert set(ser) == {"r5", "r0n"} and ser["r5"]["ood"][30.0] == pytest.approx([0.8] * 3) and ser["r5"]["in"] == {}
