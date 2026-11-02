"""S6.2 mismatch tables: the perturbation code (CPU, OpenDSS compile only; no table is built here)."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

dss = pytest.importorskip("opendssdirect")

from fault_localization.physics import mismatch as M  # noqa: E402
from fault_localization.simulation.feeder import compile_feeder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _volts(master: Path) -> np.ndarray:
    compile_feeder(master)
    dss.Solution.Solve()
    return np.array(dss.Circuit.AllBusMagPu())


def _feeder_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(M.FEEDER_DIR.glob("*")):
        if p.is_file():
            h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def test_impedance_factors_are_fixed_by_seed_and_exactly_plus_minus_level():
    names = [f"L{i}" for i in range(50)]
    a = M.impedance_factors(names, 0.05)
    assert a == M.impedance_factors(names, 0.05)                          # same seed, same draw
    assert set(np.round(list(a.values()), 6)) == {0.95, 1.05}
    assert a != M.impedance_factors(names, 0.05, seed=1)


def test_matrix_text_is_lower_triangular():
    assert M._matrix_text([1, 2, 3, 4], 2) == "(1 | 3 4)"
    assert M._matrix_text([5.0], 1) == "(5)"


def test_zero_level_impedance_edit_reproduces_the_nominal_feeder(tmp_path):
    nominal = _volts(M.FEEDER_DIR / M.MASTER_NAME)
    master = M.copy_feeder(tmp_path / "f")
    edits, info = M.impedance_edit_lines(master, 0.0)
    master.write_text(master.read_text() + "\n" + "\n".join(edits) + "\n")
    assert info["n_lines"] == 118                                         # 123 lines minus 5 switches
    assert np.abs(_volts(master) - nominal).max() < 1e-12


def test_the_three_perturbations_move_voltages_and_leave_the_nominal_files_alone(tmp_path):
    before = _feeder_hash()
    nominal = _volts(M.FEEDER_DIR / M.MASTER_NAME)
    shifts = {}
    for name in M.MISMATCH_LEVELS:
        spec = M.make_perturbed_feeder(name, tmp_path)
        shifts[name] = np.abs(_volts(Path(spec["master"])) - nominal)
        text = Path(spec["master"]).read_text()
        assert "S6.2 mismatch perturbation: " + name in text
    assert _feeder_hash() == before                                       # nominal feeder files untouched
    # one tap step is about 6.4e-3 p.u. (08_STAGE_G_eval.md S6.2); median shift of buses
    assert 4e-3 < np.median(shifts["tap_p1"]) < 1e-2 and 4e-3 < np.median(shifts["tap_m1"]) < 1e-2
    assert 0 < shifts["imp5"].max() < 5e-3
    assert M.make_perturbed_feeder("tap_p1", tmp_path)["n_edit_lines"] == 7           # 7 regulator controls


def test_tap_edit_moves_every_vreg_by_exactly_one_step(tmp_path):
    master = M.copy_feeder(tmp_path / "f")
    up = M.tap_edit_lines(master, +1.0)
    down = M.tap_edit_lines(master, -1.0)
    assert len(up) == len(down) == 7
    vr = {l.split()[1].split(".")[1]: float(l.split("vreg=")[1]) for l in up}
    vd = {l.split()[1].split(".")[1]: float(l.split("vreg=")[1]) for l in down}
    for k in vr:
        assert vr[k] - vd[k] == pytest.approx(2 * M.TAP_STEP_VOLTS, abs=1e-5)


def test_table_shift_summary(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    base = np.zeros((2, 3, 4), dtype=np.float32)
    np.save(a / "mu.npy", base)
    np.save(b / "mu.npy", base + np.float32(0.01))
    s = M.table_shift(a, b)
    assert s["median"] == pytest.approx(0.01, abs=1e-6) and s["n"] == 24
    np.save(b / "mu.npy", base[:1])
    with pytest.raises(ValueError):
        M.table_shift(a, b)


def test_precheck_windows_are_fixed_fault_windows_after_inception():
    x1, y1, stats, w1 = M.load_precheck_windows(REPO / "dataset_fullscale_50", "val", 16, seed=7)
    x2, y2, _s, w2 = M.load_precheck_windows(REPO / "dataset_fullscale_50", "val", 16, seed=7)
    assert w1 == w2 and np.array_equal(x1, x2)
    assert x1.shape == (16, 25, 20, 6) and (y1 != 0).all() and 1 <= y1.min() and y1.max() <= 25
    assert all(int(w.rsplit("_w", 1)[1]) >= 21 for w in w1)


def test_nominal_table_beats_a_scaled_table_on_the_true_class(tmp_path):
    """The pre-check separates a right table from a wrong one (a table scaled by 2 % is clearly wrong)."""
    x, y, stats, _ = M.load_precheck_windows(REPO / "dataset_fullscale_50", "val", 24, seed=7)
    bad = tmp_path / "bad"
    shutil.copytree(REPO / "operator", bad)
    mu = np.load(bad / "mu.npy")
    np.save(bad / "mu.npy", mu * np.float32(1.02))
    good = M.true_class_residual(REPO / "operator", x, y, stats, 7.255549105131452e-4, sigma0s=(0.0,))["sigma0=0"]
    worse = M.true_class_residual(bad, x, y, stats, 7.255549105131452e-4, sigma0s=(0.0,))["sigma0=0"]
    assert worse["raw_half_median"] > 5 * good["raw_half_median"]
