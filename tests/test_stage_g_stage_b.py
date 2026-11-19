"""Stage B tables built by scripts/build_stage_g_stage_b.py: numbers equal the audit JSON they come from."""
import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
T = REPO / "results/stage_g/tables"


def _rows(name):
    return list(csv.reader(open(T / name)))


def test_T3_solve_count_matches_grid_json():
    g = json.loads((REPO / "operator/grid.json").read_text())
    rows = {r[0]: r[1] for r in _rows("T3_signature_table_config.csv")[1:]}
    assert f"{g['n_solves']:,}" in rows["Solves (OpenDSS)"]
    assert rows["Axes (μ[f,τ,R,L,i,p])"] == "25 × 11 × 18 × 17 × 25 × 3"


def test_T4_fidelity_matches_summary():
    s = json.loads((REPO / "results/pcstgnn/audit/s1_4_7/s1_4_7_summary.json").read_text())
    rows = {r[0]: r for r in _rows("T4_fidelity_summary.csv")[1:]}
    assert rows["all"][2] == f"{s['S1.4_bias']['median_abs_e_max_channel']:.2e}"
    assert rows["L* within 1 step"][2] == f"{s['S1.5_load']['l_star_within_1_step_frac']:.4f}"


def test_T2_ood_event_counts():
    text = (T / "T2_dataset_ood_config.md").read_text()
    assert "11,000 events" in text and "1,100 events" in text and "825 scenarios" in text


@pytest.mark.parametrize("fig", ["F-B3_fidelity_by_R_type", "F-B4_profile_error_hist", "F-B6_rgrid_density"])
def test_stage_b_figures_exist(fig):
    assert (REPO / "results/stage_g/figures" / f"{fig}.png").stat().st_size > 5000


def test_FB1_network_check_passes_and_ratio_matrix_is_complete():
    r = json.loads((REPO / "results/pcstgnn/audit/s1_1_network/s1_1_network.json").read_text())
    assert r["pass"] and r["y_v_vs_ycurrents_rel_err"] < 1e-3 and r["n_nan_pairs"] == 0
    import numpy as np
    S = np.load(REPO / "results/pcstgnn/audit/s1_1_network/transfer_sensitivity.npy")
    assert S.shape == (25, 25) and np.isfinite(S).all()


def test_FB5_floor_by_R_matches_recorded_median():
    d = json.loads((REPO / "results/pcstgnn/audit/s1_6_floor_by_R/floor_by_R.json").read_text())
    assert d["n_events"] == 28875 and abs(d["median_all"] - 1.0) < 1e-3          # 03_STAGE_B S1.6 record: median 1.0000
    assert sum(v["n"] for v in d["by_R"].values()) == d["n_events"]


def test_release_check_B1_and_B1a_reproduce_the_stored_values():
    r = json.loads((REPO / "results/stage_g/release_check/b1_b1a_check.json").read_text())
    assert all(v["abs_diff"] == 0.0 for v in r["B1"]["f1_by_k"].values()) and r["B1"]["v_nominal_max_abs_diff"] == 0.0
    assert r["B1a"]["curve_max_abs_diff"] == 0.0 and r["B1a"]["val_f1_k0"]["new"] == r["B1a"]["val_f1_k0"]["stored"]
