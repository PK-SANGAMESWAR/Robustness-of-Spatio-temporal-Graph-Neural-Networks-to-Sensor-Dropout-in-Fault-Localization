"""The signature solver is the only bridge between the corpus and the operator.

If it converges but describes a slightly different circuit than the one that
generated the corpus, every downstream number inherits that error silently --
``03_STAGE_B_operator.md`` S1.4 would read it as bias and try to absorb it into
sigma_eps.  These tests check the cheap, structural half of that (shapes,
convergence, the instrumentation pattern, the frozen load level); the
quantitative half is ``scripts/audit_snapshot_equivalence.py`` (S0.4), which
needs the corpus and so is not a unit test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fault_localization.config.constants import FAULT_BUSES, FAULT_TYPES, PRE_FAULT_MS
from fault_localization.physics import grids
from fault_localization.physics.signature_solver import (
    instrumented_mask,
    reset_engine,
    solve_nofault,
    solve_signature,
)

CORPUS = Path("dataset_fullscale_50")


def test_prefault_freeze_is_the_step_39_value():
    """The corpus's fault solves run at the step-39 load, not the step-40 one.

    ``monitors.solve_prefault_timeseries`` rescales the loads at the top of each
    pre-fault step and the fault loop of ``simulation/runner.py`` never rescales
    again, so the sinusoid is frozen one step short of its period.  Getting this
    wrong shifts every signature by ~5e-4 p.u. -- the same order as sigma_eps.
    """
    expected = 1.0 + 0.003 * np.sin(2.0 * np.pi * 39 / 40)
    assert grids.PREFAULT_FREEZE == pytest.approx(expected, rel=1e-12)
    assert grids.PREFAULT_FREEZE == pytest.approx(0.99953, abs=1e-5)
    assert grids.PREFAULT_FREEZE != 1.0
    assert PRE_FAULT_MS == 40


def test_grids_are_the_specified_axes():
    # [Corrected 2026-09-25] R_GRID is the original 15-pt log grid unioned with the
    # corpus's own {0.1, 1, 10} Ohm -- 18 points, no longer uniformly log-spaced.
    # See grids.py's R_GRID docstring: the pure log grid failed S1.5 (L* recovery
    # 38.4%, need >90%) via R-L compensation from off-grid R_true.
    assert grids.R_GRID.shape == (18,)
    assert grids.R_GRID[0] == pytest.approx(0.05)
    assert grids.R_GRID[-1] == pytest.approx(30.0)
    for corpus_r in (0.1, 1.0, 10.0):
        assert np.any(np.isclose(grids.R_GRID, corpus_r)), f"{corpus_r} must be on-grid"
    assert np.all(np.diff(grids.R_GRID) > 0)                   # strictly increasing
    assert len(grids.l_grid(0.05)) == 17
    assert len(grids.l_grid(0.025)) == 33
    np.testing.assert_allclose(grids.l_grid(0.05)[[0, -1]], [0.5, 1.3])
    with pytest.raises(ValueError):
        grids.l_grid(0.03)      # does not divide [0.5, 1.3] exactly


def test_single_solve_converges_and_has_the_right_shape():
    reset_engine()
    volts, mask, converged = solve_signature(FAULT_BUSES[0], FAULT_TYPES[0], 1.0, 1.0)
    assert converged
    assert volts.shape == (25, 3) and mask.shape == (25, 3)
    assert np.isfinite(volts).all()
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # Per-unit magnitudes, not volts.
    assert 0.0 <= volts.max() < 2.0


def test_nofault_solve_is_a_healthy_feeder():
    reset_engine()
    volts, mask, converged = solve_nofault(1.0)
    assert converged
    instrumented = mask > 0.5
    assert 0.9 < volts[instrumented].min() and volts[instrumented].max() < 1.1


def test_a_fault_actually_sags_the_feeder():
    """Guards the failure mode where the Fault element is created but not enabled."""
    reset_engine()
    healthy, mask, _ = solve_nofault(1.0)
    faulted, _, _ = solve_signature(FAULT_BUSES[1], "ABCG", 0.1, 1.0)
    instrumented = mask > 0.5
    assert np.abs(faulted - healthy)[instrumented].max() > 1e-3


def test_higher_resistance_gives_a_shallower_sag():
    """A bolted fault must sag the feeder harder than a 30 ohm one."""
    reset_engine()
    healthy, mask, _ = solve_nofault(1.0)
    instrumented = mask > 0.5
    bolted, _, _ = solve_signature(FAULT_BUSES[4], "ABCG", 0.05, 1.0)
    weak, _, _ = solve_signature(FAULT_BUSES[4], "ABCG", 30.0, 1.0)
    assert (healthy - bolted)[instrumented].max() > (healthy - weak)[instrumented].max()


def test_reuse_mode_matches_a_fresh_compile():
    """The 70k-solve build reuses the circuit; S0.4 trusts it only if this holds."""
    reset_engine()
    fresh, _, _ = solve_signature(FAULT_BUSES[7], "BCG", 1.0, 0.75, mode="daily")
    reset_engine()
    # Warm the engine with unrelated solves, then re-solve the same hypothesis.
    solve_signature(FAULT_BUSES[2], "AG", 10.0, 1.2, mode="reuse", reuse_limit=100)
    solve_nofault(0.6, mode="reuse", reuse_limit=100)
    reused, _, _ = solve_signature(FAULT_BUSES[7], "BCG", 1.0, 0.75,
                                   mode="reuse", reuse_limit=100)
    np.testing.assert_allclose(reused, fresh, atol=1e-6)


def test_the_freeze_factor_actually_changes_the_answer():
    """If applying the freeze made no difference, S0.4's assertion would be vacuous."""
    reset_engine()
    frozen, mask, _ = solve_nofault(1.0, apply_freeze=True)
    unfrozen, _, _ = solve_nofault(1.0, apply_freeze=False)
    assert np.abs(frozen - unfrozen)[mask > 0.5].max() > 0.0


@pytest.mark.skipif(not CORPUS.exists(), reason="frozen corpus not present")
def test_instrumentation_pattern_equals_the_corpus():
    """The solver's mask must be the corpus's mask, channel for channel.

    Only 14 of the 25 uPMUs are three-phase.  If the solver disagreed with the
    corpus about which phases exist, the residual would sum over channels that
    were never measured.
    """
    reset_engine()
    mask = instrumented_mask()
    event = sorted((CORPUS / "raw_sim").glob("*.npz"))[0]
    with np.load(event, allow_pickle=True) as loaded:
        record = loaded["data"]
    corpus_mask = record[:, 0, 3:]
    np.testing.assert_array_equal(mask.astype(float), corpus_mask)
    assert int((mask.sum(axis=1) == 3).sum()) == 14
    # The mask is static: identical in every timestep of the record.
    assert np.array_equal(record[:, :, 3:], np.broadcast_to(
        corpus_mask[:, None, :], record[:, :, 3:].shape))
