"""The residual is the one piece of PC-STGNN that cannot be debugged by looking
at a training curve: it has no parameters, so a wrong rho just produces a model
that quietly learns to ignore the physics term.  These tests pin the equations
of ``PC-STGNN/01_SPEC.md`` §4 and §7 against a reference implementation and
against planted ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.physics import grids, residual as R


@pytest.fixture(scope="module")
def smooth_table():
    """A table whose hypotheses vary *smoothly*, as the real (tau, R, L) grid does.

    ``tiny_table`` draws every hypothesis independently, which makes them all
    mutually far apart -- no single sensor can then move the arg-min, and the
    leave-one-out refit trivially returns the same fit as the in-sample one.
    The real operator is not like that: neighbouring grid points differ by
    ~1e-3 p.u., so a corrupted sensor genuinely can drag the profiled
    ``(tau*, R*, L*)`` toward itself.  This fixture reproduces that structure --
    a well-separated base signature per location plus one smooth nuisance
    direction along the hypothesis axis -- so the LOO test exercises the
    mechanism it is meant to guard.
    """
    rng = np.random.default_rng(424242)
    n_l = 3
    n_h = grids.num_hypotheses(n_l)
    base = rng.normal(1.0, 0.05, (26, 75)).astype(np.float32)
    direction = rng.normal(0.0, 1.0, (26, 75)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    t = np.linspace(-0.5, 0.5, n_h, dtype=np.float32)
    table = base[:, None, :] + t[None, :, None] * direction[:, None, :]
    return torch.from_numpy(table)


@pytest.fixture(scope="module")
def tiny_table():
    """A small but correctly shaped table: 3 load points, so H = 11*15*3 = 495."""
    rng = np.random.default_rng(20260923)
    n_l = 3
    mu = rng.normal(1.0, 0.05, (25, 11, 15, n_l, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (n_l, 25, 3)).astype(np.float32)
    return mu, mu0, R.flatten_table(mu, mu0), n_l


def test_flatten_table_keeps_the_no_fault_row(tiny_table):
    mu, mu0, table, n_l = tiny_table
    assert table.shape == (26, grids.num_hypotheses(n_l), 75)
    # Row 0 is mu0 broadcast over (tau, R): every block of n_l entries repeats it.
    for l_idx in range(n_l):
        assert torch.allclose(table[0, l_idx], torch.from_numpy(mu0[l_idx].reshape(-1)))
        assert torch.allclose(table[0, 7 * 15 * n_l + 2 * n_l + l_idx],
                              torch.from_numpy(mu0[l_idx].reshape(-1)))
    # Row f+1 is location f, in C order over (tau, R, L).
    assert torch.allclose(table[4, 3 * 15 * n_l + 6 * n_l + 1],
                          torch.from_numpy(mu[3, 3, 6, 1].reshape(-1)))


def test_unravel_inverts_the_flattening(tiny_table):
    _, _, _, n_l = tiny_table
    for tau, r, l in [(0, 0, 0), (10, 14, 2), (5, 7, 1)]:
        flat = tau * 15 * n_l + r * n_l + l
        assert grids.unravel_hypothesis(flat, n_l) == (tau, r, l)


def test_rho_is_non_negative_with_exactly_one_zero(tiny_table):
    _, _, table, _ = tiny_table
    rng = np.random.default_rng(1)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (6, 25, 3)).astype(np.float32))
    a = torch.ones(6, 25, 3)
    rho, _ = R.rho(s, a, table, 1e-3, chunk_f=4)
    assert (rho >= 0).all()
    assert (rho == 0).sum(dim=1).tolist() == [1] * 6


def test_argmin_recovers_a_planted_hypothesis(tiny_table):
    """An observation taken *from* the table must be matched back to it exactly."""
    _, _, table, n_l = tiny_table
    f, h = 7, 123
    s = table[f, h].reshape(1, 25, 3).clone()
    a = torch.ones(1, 25, 3)
    rho, hyp = R.rho(s, a, table, 1e-3, chunk_f=26)
    assert int(R.psm_predict(rho)) == f
    assert int(hyp[0, f]) == h
    assert float(rho[0, f]) == 0.0
    # And the margin to the runner-up is large, i.e. the match is decisive.
    assert float(R.margin(rho)) > 1.0


def test_matches_the_reference_loop(tiny_table):
    """Gate 3: the batched layer must be the spec's equations, not an approximation."""
    _, _, table, _ = tiny_table
    rng = np.random.default_rng(2)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (3, 25, 3)).astype(np.float32))
    a = torch.ones(3, 25, 3)
    a[1, 4] = 0.0
    a[2, :, 2] = 0.0
    fast, _ = R.rho(s, a, table, 1e-3, chunk_f=3)
    slow, _ = R.reference_loop(s, a, table, 1e-3)
    # rho reaches ~1e5 here, so the comparison is relative: float32 accumulation
    # over 75 channels is the only difference between the two.
    np.testing.assert_allclose(fast.numpy(), slow, rtol=1e-4, atol=1e-3)


def test_gram_method_agrees_in_float64(tiny_table):
    _, _, table, _ = tiny_table
    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (3, 25, 3))).double()
    a = torch.ones(3, 25, 3).double()
    direct, _ = R.rho(s, a, table, 1e-3, chunk_f=26, method="direct")
    gram, _ = R.rho(s, a, table, 1e-3, method="gram")
    np.testing.assert_allclose(gram.numpy(), direct.numpy(), rtol=1e-6, atol=1e-6)


def test_gram_method_refuses_float32(tiny_table):
    """float32 loses the true hypothesis to cancellation -- fail loudly, not quietly."""
    _, _, table, _ = tiny_table
    s = torch.ones(1, 25, 3)
    with pytest.raises(TypeError):
        R.rho(s, torch.ones(1, 25, 3), table, 1e-3, method="gram")


def test_dropped_channels_contribute_nothing(tiny_table):
    """Zeroing a node's availability must be identical to it never existing."""
    _, _, table, _ = tiny_table
    rng = np.random.default_rng(4)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (2, 25, 3)).astype(np.float32))
    a = torch.ones(2, 25, 3)
    a[:, 9] = 0.0
    rho_ref, _ = R.rho(s, a, table, 1e-3, chunk_f=26)
    # Put garbage on the dropped node: rho must not move at all.
    s_garbage = s.clone()
    s_garbage[:, 9] = 42.0
    rho_garbage, _ = R.rho(s_garbage, a, table, 1e-3, chunk_f=26)
    assert torch.equal(rho_ref, rho_garbage)


def test_p_phys_is_a_distribution(tiny_table):
    _, _, table, _ = tiny_table
    s = table[3, 11].reshape(1, 25, 3).clone()
    p = R.p_phys(R.rho(s, torch.ones(1, 25, 3), table, 1e-3, chunk_f=26)[0])
    assert pytest.approx(1.0, abs=1e-6) == float(p.sum())
    assert int(p.argmax()) == 3


def test_loo_beats_in_sample_on_a_planted_corrupt_sensor(smooth_table):
    """The S5.2 premise, and the reason for the ordering in ``01_SPEC.md`` §7.

    Read sensor i's error at the arg-min fitted on *all* of A and a badly
    corrupted sensor drags that fit toward itself, hiding its own error.
    Refitting on A \\ {i} first is what exposes it.  If this ever fails, S5.2 is
    comparing a variant against itself.
    """
    table = smooth_table
    f, h, bad = 12, 250, 6
    s = table[f, h].reshape(1, 25, 3).clone()
    s[0, bad] += 0.25                       # a large, silent offset on one sensor
    a = torch.ones(1, 25, 3)
    f_hat = torch.tensor([f])
    loo = R.rho_loo(s, a, table, f_hat, 1e-3)
    insample = R.rho_insample(s, a, table, f_hat, 1e-3)
    assert float(loo[0, bad]) > float(insample[0, bad])
    # And the corrupted sensor is the worst one under LOO.
    assert int(loo[0].argmax()) == bad


def test_in_sample_can_hide_a_corruption_entirely():
    """The worst case the LOO ordering exists to defeat.

    Build a table whose only nuisance freedom lives on sensor ``bad``: one
    hypothesis differs from another *only* in that sensor's reading.  A silent
    failure that happens to look like that nuisance direction is then explained
    away completely by the in-sample fit -- its score is exactly 0, a perfectly
    healthy-looking sensor -- while the leave-one-out refit, which never saw
    sensor ``bad``, reports the full error.

    This is not a contrived corner: profiling (tau, R, L) is exactly a search
    over signatures that move the sensors, so a drifted sensor competing with a
    slightly different load level is the normal case.
    """
    bad = 6
    base = np.full((26, 2, 75), 1.0, dtype=np.float32)
    # Hypothesis 1 differs from hypothesis 0 on sensor `bad`'s phase-A channel only.
    base[:, 1, bad * 3] = 1.25
    # Keep the locations apart so the class arg-min is unambiguous.
    base[:, :, 0] += np.arange(26, dtype=np.float32)[:, None]
    table = torch.from_numpy(base)

    f = 12
    s = table[f, 0].reshape(1, 25, 3).clone()
    s[0, bad, 0] = 1.25                      # the sensor now reads hypothesis 1's value
    a = torch.ones(1, 25, 3)

    insample = R.rho_insample(s, a, table, torch.tensor([f]), 1e-3)
    loo = R.rho_loo(s, a, table, torch.tensor([f]), 1e-3)
    assert float(insample[0, bad]) == pytest.approx(0.0, abs=1e-6)
    assert float(loo[0, bad]) > 1.0
    # Every other sensor is clean under both scores.
    others = [i for i in range(25) if i != bad]
    assert float(loo[0, others].max()) == pytest.approx(0.0, abs=1e-6)


def test_loo_is_zero_on_unavailable_sensors(tiny_table):
    _, _, table, _ = tiny_table
    s = table[2, 40].reshape(1, 25, 3).clone()
    a = torch.ones(1, 25, 3)
    a[0, 17] = 0.0
    loo = R.rho_loo(s, a, table, torch.tensor([2]), 1e-3)
    assert float(loo[0, 17]) == 0.0


def test_heldout_residual_uses_the_hypothesis_profiled_on_a(tiny_table):
    """lambda_2: profile on A, score on D.  D must never influence the profiling."""
    _, _, table, _ = tiny_table
    f, h = 5, 77
    s_clean = table[f, h].reshape(1, 25, 3).clone()
    a = torch.ones(1, 25, 3)
    d = torch.zeros(1, 25, 3)
    a[0, 3] = 0.0
    d[0, 3] = 1.0                            # node 3 is the held-out sensor
    rho_a, hyp = R.rho(s_clean, a, table, 1e-3, chunk_f=26)
    rho_d = R.rho_heldout(s_clean, d, table, hyp, 1e-3)
    # The true hypothesis explains the held-out sensor perfectly too.
    assert float(rho_d[0, f]) == pytest.approx(0.0, abs=1e-5)
    assert int(rho_d.argmin()) == f
