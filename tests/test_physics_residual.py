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
    """A small but correctly shaped table: 3 load points, so H = 11*NUM_R*3.

    Uses ``grids.NUM_R`` (18 since the 2026-09-25 R-grid fix, was 15) rather than a
    literal, so this fixture tracks whatever the real operator's R axis is.
    """
    rng = np.random.default_rng(20260923)
    n_l = 3
    mu = rng.normal(1.0, 0.05, (25, 11, grids.NUM_R, n_l, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (n_l, 25, 3)).astype(np.float32)
    return mu, mu0, R.flatten_table(mu, mu0), n_l


def test_flatten_table_keeps_the_no_fault_row(tiny_table):
    mu, mu0, table, n_l = tiny_table
    n_r = grids.NUM_R
    assert table.shape == (26, grids.num_hypotheses(n_l), 75)
    # Row 0 is mu0 broadcast over (tau, R): every block of n_l entries repeats it.
    for l_idx in range(n_l):
        assert torch.allclose(table[0, l_idx], torch.from_numpy(mu0[l_idx].reshape(-1)))
        assert torch.allclose(table[0, 7 * n_r * n_l + 2 * n_l + l_idx],
                              torch.from_numpy(mu0[l_idx].reshape(-1)))
    # Row f+1 is location f, in C order over (tau, R, L).
    assert torch.allclose(table[4, 3 * n_r * n_l + 6 * n_l + 1],
                          torch.from_numpy(mu[3, 3, 6, 1].reshape(-1)))


def test_unravel_inverts_the_flattening(tiny_table):
    _, _, _, n_l = tiny_table
    n_r = grids.NUM_R
    for tau, r, l in [(0, 0, 0), (10, n_r - 1, 2), (5, 7, 1)]:
        flat = tau * n_r * n_l + r * n_l + l
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


# ---------------------------------------------------------------------------
# Adversarial degenerate inputs (12_CHECKLISTS.md sampling rule, S3.4)
#
# The spec's original per-channel average r~ = r/|A| divided by the number of
# surviving channels, which is zero when every sensor is dropped -- and r~_f(D)
# divided by |D|, which is zero on every clean batch, i.e. the lambda_2 arm would
# have hit it constantly.  The shifted/halved rho of 01_SPEC.md §4 removed both
# divisions.  These tests pin that: they are the regression guard for a NaN that
# the rewrite made structurally impossible, and they fail loudly if any future
# change reintroduces a normalisation by |A| or |D|.
# ---------------------------------------------------------------------------


def test_empty_D_gives_zero_not_nan(smooth_table):
    """|D| = 0 is the *normal* case at k = 0: every clean batch hits it."""
    table = smooth_table
    rng = np.random.default_rng(7)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (4, 25, 3)).astype(np.float32))
    a = torch.ones(4, 25, 3)
    _, hyp = R.rho(s, a, table, 1e-3, chunk_f=26)
    d_empty = torch.zeros(4, 25, 3)
    rho_d = R.rho_heldout(s, d_empty, table, hyp, 1e-3)
    assert torch.isfinite(rho_d).all()
    assert torch.equal(rho_d, torch.zeros_like(rho_d))


def test_all_channels_dropped_gives_zero_not_nan(smooth_table):
    """|A| = 0: no evidence, so rho is flat and P_phys uniform -- never NaN."""
    table = smooth_table
    s = torch.zeros(3, 25, 3)
    a = torch.zeros(3, 25, 3)
    rho, hyp = R.rho(s, a, table, 1e-3, chunk_f=26)
    assert torch.isfinite(rho).all()
    assert torch.equal(rho, torch.zeros_like(rho))
    p = R.p_phys(rho)
    assert torch.allclose(p, torch.full_like(p, 1.0 / 26), atol=1e-6)
    assert torch.isfinite(R.rho_loo(s, a, table, torch.zeros(3, dtype=torch.long), 1e-3)).all()


def test_single_surviving_channel_is_finite(smooth_table):
    """|A| = 1 -- the thinnest non-empty evidence the residual can be asked for."""
    table = smooth_table
    rng = np.random.default_rng(11)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (2, 25, 3)).astype(np.float32))
    a = torch.zeros(2, 25, 3)
    a[:, 4, 1] = 1.0
    rho, hyp = R.rho(s, a, table, 1e-3, chunk_f=26)
    assert torch.isfinite(rho).all()
    assert (rho >= 0).all()
    loo = R.rho_loo(s, a, table, R.psm_predict(rho), 1e-3)
    assert torch.isfinite(loo).all()
    # Only the one instrumented node can carry a trust score; the rest are gated off.
    assert float(loo[:, [i for i in range(25) if i != 4]].abs().max()) == 0.0


def test_exact_ties_are_resolved_without_nan(smooth_table):
    """Two locations fitting equally well: rho has two zeros, and that is legal."""
    table = smooth_table.clone()
    table[5] = table[9]                       # locations 5 and 9 now identical
    s = table[5, 7].reshape(1, 25, 3).clone()
    a = torch.ones(1, 25, 3)
    rho, _ = R.rho(s, a, table, 1e-3, chunk_f=26)
    assert torch.isfinite(rho).all()
    assert int((rho[0] == 0).sum()) == 2
    assert float(R.margin(rho)) == 0.0        # zero margin: indistinguishable
    assert int(R.psm_predict(rho)) in (5, 9)


def test_pc_loss_lambda2_is_zero_when_D_is_empty(smooth_table):
    """The end-to-end guard: the lambda_2 arm on a clean batch contributes 0."""
    from fault_localization.models.common import pc_loss

    table = smooth_table
    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.normal(1.0, 0.05, (4, 25, 3)).astype(np.float32))
    a = torch.ones(4, 25, 3)
    rho_a, hyp = R.rho(s, a, table, 1e-3, chunk_f=26)
    rho_d = R.rho_heldout(s, torch.zeros(4, 25, 3), table, hyp, 1e-3)
    out = {
        "logits": torch.randn(4, 25, 26, requires_grad=True),
        "p_graph": torch.full((4, 26), 1.0 / 26),
        "rho_A": rho_a,
        "rho_D": rho_d,
    }
    y = torch.randint(0, 26, (4,))
    both = pc_loss(out, y, lam1=1.0, lam2=1.0)
    only1 = pc_loss(out, y, lam1=1.0, lam2=0.0)
    assert torch.isfinite(both["total"])
    assert float(both["l2"]) == 0.0
    assert torch.allclose(both["total"], only1["total"])


def test_effective_r_levels_collapses_duplicate_resistances():
    """A table whose R axis repeats itself is effectively lower-dimensional."""
    from fault_localization.physics import identifiability as I

    rng = np.random.default_rng(5)
    mu = rng.normal(1.0, 0.05, (3, 2, 15, 3, 25, 3)).astype(np.float32)
    # Collapse the R axis onto 5 distinct values, each repeated 3 times.
    for r in range(15):
        mu[:, :, r] = mu[:, :, r % 5]
    out = I.effective_r_levels(mu, sigma_eps=1e-4)
    assert out["mean_levels"] == 5.0
    assert out["effective_fraction"] == 5 / 15
    # A coarse tolerance merges everything; a fine one keeps all 15 distinct.
    assert I.effective_r_levels(mu, sigma_eps=10.0)["mean_levels"] == 1.0
    mu2 = rng.normal(1.0, 0.05, (3, 2, 15, 3, 25, 3)).astype(np.float32)
    assert I.effective_r_levels(mu2, sigma_eps=1e-6)["mean_levels"] == 15.0
