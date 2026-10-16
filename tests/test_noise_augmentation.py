"""Noise augmentation for PC-STGNN training (06_STAGE_E_pcstgnn.md S4.3b / S4.3c).

Three properties, each of which would silently bias the S4.3 rerun if broken:
1. A per-window sigma0 in the residual equals the scalar computation applied row by row.
2. Noise touches only instrumented voltage channels, never the mask, never sigma = 0 rows.
3. Adding noise does not change the k sequence or any dropout mask at the same seed, so a
   noise-augmented arm differs from its noise-free counterpart in exactly one variable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.dropout.training import PhysicsDropoutLoader
from fault_localization.physics import grids, residual as R

STD = np.array([0.3752, 0.5012, 0.4369], dtype=np.float32)
GRID = [0.0, 5e-4, 2e-3, 1e-2]


def _batches(n_batches=6, b=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_batches):
        x = torch.randn(b, 25, 20, 6, generator=g)
        mask = (torch.rand(b, 25, 1, 3, generator=g) > 0.3).float().expand(b, 25, 20, 3)
        x[..., 3:] = mask
        x[..., :3] = x[..., :3] * mask
        out.append({"x": x, "y": torch.zeros(b, dtype=torch.long),
                    "window_id": [f"e{i}_w{j:04d}" for j in range(b)]})
    return out


def test_channel_weights_per_window_sigma0_matches_scalar_rows():
    a = (torch.rand(5, 25, 3) > 0.2).float()
    s0 = torch.tensor([0.0, 5e-4, 2e-3, 1e-2, 0.0])
    batched = R.channel_weights(a, 7.2555e-4, s0)
    for i in range(5):
        row = R.channel_weights(a[i:i + 1], 7.2555e-4, float(s0[i]))
        torch.testing.assert_close(batched[i:i + 1], row)


def test_rho_per_window_sigma0_matches_scalar_rows():
    rng = np.random.default_rng(3)
    mu = rng.normal(1.0, 0.05, (25, 11, grids.NUM_R, 2, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (2, 25, 3)).astype(np.float32)
    table = torch.as_tensor(R.flatten_table(mu, mu0))
    s = torch.as_tensor(rng.normal(1.0, 0.05, (4, 25, 3)).astype(np.float32))
    a = (torch.rand(4, 25, 3) > 0.2).float()
    s0 = torch.tensor([0.0, 5e-4, 2e-3, 1e-2])
    rho_b, hyp_b = R.rho(s, a, table, 7.2555e-4, s0)
    for i in range(4):
        rho_i, hyp_i = R.rho(s[i:i + 1], a[i:i + 1], table, 7.2555e-4, float(s0[i]))
        torch.testing.assert_close(rho_b[i:i + 1], rho_i, rtol=1e-5, atol=1e-3)
        assert torch.equal(hyp_b[i:i + 1], hyp_i)


def test_noise_free_loader_is_unchanged():
    base = _batches()
    plain = list(PhysicsDropoutLoader(base, k_range=[0, 1, 2, 5], rng_seed=42))
    for out, src in zip(plain, base):
        assert "sigma0" not in out
        if out["dropout_k"] == 0:
            assert torch.equal(out["x"], src["x"])
        assert torch.equal(out["x_clean"], src["x"])


def test_noise_only_on_instrumented_voltage_channels_and_mask_untouched():
    base = _batches()
    noisy = list(PhysicsDropoutLoader(base, k_range=[0], rng_seed=42, noise_sigmas=GRID,
                                      norm_std=STD))
    saw_noise = False
    for out, src in zip(noisy, base):
        s0 = out["sigma0"]
        assert s0.shape == (src["x"].shape[0],)
        assert all(min(abs(v - g) for g in GRID) < 1e-8 for v in s0.tolist())
        assert torch.equal(out["x"][..., 3:], src["x"][..., 3:])       # mask never written
        delta = out["x"][..., :3] - src["x"][..., :3]
        assert torch.all(delta[src["x"][..., 3:] < 0.5] == 0)          # no instrument, no noise
        for i, sig in enumerate(s0.tolist()):
            if sig == 0:
                assert torch.equal(out["x"][i], src["x"][i])
            else:
                saw_noise = saw_noise or bool(delta[i].abs().max() > 0)
    assert saw_noise


def test_noise_does_not_change_k_sequence_or_dropout_masks():
    base = _batches(n_batches=10)
    plain = list(PhysicsDropoutLoader(base, k_range=[0, 1, 2, 5], rng_seed=7))
    noisy = list(PhysicsDropoutLoader(base, k_range=[0, 1, 2, 5], rng_seed=7,
                                      noise_sigmas=GRID, norm_std=STD))
    for p, n in zip(plain, noisy):
        assert p["dropout_k"] == n["dropout_k"]
        assert p["dropout_nodes"] == n["dropout_nodes"]
        assert torch.equal(p["drop_mask"], n["drop_mask"])


def test_noise_is_deterministic_under_a_fixed_seed():
    base = _batches()
    a = list(PhysicsDropoutLoader(base, [0, 1, 2, 5], rng_seed=11, noise_sigmas=GRID, norm_std=STD))
    b = list(PhysicsDropoutLoader(base, [0, 1, 2, 5], rng_seed=11, noise_sigmas=GRID, norm_std=STD))
    for x, y in zip(a, b):
        assert torch.equal(x["x"], y["x"]) and torch.equal(x["sigma0"], y["sigma0"])


def test_noise_requires_norm_std():
    with pytest.raises(ValueError):
        PhysicsDropoutLoader(_batches(), [0], noise_sigmas=GRID)
