"""12_CHECKLISTS § C: the r-tilde cache key and the observation vector (last step, de-normalised, availability from mask)."""
import numpy as np

from fault_localization.physics import observation as O
from fault_localization.physics.cache import _dropout_indices, _window_dropout_seed


def test_cache_seed_is_keyed_by_window_k_and_mask_seed_and_is_stable():
    base = _window_dropout_seed("13_AG_1p0_0000_w0021", 5, 42)
    assert base == _window_dropout_seed("13_AG_1p0_0000_w0021", 5, 42)
    assert base != _window_dropout_seed("13_AG_1p0_0000_w0022", 5, 42)
    assert base != _window_dropout_seed("13_AG_1p0_0000_w0021", 2, 42)
    assert base != _window_dropout_seed("13_AG_1p0_0000_w0021", 5, 43)
    assert _dropout_indices("13_AG_1p0_0000_w0021", 5, 42) == _dropout_indices("13_AG_1p0_0000_w0021", 5, 42)
    assert len(set(_dropout_indices("13_AG_1p0_0000_w0021", 5, 42))) == 5
    assert _dropout_indices("13_AG_1p0_0000_w0021", 0, 42) == []


def test_observation_is_last_timestep_denormalised_with_mask_availability():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 25, 20, 6)).astype(np.float32)
    x[..., 3:] = 1.0
    x[0, 3, :, 3:] = 0.0                      # a node without instruments or dropped: availability 0
    mean = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    std = np.array([0.05, 0.06, 0.07], dtype=np.float32)
    s, a = O.observe(x, mean, std, reduce="last")
    np.testing.assert_allclose(s, x[:, :, -1, :3] * std + mean, rtol=1e-6)
    assert a[0, 3].sum() == 0 and a[1, 3].sum() == 3
