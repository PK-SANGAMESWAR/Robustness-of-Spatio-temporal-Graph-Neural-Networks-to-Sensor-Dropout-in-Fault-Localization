"""Silent failures are the one injector whose *definition* can invalidate the
experiment.

If the injector zeroes the availability mask it is ordinary dropout wearing a
different name, and every PCRG-STGNN number would be measuring something Paper 1
already measured.  If it writes to uninstrumented phases it manufactures
readings on channels that have no instrument.  Both are easy to do by accident
and neither shows up as a failing training run, so they are pinned here
(``05_STAGE_D_infra.md`` S3.1, ``12_CHECKLISTS.md`` §C).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fault_localization.dropout.silent import (
    SILENT_KINDS,
    apply_silent_failure_batch,
)

MEAN = np.array([0.8285, 0.6046, 0.7576], dtype=np.float32)
STD = np.array([0.3752, 0.5012, 0.4369], dtype=np.float32)
STATS = (MEAN, STD)


@pytest.fixture
def batch():
    """A batch with a realistic instrumentation pattern: some nodes are 1-phase."""
    rng = np.random.default_rng(7)
    x = np.zeros((4, 25, 20, 6), dtype=np.float32)
    mask = np.zeros((25, 3), dtype=np.float32)
    mask[:14] = 1.0                       # 14 three-phase uPMUs, as on the feeder
    mask[14:20, 0] = 1.0                  # six single-phase
    mask[20:, [0, 2]] = 1.0               # five two-phase
    volts = rng.normal(1.0, 0.02, (4, 25, 20, 3)).astype(np.float32)
    volts = (volts - MEAN) / STD
    broadcast = mask[None, :, None, :]
    x[..., :3] = volts * broadcast        # uninstrumented phases carry exactly 0
    x[..., 3:] = np.broadcast_to(broadcast, x[..., 3:].shape)
    return torch.from_numpy(x)


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_no_corruption_is_byte_identical(batch, kind):
    out, nodes = apply_silent_failure_batch(batch, 0, kind, seed=1, norm_stats=STATS)
    assert torch.equal(out, batch)
    assert nodes == [[] for _ in range(batch.shape[0])]


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_mask_channels_are_never_written(batch, kind):
    """The whole point: m stays 1, so no mask-based model can see the failure."""
    out, nodes = apply_silent_failure_batch(batch, 3, kind, seed=2, norm_stats=STATS)
    assert torch.equal(out[..., 3:], batch[..., 3:])
    for sample_nodes in nodes:
        for node in sample_nodes:
            assert (out[:, node, :, 3:] > 0.5).any()


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_uninstrumented_phases_are_untouched(batch, kind):
    out, _ = apply_silent_failure_batch(batch, 3, kind, seed=3, norm_stats=STATS)
    absent = batch[..., 3:] <= 0.5
    assert torch.equal(out[..., :3][absent], batch[..., :3][absent])
    assert torch.all(out[..., :3][absent] == 0.0)


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_deterministic_under_a_fixed_seed(batch, kind):
    """Models must see identical corruption, or the comparison is not paired."""
    a_out, a_nodes = apply_silent_failure_batch(batch, 2, kind, seed=99, norm_stats=STATS)
    b_out, b_nodes = apply_silent_failure_batch(batch, 2, kind, seed=99, norm_stats=STATS)
    assert torch.equal(a_out, b_out)
    assert a_nodes == b_nodes


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_different_seeds_select_different_sensors(batch, kind):
    _, a_nodes = apply_silent_failure_batch(batch, 2, kind, seed=1, norm_stats=STATS)
    _, b_nodes = apply_silent_failure_batch(batch, 2, kind, seed=500, norm_stats=STATS)
    assert a_nodes != b_nodes


@pytest.mark.parametrize("kind", SILENT_KINDS)
def test_only_the_selected_sensors_change(batch, kind):
    out, nodes = apply_silent_failure_batch(batch, 2, kind, seed=4, norm_stats=STATS)
    for sample, sample_nodes in enumerate(nodes):
        assert len(sample_nodes) == 2
        clean = [i for i in range(25) if i not in sample_nodes]
        assert torch.equal(out[sample, clean], batch[sample, clean])
        for node in sample_nodes:
            assert not torch.equal(out[sample, node], batch[sample, node])


def test_stuck_freezes_at_the_first_timestep(batch):
    """Input-only by construction: no event metadata, no inception offset."""
    out, nodes = apply_silent_failure_batch(batch, 1, "stuck", seed=5, norm_stats=STATS)
    for sample, sample_nodes in enumerate(nodes):
        node = sample_nodes[0]
        frozen = out[sample, node, :, :3]
        assert torch.allclose(frozen, frozen[0:1].expand_as(frozen))
        assert torch.allclose(frozen[0], batch[sample, node, 0, :3])


def test_bias_is_a_relative_error_in_per_unit(batch):
    out, nodes = apply_silent_failure_batch(batch, 1, "bias", seed=6,
                                            norm_stats=STATS, severity=0.05)
    mean_t, std_t = torch.from_numpy(MEAN), torch.from_numpy(STD)
    for sample, sample_nodes in enumerate(nodes):
        node = sample_nodes[0]
        phases = batch[sample, node, 0, 3:] > 0.5
        before = batch[sample, node, :, :3] * std_t + mean_t
        after = out[sample, node, :, :3] * std_t + mean_t
        rel = ((after - before) / before)[:, phases]
        assert torch.allclose(rel.abs(), torch.full_like(rel, 0.05), atol=1e-5)


def test_gauss_severity_is_in_per_unit(batch):
    out, nodes = apply_silent_failure_batch(batch, 1, "gauss", seed=8,
                                            norm_stats=STATS, severity=0.02)
    mean_t, std_t = torch.from_numpy(MEAN), torch.from_numpy(STD)
    deltas = []
    for sample, sample_nodes in enumerate(nodes):
        node = sample_nodes[0]
        phases = batch[sample, node, 0, 3:] > 0.5
        before = batch[sample, node, :, :3] * std_t + mean_t
        after = out[sample, node, :, :3] * std_t + mean_t
        deltas.append((after - before)[:, phases].flatten())
    spread = torch.cat(deltas).std()
    assert 0.01 < float(spread) < 0.035      # ~0.02 p.u., sampling noise aside


def test_swap_permutes_instrumented_phases_only(batch):
    """A swapped sensor keeps its *set* of readings; only the assignment changes."""
    out, nodes = apply_silent_failure_batch(batch, 2, "swap", seed=9, norm_stats=STATS)
    for sample, sample_nodes in enumerate(nodes):
        for node in sample_nodes:
            phases = batch[sample, node, 0, 3:] > 0.5
            assert int(phases.sum()) >= 2     # 1-phase nodes cannot be swapped
            before = batch[sample, node, :, :3][:, phases]
            after = out[sample, node, :, :3][:, phases]
            assert torch.allclose(before.sort(dim=1).values, after.sort(dim=1).values)
            assert not torch.allclose(before, after)


def test_swap_skips_single_phase_sensors(batch):
    _, nodes = apply_silent_failure_batch(batch, 3, "swap", seed=10, norm_stats=STATS)
    single_phase = set(range(14, 20))
    for sample_nodes in nodes:
        assert not (set(sample_nodes) & single_phase)


def test_bias_and_gauss_require_norm_stats(batch):
    for kind in ("bias", "gauss"):
        with pytest.raises(ValueError):
            apply_silent_failure_batch(batch, 1, kind, seed=1, norm_stats=None)


def test_rejects_unknown_kind(batch):
    with pytest.raises(ValueError):
        apply_silent_failure_batch(batch, 1, "drift", seed=1, norm_stats=STATS)
