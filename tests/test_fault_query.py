"""Invariants of FaultQuery-STGNN and its physics prior.

These run on CPU in seconds and gate every GPU run that follows: a silent
misalignment between the hop-distance matrix and the model's node axis, or an
attention mask that leaks weight onto a dropped sensor, would corrupt every
number downstream while still training to a plausible-looking accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fault_localization.config import constants as C
from fault_localization.graph.fault_distance import (CACHE_PATH,
                                                     load_fault_hop_distance)
from fault_localization.models.fault_query import FaultQuerySTGNN

DATA_ROOT = Path("dataset_fullscale_50")
BATCH, NODES, STEPS, FEATURES = 4, 25, 20, 6


@pytest.fixture(scope="module")
def hop_distance() -> np.ndarray:
    if not CACHE_PATH.exists():
        pytest.skip(f"hop-distance cache missing at {CACHE_PATH}")
    return load_fault_hop_distance()


@pytest.fixture(scope="module")
def edge_index() -> np.ndarray:
    path = DATA_ROOT / "processed" / "edge_index.npy"
    if not path.exists():
        pytest.skip(f"{path} missing")
    return np.load(path)


@pytest.fixture
def model(edge_index, hop_distance) -> FaultQuerySTGNN:
    torch.manual_seed(0)
    return FaultQuerySTGNN(edge_index, hop_distance, num_layers=4).eval()


def make_batch(dropped: list[int] | None = None) -> torch.Tensor:
    """A batch with all sensors reporting, optionally with some dropped.

    Dropping zeroes all six channels, matching
    ``fault_localization.dropout.masking.apply_sensor_dropout``.
    """
    torch.manual_seed(1)
    x = torch.rand(BATCH, NODES, STEPS, FEATURES)
    x[..., 3:] = 1.0
    if dropped:
        x[:, dropped] = 0.0
    return x


# ---------------------------------------------------------------------------
# Hop-distance matrix: alignment with the model's node axis
# ---------------------------------------------------------------------------

def test_distance_matrix_shape_and_range(hop_distance):
    assert hop_distance.shape == (26, 25)
    assert hop_distance.min() >= 0


def test_nofault_row_is_uniform(hop_distance):
    """Row 0 must be constant so it cancels in the softmax (no spurious prior)."""
    assert (hop_distance[0] == 0).all()


def test_measured_fault_buses_have_zero_distance(hop_distance):
    """Exactly the 6 fault buses that carry a μPMU may have a zero entry.

    This is the alignment check: a zero can only occur where a fault bus IS a
    sensor bus, so any other zero (or a missing one) means the matrix and the
    node axis disagree.
    """
    mapping = json.loads((DATA_ROOT / "processed" / "node_mapping.json").read_text())
    measured = sorted(set(C.PMU_BUSES) & set(C.FAULT_BUSES))
    assert len(measured) == 6, "dataset fact: 6 of 25 fault buses are instrumented"

    for label in range(1, 26):
        bus = C.LABEL_TO_FAULT_BUS[label]
        zeros = np.flatnonzero(hop_distance[label] == 0)
        if bus in measured:
            assert zeros.tolist() == [mapping[str(bus)]], (
                f"class {label} (bus {bus}) should be zero-distance at exactly "
                f"its own sensor node")
        else:
            assert zeros.size == 0, f"unmeasured bus {bus} must not be at distance 0"


# ---------------------------------------------------------------------------
# Output contract: [B, 1, 26] keeps soft_vote and expanded CE working unchanged
# ---------------------------------------------------------------------------

def test_output_shape_is_single_node(model):
    assert model(make_batch()).shape == (BATCH, 1, 26)


def test_soft_vote_is_identity_on_this_output(model):
    from fault_localization.models.voting import soft_vote
    logits = model(make_batch())
    _, votes = soft_vote(logits)
    assert torch.equal(votes, logits.squeeze(1).argmax(-1))


def test_expanded_cross_entropy_matches_plain(model):
    from fault_localization.models.common import expanded_cross_entropy
    logits = model(make_batch())
    y = torch.randint(0, 26, (BATCH,))
    plain = torch.nn.functional.cross_entropy(logits.squeeze(1), y)
    assert torch.allclose(expanded_cross_entropy(logits, y), plain)


# ---------------------------------------------------------------------------
# Attention behaviour
# ---------------------------------------------------------------------------

def test_attention_rows_are_distributions(model):
    attention = model.attention_weights(make_batch())
    assert attention.shape == (BATCH, 4, 26, NODES)
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)))
    assert (attention >= 0).all()


def test_dropped_sensors_receive_exactly_zero_attention(model):
    dropped = [0, 5, 11, 17]
    attention = model.attention_weights(make_batch(dropped))
    assert (attention[..., dropped] == 0).all(), \
        "a dropped sensor must contribute no evidence to any query"
    live = [i for i in range(NODES) if i not in dropped]
    assert (attention[..., live].sum(-1) > 0.99).all()


def test_all_sensors_dropped_stays_finite(model):
    """Degenerate input must not produce NaN — softmax over an all-masked row."""
    x = make_batch(list(range(NODES)))
    logits = model(x)
    assert torch.isfinite(logits).all()
    attention = model.attention_weights(x)
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)))


def test_availability_ablation_lets_dropped_sensors_through(edge_index, hop_distance):
    """Without the mask term, dropped sensors DO get attention.

    Guards against the mask being a no-op that happens to look correct because
    zeroed inputs already produce small logits.
    """
    torch.manual_seed(0)
    ablated = FaultQuerySTGNN(edge_index, hop_distance, num_layers=4,
                              use_availability_mask=False).eval()
    attention = ablated.attention_weights(make_batch([0, 5, 11, 17]))
    assert attention[..., [0, 5, 11, 17]].sum() > 0


# ---------------------------------------------------------------------------
# The physics prior actually biases attention
# ---------------------------------------------------------------------------

def test_distance_bias_shifts_attention_toward_near_sensors(edge_index, hop_distance):
    """With a large alpha and no other signal, each query should attend nearest.

    Uses identical sensor embeddings (constant input) so the content term is the
    same for every sensor and the distance bias is the only thing left to break
    the tie.
    """
    torch.manual_seed(0)
    model = FaultQuerySTGNN(edge_index, hop_distance, num_layers=4,
                            alpha_init=5.0, use_reliability=False).eval()
    x = torch.ones(1, NODES, STEPS, FEATURES)
    attention = model.attention_weights(x).mean(dim=1)[0]      # [26, 25]

    for label in range(1, 26):
        nearest = int(np.argmin(hop_distance[label]))
        assert int(attention[label].argmax()) == nearest, (
            f"class {label} should attend most to its nearest sensor")


def test_distance_bias_ablation_removes_the_prior(edge_index, hop_distance):
    torch.manual_seed(0)
    model = FaultQuerySTGNN(edge_index, hop_distance, num_layers=4,
                            alpha_init=5.0, use_reliability=False,
                            use_distance_bias=False).eval()
    x = torch.ones(1, NODES, STEPS, FEATURES)
    attention = model.attention_weights(x).mean(dim=1)[0]
    # With no content signal and no distance term, attention is uniform.
    assert torch.allclose(attention, torch.full_like(attention, 1.0 / NODES),
                          atol=1e-5)


def test_alpha_is_non_negative_under_reparameterisation(model):
    """softplus keeps distance a penalty even if the raw parameter goes negative."""
    with torch.no_grad():
        model.alpha_raw.fill_(-50.0)
    assert (torch.nn.functional.softplus(model.alpha_raw) >= 0).all()


# ---------------------------------------------------------------------------
# Ablations change the parameter count only where they should
# ---------------------------------------------------------------------------

def test_local_gnn_ablation_drops_spatial_layers(edge_index, hop_distance):
    torch.manual_seed(0)
    ablated = FaultQuerySTGNN(edge_index, hop_distance, num_layers=4,
                              use_local_gnn=False).eval()
    assert len(ablated.spatial_layers) == 0
    assert ablated(make_batch()).shape == (BATCH, 1, 26)


def test_gradients_reach_every_mechanism(model):
    logits = model(make_batch())
    logits.sum().backward()
    for name in ("queries", "alpha_raw", "beta"):
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all(), f"{name} got no gradient"
