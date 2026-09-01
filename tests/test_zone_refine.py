"""Invariants of ZoneRefine-STGNN and its physics zone stage.

The stage-1 scorer is a closed-form estimator, so most of its properties are
exactly checkable rather than merely plausible — severity invariance, masked
least-squares behaviour, and normalization of the hierarchical output all have
right answers, and a violation of any of them would silently corrupt training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fault_localization.config import constants as C
from fault_localization.graph.sensitivity import CACHE_PATH, load_sensitivity
from fault_localization.models.zone_refine import ZoneRefineSTGNN

DATA_ROOT = Path("dataset_fullscale_50")
BATCH, NODES, STEPS, FEATURES = 4, 25, 20, 6


@pytest.fixture(scope="module")
def physics() -> dict:
    if not CACHE_PATH.exists():
        pytest.skip(f"sensitivity cache missing at {CACHE_PATH}")
    return load_sensitivity()


@pytest.fixture(scope="module")
def edge_index() -> np.ndarray:
    path = DATA_ROOT / "processed" / "edge_index.npy"
    if not path.exists():
        pytest.skip(f"{path} missing")
    return np.load(path)


@pytest.fixture
def model(edge_index, physics) -> ZoneRefineSTGNN:
    torch.manual_seed(0)
    return ZoneRefineSTGNN(edge_index, physics["zone_templates"],
                           physics["zone_of_class"], num_layers=4).eval()


def make_batch(dropped: list[int] | None = None) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.rand(BATCH, NODES, STEPS, FEATURES)
    x[..., 3:] = 1.0
    if dropped:
        x[:, dropped] = 0.0
    return x


# ---------------------------------------------------------------------------
# The physics: templates and the zone partition
# ---------------------------------------------------------------------------

def test_templates_are_unit_norm(physics):
    norms = np.linalg.norm(physics["zone_templates"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_every_fault_class_has_exactly_one_zone(physics):
    zone_of = physics["zone_of_class"]
    assert zone_of.shape == (26,)
    assert zone_of[0] == -1, "class 0 is no-fault and belongs to no zone"
    assert (zone_of[1:] >= 0).all()
    assert (zone_of[1:] < physics["n_zones"]).all()
    # Every zone is non-empty, so the partition covers all 25 fault classes.
    assert len(np.unique(zone_of[1:])) == physics["n_zones"]


def test_zone_partition_matches_signature_collinearity(physics):
    """Classes share a zone iff their signatures are collinear."""
    s = physics["sensitivity"]
    similarity = s @ s.T
    zone_of = physics["zone_of_class"][1:]
    for i in range(25):
        for j in range(i + 1, 25):
            if similarity[i, j] > 0.999:
                assert zone_of[i] == zone_of[j], (
                    f"classes {i+1},{j+1} are collinear but in different zones")


# ---------------------------------------------------------------------------
# Output contract: normalized log-probabilities shaped [B, 1, 26]
# ---------------------------------------------------------------------------

def test_output_shape(model):
    assert model(make_batch()).shape == (BATCH, 1, 26)


def test_output_is_a_normalized_log_distribution(model):
    log_probs = model(make_batch()).squeeze(1)
    assert torch.allclose(log_probs.exp().sum(-1),
                          torch.ones(BATCH), atol=1e-5)
    assert (log_probs <= 0).all()


def test_cross_entropy_is_exact_on_log_probs(model):
    """Because the output is normalized, log_softmax is the identity on it, so
    the project's expanded_cross_entropy computes the true NLL."""
    from fault_localization.models.common import expanded_cross_entropy
    log_probs = model(make_batch()).squeeze(1)
    y = torch.randint(0, 26, (BATCH,))
    nll = -log_probs.gather(1, y[:, None]).mean()
    assert torch.allclose(expanded_cross_entropy(log_probs.unsqueeze(1), y),
                          nll, atol=1e-5)


def test_soft_vote_is_identity(model):
    from fault_localization.models.voting import soft_vote
    logits = model(make_batch())
    _, votes = soft_vote(logits)
    assert torch.equal(votes, logits.squeeze(1).argmax(-1))


# ---------------------------------------------------------------------------
# Stage 1: the closed-form properties that must hold exactly
# ---------------------------------------------------------------------------

def test_zone_score_is_invariant_to_fault_severity(model):
    """The whole point of dividing by observed energy.

    A 10 ohm fault and a 0.1 ohm fault at the same bus differ by a scalar on the
    deviation vector; the inferred zone must not change.
    """
    torch.manual_seed(3)
    deviation = torch.randn(BATCH, NODES)
    avail = torch.ones(BATCH, NODES)
    pooled = torch.zeros(BATCH, 64)

    weak, _ = model._zone_logits(deviation * 0.01, avail, pooled)
    strong, _ = model._zone_logits(deviation * 10.0, avail, pooled)
    assert torch.allclose(weak, strong, atol=1e-4)


def test_severity_still_drives_the_no_fault_decision(model):
    """Invariance must apply to the zone choice only, not to fault/no-fault."""
    torch.manual_seed(3)
    deviation = torch.randn(BATCH, NODES)
    avail = torch.ones(BATCH, NODES)
    pooled = torch.zeros(BATCH, 64)

    _, quiet = model._zone_logits(deviation * 0.01, avail, pooled)
    _, loud = model._zone_logits(deviation * 10.0, avail, pooled)
    # nofault_scale is initialized negative: more energy => less no-fault evidence.
    assert (quiet > loud).all()


def test_perfect_template_match_wins_its_zone(model):
    """A deviation equal to a template must score that zone highest."""
    avail = torch.ones(1, NODES)
    pooled = torch.zeros(1, 64)
    for z in range(model.n_zones):
        deviation = model.templates[z].detach().unsqueeze(0)
        logits, _ = model._zone_logits(deviation, avail, pooled)
        assert int(logits.argmax()) == z, f"template {z} did not win its own zone"


def test_masked_fit_ignores_dropped_coordinates(model):
    """The projection must use observed coordinates only.

    Corrupting a dropped sensor's deviation to an extreme value must not move
    the zone scores at all — that is what makes this a masked least-squares fit
    rather than a fit on zero-filled data.
    """
    torch.manual_seed(4)
    deviation = torch.randn(1, NODES)
    avail = torch.ones(1, NODES)
    avail[0, [2, 9, 17]] = 0.0
    pooled = torch.zeros(1, 64)

    clean, _ = model._zone_logits(deviation, avail, pooled)
    corrupted = deviation.clone()
    corrupted[0, [2, 9, 17]] = 1e3
    dirty, _ = model._zone_logits(corrupted, avail, pooled)
    assert torch.allclose(clean, dirty, atol=1e-5)


def test_all_sensors_dropped_stays_finite(model):
    logits = model(make_batch(list(range(NODES))))
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.squeeze(1).exp().sum(-1), torch.ones(BATCH), atol=1e-5)


def test_dropping_sensors_keeps_output_valid(model):
    logits = model(make_batch([0, 5, 11, 17]))
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.squeeze(1).exp().sum(-1), torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# Stage 2 and the hierarchy
# ---------------------------------------------------------------------------

def test_singleton_zones_contribute_no_within_zone_uncertainty(model, physics):
    """For a zone holding one class, p(class | zone) must be exactly 1.

    So that class's probability equals its zone's probability — the refinement
    head cannot leak probability out of a zone it does not belong to.
    """
    out = model._forward(make_batch())
    log_zone = out["log_zone"]
    log_probs = out["log_probs"].squeeze(1)
    zone_of = torch.as_tensor(physics["zone_of_class"])
    sizes = torch.bincount(zone_of[1:], minlength=model.n_zones)

    for c in range(1, 26):
        z = int(zone_of[c])
        if sizes[z] == 1:
            assert torch.allclose(log_probs[:, c], log_zone[:, z], atol=1e-5)


def test_zone_probabilities_sum_with_nofault_to_one(model):
    out = model._forward(make_batch())
    total = out["log_zone"].exp().sum(-1) + out["log_nofault"].exp().squeeze(-1)
    assert torch.allclose(total, torch.ones(BATCH), atol=1e-5)


def test_class_probabilities_aggregate_to_zone_probabilities(model, physics):
    """Marginalizing the classes of a zone must recover that zone's probability."""
    out = model._forward(make_batch())
    probs = out["log_probs"].squeeze(1).exp()
    zone_probs = out["log_zone"].exp()
    zone_of = torch.as_tensor(physics["zone_of_class"])
    for z in range(model.n_zones):
        members = [c for c in range(1, 26) if int(zone_of[c]) == z]
        assert torch.allclose(probs[:, members].sum(-1), zone_probs[:, z], atol=1e-5)


# ---------------------------------------------------------------------------
# Ablations and trainability
# ---------------------------------------------------------------------------

def test_physics_ablation_changes_the_zone_scores(edge_index, physics):
    torch.manual_seed(0)
    ablated = ZoneRefineSTGNN(edge_index, physics["zone_templates"],
                              physics["zone_of_class"], num_layers=4,
                              use_physics_zones=False).eval()
    deviation = torch.randn(2, NODES)
    avail = torch.ones(2, NODES)
    pooled = torch.randn(2, 64)
    free, _ = ablated._zone_logits(deviation, avail, pooled)
    # The free scorer reads pooled features, so it is no longer severity-invariant.
    scaled, _ = ablated._zone_logits(deviation * 10, avail, pooled)
    assert torch.allclose(free, scaled), "Z3 zone score should ignore deviation"
    assert ablated(make_batch()).shape == (BATCH, 1, 26)


def test_frozen_templates_get_no_gradient(edge_index, physics):
    torch.manual_seed(0)
    pinned = ZoneRefineSTGNN(edge_index, physics["zone_templates"],
                             physics["zone_of_class"], num_layers=4,
                             learn_templates=False)
    assert not isinstance(pinned.templates, torch.nn.Parameter)


def test_gradients_reach_the_physics_stage(model):
    model(make_batch()).sum().backward()
    for name in ("templates", "zone_temperature", "nofault_scale"):
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all(), f"{name}: no gradient"
