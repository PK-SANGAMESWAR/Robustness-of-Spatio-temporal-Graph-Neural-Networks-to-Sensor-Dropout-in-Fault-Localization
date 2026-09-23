"""OOD seeds must be disjoint from the corpus's, provably.

The central claim of Stage C is that the operator handles fault resistances the
network never saw.  If an OOD event happened to reuse a corpus event's seed it
would reuse that event's load draw, and the claim would rest on a partially
shared random stream.  ``deterministic_seed`` also physically cannot generate
these events -- it indexes ``FAULT_RESISTANCES`` and raises for anything outside
{0.1, 1, 10} -- so a separate function is required anyway
(``PC-STGNN-PLAN.md`` §0, ``04_STAGE_C_gonogo.md`` S2.0).
"""

from __future__ import annotations

import pytest

from fault_localization.config.constants import (
    FAULT_BUSES,
    FAULT_RESISTANCES,
    FAULT_TYPES,
    OOD_SEED_OFFSET,
    RUNS_PER_SCENARIO,
    deterministic_seed,
    deterministic_seed_ood,
)

PILOT_R = [5.0, 20.0]
TEST_R = [3.0, 5.0, 20.0, 30.0]


def corpus_seeds() -> set[int]:
    """Every seed the frozen corpus could have drawn, enumerated exhaustively."""
    return {
        deterministic_seed(bus, ftype, r, run)
        for bus in FAULT_BUSES
        for ftype in FAULT_TYPES
        for r in FAULT_RESISTANCES
        for run in range(RUNS_PER_SCENARIO)
    }


def test_the_old_seed_function_cannot_express_an_ood_event():
    """The reason a second function exists at all."""
    with pytest.raises(ValueError):
        deterministic_seed(13, "AG", 5.0, 1000)


def test_ood_seeds_are_deterministic():
    a = deterministic_seed_ood(13, "AG", 5.0, 1000, "pilot")
    b = deterministic_seed_ood(13, "AG", 5.0, 1000, "pilot")
    assert a == b


def test_ood_seeds_sit_in_their_own_range():
    for set_name, resistances in (("pilot", PILOT_R), ("test", TEST_R)):
        for r in resistances:
            seed = deterministic_seed_ood(FAULT_BUSES[3], "BCG", r, 2000, set_name)
            assert OOD_SEED_OFFSET <= seed < 2 ** 31


def test_no_collision_with_any_corpus_seed():
    """Exhaustive on the corpus side, broad on the OOD side."""
    corpus = corpus_seeds()
    assert max(corpus) < OOD_SEED_OFFSET      # disjoint by construction
    ood = {
        deterministic_seed_ood(bus, ftype, r, run, set_name)
        for bus in FAULT_BUSES
        for ftype in FAULT_TYPES
        for set_name, resistances, runs in (
            ("pilot", PILOT_R, range(1000, 1002)),
            ("test", TEST_R, range(2000, 2010)),
        )
        for r in resistances
        for run in runs
    }
    assert not (corpus & ood)


def test_pilot_and_sealed_sets_share_no_seeds():
    """The pilot informs the Gate-2 decision; the sealed set carries the claim.
    They must not overlap, or the decision contaminates the evidence."""
    pilot = {
        deterministic_seed_ood(bus, ftype, r, run, "pilot")
        for bus in FAULT_BUSES for ftype in FAULT_TYPES
        for r in PILOT_R for run in range(1000, 1002)
    }
    sealed = {
        deterministic_seed_ood(bus, ftype, r, run, "test")
        for bus in FAULT_BUSES for ftype in FAULT_TYPES
        for r in TEST_R for run in range(2000, 2010)
    }
    assert not (pilot & sealed)


def test_ood_seeds_are_unique_within_a_set():
    """A collision inside the sealed set would silently duplicate events."""
    seeds = [
        deterministic_seed_ood(bus, ftype, r, run, "test")
        for bus in FAULT_BUSES for ftype in FAULT_TYPES
        for r in TEST_R for run in range(2000, 2010)
    ]
    assert len(seeds) == len(set(seeds)) == 25 * 11 * 4 * 10


def test_resistance_is_keyed_by_value_not_index():
    """5 ohm and 20 ohm are different events even though neither is in the corpus."""
    assert (deterministic_seed_ood(13, "AG", 5.0, 2000)
            != deterministic_seed_ood(13, "AG", 20.0, 2000))


def test_rejects_unknown_bus_or_type():
    with pytest.raises(ValueError):
        deterministic_seed_ood(9999, "AG", 5.0, 2000)
    with pytest.raises(ValueError):
        deterministic_seed_ood(13, "XY", 5.0, 2000)
