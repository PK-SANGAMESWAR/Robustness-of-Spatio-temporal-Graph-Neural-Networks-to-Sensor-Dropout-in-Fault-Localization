"""Golden events (O36, author 2026-11-02): 60 draft events from the TEST split at w = 25. Tests of the picker (synthetic) and of the written file (needs the Git-ignored Layer-0 latents)."""
import hashlib
import json

import numpy as np
import pytest

from gridsense.config import ROOT, TRACK
from gridsense.data.splits import event_of, scenario_of, window_index
from gridsense.events import draft_events as D

EVENTS = TRACK / "data" / "eval" / "golden_events.jsonl"


def _items():
    return [json.loads(l) for l in EVENTS.read_text(encoding="utf-8").splitlines()]


def _fake_cls(n, rng):
    cls = {}
    for k in D.KS:
        outcome = rng.integers(0, 2, n)
        correct = rng.random(n) < 0.6
        cls[k] = {"outcome": outcome, "correct": correct, "p": rng.random(n) * 0.5 + 0.5, "reason": np.zeros(n, int), "margin": rng.random(n), "pred": np.zeros(n, int),
                  "set": np.zeros((n, 26), bool), "top1_share": rng.random(n)}
    return cls


def test_O36_picker_is_seeded_balanced_and_uses_each_run_once():
    n = 3000
    rng = np.random.default_rng(0)
    types = ["AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC", "ABCG"]
    ids = [f"{1 + i % 25}_{types[i % 11]}_10p0_{i:04d}_w0025" for i in range(n)]
    y = np.array([1 + i % 25 for i in range(n)])
    cls = _fake_cls(n, rng)
    meta = {event_of(w): event_of(w).split("_")[1] for w in ids}
    a, b = D.pick(ids, y, cls, meta, seed=5), D.pick(ids, y, cls, meta, seed=5)
    assert a == b and len(a) == 60
    assert len({event_of(ids[c["row"]]) for c in a}) == 60
    per_k = {k: [c["slot"] for c in a if c["k"] == k] for k in D.KS}
    assert all(len(v) == 12 and v.count("confident_correct") == 4 and v.count("high_conf_error") == 4 and v.count("low_confidence_correct") == 2 for v in per_k.values())
    counts = list(np.bincount([types.index(meta[event_of(ids[c["row"]])]) for c in a], minlength=11))
    assert max(counts) - min(counts) <= 1 and D.pick(ids, y, cls, meta, seed=6) != a


def test_O36_slot_masks_follow_the_definitions():
    c = {"outcome": np.array([0, 0, 0, 1, 1]), "correct": np.array([True, False, False, True, False]), "p": np.array([0.99, 0.95, 0.5, 0.6, 0.7])}
    m = D.slot_masks(c)
    assert m["confident_correct"].tolist() == [True, False, False, False, False]
    assert m["high_conf_error"].tolist() == [False, True, False, False, False]        # a wrong CONFIDENT window with p < 0.9 is not a high-confidence error
    assert m["low_confidence_correct"].tolist() == [False] * 3 + [True, False] and m["low_confidence_wrong"].tolist() == [False] * 4 + [True]


@pytest.fixture(scope="module")
def items():
    if not EVENTS.exists() or not (D.LAT / "test" / "window_ids.txt").exists():
        pytest.skip("golden_events.jsonl or the Layer-0 latents are missing")
    return _items()


def test_O36_events_cover_every_k_and_fault_type_on_the_test_split_only(items):
    assert len(items) == 60 and len({i["event_id"] for i in items}) == 60
    assert all(i["split"] == "test" and window_index(i["window_id"]) == 25 and i["event_object"]["report_window_index"] == 25 for i in items)
    assert {i["k"] for i in items} == set(D.KS) and all(sum(i["k"] == k for i in items) == 12 for k in D.KS)
    assert len({i["fault_type"] for i in items}) == 11
    test_events = {event_of(w) for w in (ROOT / "dataset_fullscale_50" / "splits" / "test.txt").read_text().split()}
    assert {i["event_id"] for i in items} <= test_events
    assert {i["stratum"] for i in items} == {"confident_correct", "high_conf_error", "low_confidence"}
    assert all(i["event_object"]["calibrated_probability"] >= 0.9 for i in items if i["stratum"] == "high_conf_error")


def test_O36_neighbours_are_train_only_and_held_out(items):
    test_events = {event_of(w) for w in (ROOT / "dataset_fullscale_50" / "splits" / "test.txt").read_text().split()}
    for i in items:
        nb = i["event_object"]["knn"]["neighbours"]
        assert i["event_object"]["knn"]["held_out"] and len(nb) == 10
        for n in nb:
            assert event_of(n["window_id"]) not in test_events and scenario_of(event_of(n["window_id"])) != scenario_of(i["event_id"])


def test_O36_event_object_matches_a_recomputed_gate_and_candidate_set(items):
    from gridsense.verification.candidate_sets import CandidateSets
    from gridsense.verification.gate import OUTCOME_NAMES, Gate
    ids = (D.LAT / "test" / "window_ids.txt").read_text().split()
    row = {w: r for r, w in enumerate(ids)}
    gate, cs = Gate.load(TRACK / "configs" / "gate_v1.json"), CandidateSets.load(TRACK / "configs" / "candidate_sets_v1.json")
    from fault_localization.config.constants import FAULT_BUSES
    for i in items:
        o, k = i["event_object"], i["k"]
        v = np.load(D.LAT / "test" / f"votes_k{k}_o0.npy", mmap_mode="r")[row[i["window_id"]]][None].astype(float)
        r = gate.decide(k, v, np.zeros(1))
        assert OUTCOME_NAMES[int(r["outcome"][0])] == o["gate_outcome"] and o["predicted_class"] == int(v.argmax())
        assert abs(float(r["p"][0]) - o["calibrated_probability"]) < 1e-3
        if o["gate_outcome"] == "LOW_CONFIDENCE":
            assert o["candidate_set_buses"] == [int(FAULT_BUSES[q - 1]) for q in np.flatnonzero(cs.sets(k, v)[0]) if q > 0]
        else:
            assert o["candidate_set_buses"] is None
        assert i["prediction_correct"] == (o["predicted_class"] == i["true_label"])


def test_O36_statuses_follow_the_origin_and_the_truth_is_hidden_from_the_event_object(items):
    from gridsense.eval import golden as G
    from gridsense.ingest.corpus import load_corpus
    docs = {d.document_id: d for d in load_corpus()}
    for i in items:
        assert i["status"] == "draft" and i["reviewed"] is None
        assert not ({"true_bus", "true_label", "prediction_correct", "truth_in_candidate_set", "fault_type"} & set(i["event_object"]))
        kinds = {p["kind"] for p in i["reference_report_points"] if p["status"] == "reviewed"}
        assert {"location", "gate_outcome", "neighbour_locations", "dark_sensors"} <= kinds and ("candidate_set" in kinds) == (i["event_object"]["gate_outcome"] == "LOW_CONFIDENCE")
        assert all((p["status"] == "reviewed") == (p["derived_from"] in ("event_object", "author_approved_fixed_statement")) for p in i["reference_report_points"])
        assert len(i["event_object"]["dropped_nodes"]) == i["k"]
        for c in i["golden_passages"]:
            assert len(G.locate(docs[c["document_id"]], c["span"])) == 1 and c["span_hash"] == G.span_hash(c["span"])


def test_O36_written_file_matches_its_selection_record(items):
    sel = json.loads((TRACK / "reports" / "events" / "selection.json").read_text())
    assert sel["sha256_of_jsonl"] == hashlib.sha256(EVENTS.read_bytes()).hexdigest() and sel["seed"] == 36
    assert max(i["replay"]["max_vote_diff_vs_stored"] for i in items) < 1e-3


def test_O36_s4_selection_is_two_per_cell_seeded_and_not_by_hand(items):
    sel = D.select_s4(items)
    assert len(sel["selected"]) == 30 == len(set(sel["selected"])) and len(sel["cells"]) == 15
    by = {i["event_id"]: i for i in items}
    for c in sel["cells"]:
        assert len(c["selected"]) == 2 and set(c["selected"]) <= set(c["pool"]) and c["seed"] == D.S4_SEED_BASE + 10 * c["k"] + D.STRATA.index(c["stratum"])
        assert all(by[e]["k"] == c["k"] and by[e]["stratum"] == c["stratum"] for e in c["selected"])
    assert D.select_s4(items) == sel                                                   # deterministic
    written = json.loads((TRACK / "reports" / "events" / "s4_selection.json").read_text())
    assert written["selected"] == sel["selected"]


def test_O36_fixed_statements_are_approved_and_the_review_sheets_hold_the_other_points(items):
    import csv
    pts = [p for i in items for p in i["reference_report_points"]]
    fixed = [p for p in pts if p["kind"] in D.FIXED_KINDS]
    assert len(fixed) == 180 and {p["text"] for p in fixed} == set(D.FIXED_STATEMENTS.values()) and all(p["status"] == "reviewed" for p in fixed)
    assert D.FIXED_STATEMENTS["limits"] == "The report does not claim that the result shows safety or performance on a real grid."       # changed by the author 2026-11-03
    draft = [p for p in pts if p["status"] == "draft"]
    with (TRACK / "reports" / "events" / "judgement_points_review.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(draft) == 67 and not {r["kind"] for r in rows} & set(D.FIXED_KINDS)
    s4 = set(D.select_s4(items)["selected"])
    with (TRACK / "reports" / "events" / "judgement_points_s4.csv").open(encoding="utf-8") as f:
        rows4 = list(csv.DictReader(f))
    assert len(rows4) == sum(p["status"] == "draft" for i in items if i["event_id"] in s4 for p in i["reference_report_points"]) == 33
    assert {r["event_id"] for r in rows4} <= s4 and {r["point_id"] for r in rows4} <= {r["point_id"] for r in rows}


def test_O36_stratum_shares_cover_the_pool():
    sh = json.loads((TRACK / "reports" / "events" / "stratum_shares.json").read_text())
    assert set(sh) == {"0", "1", "2", "5", "10"}
    for v in sh.values():
        assert abs(sum(x for k, x in v.items() if k != "runs") - 1.0) < 1e-3 and v["runs"] == 6175
