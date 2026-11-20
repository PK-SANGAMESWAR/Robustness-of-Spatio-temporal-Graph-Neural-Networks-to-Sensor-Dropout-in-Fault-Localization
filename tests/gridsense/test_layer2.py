"""Layer 2 (generator with golden context): selection rule by hand, caps, aborts, no live call. Mock generator only."""
from __future__ import annotations

import pytest

from gridsense.eval import layer2 as L2
from gridsense.s4.llm import CostLimitError


def _rec(model, ok=True, usd=0.001, status="sufficient", fail_check=None):
    chk = {c: {"pass": True, "detail": ""} for c in L2.CHECKS}
    if fail_check:
        chk[fail_check] = {"pass": False, "detail": "x"}
    return {"model": model, "event_id": "e1", "generation": {"usd": usd, "cached": False, "latency_s": 2.0, "input_tokens": 1000, "output_tokens": 500},
            "checks": chk, "evidence": {"final": status, "generator_status": status, "generator_coverage_incomplete": False, "generator_review_while_insufficient": False}}


def test_L2_summary_meets_floors_only_when_every_report_passes_every_check_and_none_fails():
    s = L2.summarize([_rec("a"), _rec("a")])
    assert s["meets_floors"] and s["cost_per_report_usd"] == pytest.approx(0.001) and s["generation_latency"]["p50_s"] == 2.0
    assert not L2.summarize([_rec("a"), _rec("a", fail_check="citation_validity")])["meets_floors"]
    assert not L2.summarize([_rec("a"), {"model": "a", "event_id": "e2", "error": "bad json"}])["meets_floors"]        # a parse failure breaks the floor
    flip = L2.summarize([_rec("a", status="sufficient"), _rec("a", status="insufficient")])
    assert flip["events_with_status_flip"] == {"events": 1, "flipping": 1}


def test_L2_choose_is_the_cheapest_that_meets_the_floors_not_the_best_of_the_rest():
    s = {"cheap_but_fails": L2.summarize([_rec("x", usd=0.0001, fail_check="field_consistency")]), "mid": L2.summarize([_rec("m", usd=0.002)]), "dear": L2.summarize([_rec("d", usd=0.01)])}
    d = L2.choose(s)
    assert d["chosen"] == "mid" and d["eligible"] == ["dear", "mid"]
    assert L2.choose({"x": s["cheap_but_fails"]}) == {"chosen": None, "reason": "no candidate meets every floor"}


def test_L2_plan_is_capped_and_fixed_before_the_run():
    assert sum(cap for _, _, cap in L2.PLAN.values()) <= 1.0 and "gpt-6.1-terra" not in L2.PLAN                      # the judge is never a candidate
    assert all(n <= 30 and r <= 3 for n, r, _ in L2.PLAN.values())


def test_L2_golden_passages_are_the_golden_spans():
    ev = {"golden_passages": [{"document_id": "d", "section_path": ["S"], "span": "text of the span", "grade": 3}]}
    p = L2.golden_passages(ev)
    assert p[0]["text"] == "text of the span" and p[0]["document_id"] == "d" and p[0]["section_path"] == ["S"]


def test_L2_a_candidate_is_aborted_after_consecutive_api_errors_and_the_cap_stops_it(monkeypatch, tmp_path):
    from gridsense.eval import snapshot as SNAP
    monkeypatch.setattr(SNAP, "SNAP_DIR", tmp_path / "snap"), monkeypatch.setattr(SNAP, "ITEMS_DIR", tmp_path / "items"), monkeypatch.setattr(SNAP, "ROOT", tmp_path), monkeypatch.setattr(L2, "ROOT", tmp_path)
    def boom(*a, **k):
        raise ConnectionError("api down")
    monkeypatch.setattr(L2, "one", boom)
    out = L2.run(False, {"m1": (5, 1, 0.1)})
    s = out["candidates"]["m1"]
    assert s["reports"] == 3 and s["failures"] == 3 and "aborted after 3 consecutive API errors" in s["note"] and out["decision"]["chosen"] is None

    def capped(*a, **k):
        raise CostLimitError("run budget 0.10 USD would be exceeded")
    monkeypatch.setattr(L2, "one", capped)
    import time
    time.sleep(1.1)                                                                                        # snapshots are append-only: one stamp per second
    out = L2.run(False, {"m2": (5, 2, 0.1)})
    assert "stopped at the spend cap" in out["candidates"]["m2"]["note"]


def test_L2_a_live_run_is_refused_without_the_authors_approval(tmp_path, monkeypatch):
    from gridsense.s4 import llm
    monkeypatch.setattr(llm, "APPROVAL", tmp_path / "none.json")
    with pytest.raises(llm.ApprovalMissing):
        L2.run(True, {"gpt-6-luna": (1, 1, 0.01)})
