"""Layer 4, application checks that are deterministic with ground truth (rag-eval.md § 4 Layer 4; EVAL.md § 7). No API call, no judge.

(1) Location correctness: the report names the predicted bus (by stratum, with the truth of that prediction).
(2) Uncertainty correctness: when the prediction is wrong, does the report state doubt? When it is right, does it state doubt without a cause?
    - doubt stated  = the gate outcome is not CONFIDENT, OR the final evidence status is insufficient, OR the report text matches DOUBT (a heuristic regex, listed below).
    - doubt cause   = a cause exists in the event itself: the gate outcome is not CONFIDENT, OR at least one of the 10 neighbours has another bus than the predicted bus.
(3) The question that matters for the evidence layer: for the wrong predictions, how many have NO signal against them anywhere (gate CONFIDENT, all neighbours agree with the wrong bus)?
    Such an error cannot be caught by the report, whatever the generator writes. This is a fact about the event, not about the generator.
Truth comes from the golden events (`prediction_correct`); the generator never saw it. The 30 S4 events over-sample wrong predictions: rates are per stratum with n, never one pooled system score (EVAL.md § 11).
`python -m gridsense.eval.layer4`
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from gridsense.config import ROOT, TRACK
from gridsense.eval.snapshot import ITEMS_DIR, write_snapshot

DOUBT = re.compile(r"\b(disagree\w*|does not (?:support|agree)|do not (?:support|agree)|contradict\w*|conflict\w*|mismatch\w*|inconsisten\w*|but all|differ\w* from|other than the predicted)\b", re.I)
OUT = TRACK / "reports" / "s7"


def _text(rep: dict) -> str:
    return " ".join([str(rep.get("case_evidence_summary", "")), str(rep.get("evidence_status_reason", ""))] + [str(x) for x in rep.get("limitations") or []])


def judge_item(item: dict, event: dict) -> dict:
    """One report against the event it describes."""
    o, rep = event["event_object"], item["report"]
    nb = [n["bus"] for n in o["knn"]["neighbours"]]
    pred = o["predicted_bus"]
    not_confident = o["gate_outcome"] != "CONFIDENT"
    disagree = any(b != pred for b in nb)
    insufficient = item["evidence"]["final"] == "insufficient"
    text_doubt = bool(DOUBT.search(_text(rep)))
    stated = not_confident or insufficient or text_doubt
    cause = not_confident or disagree
    return {"names_predicted_bus": rep.get("predicted_location") == pred, "prediction_correct": bool(item["prediction_correct"]), "doubt_stated": stated, "doubt_cause_in_event": cause,
            "doubt_without_cause": stated and not cause, "no_signal_against_prediction": (not not_confident) and (not disagree),
            "sources": {"gate_not_confident": not_confident, "evidence_insufficient": insufficient, "text_doubt": text_doubt}}


def summarize(rows: list[dict]) -> dict:
    """Per stratum and overall-by-truth. Rates with n; empty cells are None, never 0."""
    def rate(sel, f):
        return {"rate": (sum(f(r) for r in sel) / len(sel)) if sel else None, "n": len(sel)}
    out: dict = {}
    for s in ("high_conf_error", "confident_correct", "low_confidence"):
        sel = [r for r in rows if r["stratum"] == s]
        wrong, right = [r for r in sel if not r["prediction_correct"]], [r for r in sel if r["prediction_correct"]]
        out[s] = {"reports": len(sel), "distinct_events": len({r["event_id"] for r in sel}), "names_predicted_bus": rate(sel, lambda r: r["names_predicted_bus"]),
                  "wrong_predictions": {"n": len(wrong), "doubt_stated": rate(wrong, lambda r: r["doubt_stated"]), "no_signal_against_prediction": rate(wrong, lambda r: r["no_signal_against_prediction"]),
                                        "doubt_by_gate": rate(wrong, lambda r: r["sources"]["gate_not_confident"]), "doubt_by_insufficient_evidence": rate(wrong, lambda r: r["sources"]["evidence_insufficient"]),
                                        "doubt_by_text": rate(wrong, lambda r: r["sources"]["text_doubt"])},
                  "right_predictions": {"n": len(right), "doubt_stated": rate(right, lambda r: r["doubt_stated"]), "doubt_without_cause": rate(right, lambda r: r["doubt_without_cause"])}}
    return out


def build(paths: list) -> dict:
    from gridsense.s4 import run as R
    ev = R.load_events()
    rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            if it.get("error") or "report" not in it:
                continue
            rows.append(judge_item(it, ev[it["event_id"]]) | {"stratum": it["stratum"], "event_id": it["event_id"], "source": p.name})
    return {"rows": rows, "summary": summarize(rows), "sources": [p.name for p in paths]}


def markdown(d: dict) -> str:
    f = lambda x: "–" if x["rate"] is None else f"{x['rate']:.2f} (n={x['n']})"
    S = d["summary"]
    L = ["# Layer 4: location and uncertainty correctness (deterministic, with ground truth)", "",
         f"Written for the author. Source: the rescored S4 live items ({', '.join(d['sources'])}); {len(d['rows'])} reports from {len({r['event_id'] for r in d['rows']})} distinct events (the noise events are a subset of the main events). Truth comes from the golden events. Draft events. Heuristic doubt regex: `{DOUBT.pattern}`. "
         "No pooled score: the S4 events over-sample wrong predictions.", "", "## Location correctness: the report names the predicted bus", "", "| stratum | reports | distinct events | names the predicted bus |", "|---|---|---|---|"]
    L += [f"| {s} | {v['reports']} | {v['distinct_events']} | {f(v['names_predicted_bus'])} |" for s, v in S.items()]
    L += ["", "Repeats of one event are not independent: the rates below are over reports, the distinct-event counts show how few events stand behind them."]
    L += ["", "## Wrong predictions: does the report state doubt, and was any signal against the prediction available?", "",
          "| stratum | wrong n | doubt stated | by gate | by insufficient evidence | by text | **no signal against the wrong prediction** |", "|---|---|---|---|---|---|---|"]
    for s, v in S.items():
        w = v["wrong_predictions"]
        L.append(f"| {s} | {w['n']} | {f(w['doubt_stated'])} | {f(w['doubt_by_gate'])} | {f(w['doubt_by_insufficient_evidence'])} | {f(w['doubt_by_text'])} | {f(w['no_signal_against_prediction'])} |")
    L += ["", "'No signal' = gate CONFIDENT and all 10 neighbours have the predicted (wrong) bus. No generator can catch such an error from this evidence.",
          "'By insufficient evidence' follows the retrieved passages, not the risk (E27, E28): read it next to the right-prediction table below before you take it as a risk-aware doubt.", "",
          "## Right predictions: doubt without a cause", "", "| stratum | right n | doubt stated | doubt without a cause |", "|---|---|---|---|"]
    L += [f"| {s} | {v['right_predictions']['n']} | {f(v['right_predictions']['doubt_stated'])} | {f(v['right_predictions']['doubt_without_cause'])} |" for s, v in S.items()]
    return "\n".join(L) + "\n"


def main() -> None:
    paths = [sorted(ITEMS_DIR.glob(f"*s4_{k}_live_rescored.jsonl"))[-1] for k in ("main", "noise")]       # the latest rescored pair (E29, E30)
    d = build(paths)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LAYER4.md").write_text(markdown(d), encoding="utf-8")
    snap = write_snapshot("s7_layer4", {"layer4": d["summary"]}, d["rows"], config_effective={"sources": d["sources"], "doubt_regex": DOUBT.pattern},
                          notes="Layer 4 deterministic checks with ground truth on the rescored S4 live items (draft events); heuristic doubt regex")
    print(json.dumps({"markdown": str((OUT / "LAYER4.md").relative_to(ROOT)), "snapshot": str(snap.relative_to(ROOT))}, indent=1))


if __name__ == "__main__":
    main()
