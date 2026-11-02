"""One-off: golden retrieval v2 -> v3 (author, O39, 2026-11-02): R-0055 and R-0056 spans accepted (status reviewed); R-0060 rephrased to one clear meaning (stays draft).
    PYTHONPATH=src:. .venv/bin/python rag-for-dropout/data/eval/apply_o39_golden.py"""
import json
import re
from pathlib import Path

from gridsense.eval.golden import validate
from gridsense.ingest.corpus import load_corpus

PATH = Path(__file__).resolve().parent / "golden_retrieval.jsonl"
rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 68 and all(r["golden_version"] == 2 for r in rows)
corpus = "\n".join(d.text for d in load_corpus())
for r in rows:
    r["golden_version"] = 3
    if r["query_id"] in ("R-0055", "R-0056"):
        r["status"], r["reviewed"] = "reviewed", "2026-11-02"
        r["review"] = {"by": "author", "date": "2026-11-02", "round": "O39", "verdict": "accepted (the golden spans chosen by the agent)", "changes": []}
    if r["query_id"] == "R-0060":
        r["query"] = "Which five μPMUs does the targeted study remove when the fault is at bus 57?"
        r["critical_terms"] = ["targeted", "five μPMUs", "bus 57"]
        r["near_miss_of"] = ["feeder_fault_buses", "note_16_targeted_versus_random_dropout"]
        r["absence_check"] = {"patterns_not_found_in_the_corpus": [r"4th nearest", r"fifth nearest", r"5th nearest", r"worst-case run", r"dark in the worst"],
                              "note": "feeder_fault_buses lists the 3 nearest μPMUs of bus 57 (buses 54, 60, 13); note_16 and RESULTS.md describe the targeted rule, not the 5 sensors removed for one fault bus. "
                                      "O39 rephrase: the 2026-11-02 text ('dark in the worst-case run') had two meanings"}
        r["review"] = {"by": "author", "date": "2026-11-02", "round": "O39", "verdict": "rephrase to one clear meaning (done by the agent; draft until the author reads it)", "changes": ["query text"]}
for r in rows:
    if r["status"] == "draft" and r["query_type"] == "unanswerable":
        for pat in r["absence_check"]["patterns_not_found_in_the_corpus"]:
            assert not re.search(pat, corpus, flags=re.I), (r["query_id"], pat)
assert not validate(rows, load_corpus())
PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print({"items": len(rows), "reviewed": sum(r["status"] == "reviewed" for r in rows), "draft": [r["query_id"] for r in rows if r["status"] == "draft"]})
