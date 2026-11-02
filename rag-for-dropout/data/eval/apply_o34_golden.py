"""One-off: golden retrieval v1 -> v2 (author, 2026-11-02): R-0055 and R-0056 go back to `draft` (the author checks the spans that the agent chose); all other items stay reviewed.
    PYTHONPATH=src:. .venv/bin/python rag-for-dropout/data/eval/apply_o34_golden.py"""
import json
from pathlib import Path

from gridsense.eval.golden import validate
from gridsense.ingest.corpus import load_corpus

PATH = Path(__file__).resolve().parent / "golden_retrieval.jsonl"
rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 56 and all(r["golden_version"] == 1 for r in rows)
for r in rows:
    r["golden_version"] = 2
    if r["query_id"] in ("R-0055", "R-0056"):
        r["status"], r["reviewed"] = "draft", None
        r["review"] = {"by": "author", "date": "2026-11-02", "round": "O34", "verdict": "keep draft until the author has checked the golden spans (chosen by the agent)", "changes": []}
assert not validate(rows, load_corpus())
PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print({"items": len(rows), "reviewed": sum(r["status"] == "reviewed" for r in rows), "draft": [r["query_id"] for r in rows if r["status"] == "draft"]})
