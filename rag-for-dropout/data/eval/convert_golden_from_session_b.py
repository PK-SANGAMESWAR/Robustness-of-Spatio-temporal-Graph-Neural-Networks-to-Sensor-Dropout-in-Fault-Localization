"""Build data/eval/golden_retrieval.jsonl (version 0, all items status draft) from session B's 54 draft queries (author, 2026-10-31).

Each golden span is re-checked against session A's parsed corpus (gridsense.ingest.corpus): it must occur exactly once in its
document. Section paths and span hashes are recomputed. Every item records its source and the re-check result. A span that does
not occur is re-anchored by hand in RE_ANCHOR (the new span must be a substring of the old one, so the meaning does not change).
    PYTHONPATH=src:. .venv/bin/python rag-for-dropout/data/eval/convert_golden_from_session_b.py
"""
from __future__ import annotations

import json
from pathlib import Path

from gridsense.eval.golden import locate, norm, span_hash, validate
from gridsense.ingest.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "rag-for-dropout/_quarantine_session2/rag-for-dropout/data/eval/golden_retrieval.jsonl"
OUT = ROOT / "rag-for-dropout/data/eval/golden_retrieval.jsonl"
RE_ANCHOR = {  # (query_id, document_id) -> new span (substring of the old span); reason in the recheck note
    ("R-0044", "paper1_rev1170"): ("Missingness model: To simulate a lost sensor we select k of the 25 μPMUs and, for every test window, zero all six "
               "input channels of each selected node, both the RMS-voltage and the availability-mask channels",
               "session B's LaTeX parser kept a '(ref)' marker and a no-break space; session A's parser drops the reference; "
               "span cut before the cross-reference"),
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit("session B's files are archived, not in the tree. To re-run, extract ~/archive/gridsense/_quarantine_session2_archived_2026-11-01.tar.gz "
                         "(sha256 cc38a3aff77eca85...) into rag-for-dropout/")
    docs = load_corpus()
    by = {d.document_id: d for d in docs}
    out = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ctxs = []
        for c in r["golden_contexts"]:
            span, check = c["span"], {"result": "found_once", "note": ""}
            if (r["query_id"], c["document_id"]) in RE_ANCHOR:
                new, why = RE_ANCHOR[(r["query_id"], c["document_id"])]
                assert norm(new) in norm(span), r["query_id"]
                span, check = new, {"result": "re_anchored", "note": why}
            d = by[c["document_id"]]
            hits = locate(d, span)
            if len(hits) != 1:
                raise SystemExit(f"{r['query_id']}: span occurs {len(hits)} times in {c['document_id']}")
            path = list(d.section_at(hits[0][0]))
            if path != list(c["section_path"]):
                check["note"] = (check["note"] + "; " if check["note"] else "") + f"section path was {c['section_path']}"
            ctxs.append({"document_id": c["document_id"], "section_path": path, "span": span, "span_hash": span_hash(span),
                         "grade": c["grade"], "recheck": check})
        out.append({**{k: r[k] for k in ("query_id", "query", "query_type")}, "golden_contexts": ctxs,
                    **{k: r[k] for k in ("reference_answer", "answerable", "critical_terms", "split")},
                    "status": "draft", "source": f"session_b:{r['query_id']} ({r['created_by']}, {r['created']})",
                    "created": r["created"], "created_by": r["created_by"], "converted_by": "agent:claude-opus-5-5 (A)",
                    "golden_version": 0, "reviewed": None})
    problems = validate(out, docs)
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"items": len(out), "contexts": sum(len(r["golden_contexts"]) for r in out),
                      "re_anchored": [list(k) for k in RE_ANCHOR], "path_changes": sum("section path was" in c["recheck"]["note"]
                                                                             for r in out for c in r["golden_contexts"])}))


if __name__ == "__main__":
    main()
