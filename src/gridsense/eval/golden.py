"""Golden sets: normalisation, span hash, loading, validation, span matching (IMPLEMENTATION.md § 8; data/eval/SCHEMA.md).

A retrieved passage matches a golden context when the document_id is equal and the normalized passage contains the
normalized span. Chunk IDs are never used, so a change of chunk size or strategy does not break the golden set.
Normalisation: collapse whitespace, strip, lower case (the same rule as session B's set, so its hashes carry over).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from gridsense.ingest.parsers import Document

STATUSES = ("draft", "reviewed")
QUERY_TYPES = ("paraphrase", "exact_identifier", "dropout_observability", "unanswerable")


class GoldenError(ValueError):
    pass


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def span_hash(span: str) -> str:
    return hashlib.sha256(norm(span).encode()).hexdigest()


def norm_with_map(text: str) -> tuple[str, list[int]]:
    """Normalized text and, for each normalized character, its index in `text` (to recover section paths)."""
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" "), idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower()), idx.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop(), idx.pop()
    return "".join(out), idx


def locate(doc: Document, span: str) -> list[tuple[int, int]]:
    """Raw-text (start, end) of every occurrence of the normalized span in the document."""
    nt, m = norm_with_map(doc.text)
    ns, hits, pos = norm(span), [], 0
    while ns and (k := nt.find(ns, pos)) >= 0:
        hits.append((m[k], m[k + len(ns) - 1] + 1))
        pos = k + 1
    return hits


def matches(passage_doc_id: str, passage_text: str, ctx: dict) -> bool:
    return passage_doc_id == ctx["document_id"] and norm(ctx["span"]) in norm(passage_text)


def load(path: Path, statuses: tuple[str, ...] = STATUSES, split: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["status"] in statuses and (split is None or r["split"] == split)]


def validate(rows: list[dict], docs: list[Document]) -> list[str]:
    """Problems as text (empty = valid): ids, fields, hashes, and that each span occurs exactly once in its document."""
    by_id, problems, seen = {d.document_id: d for d in docs}, [], set()
    for r in rows:
        q = r.get("query_id")
        if q in seen:
            problems.append(f"{q}: duplicate query_id")
        seen.add(q)
        if r.get("status") not in STATUSES:
            problems.append(f"{q}: status {r.get('status')!r}")
        if r.get("query_type") not in QUERY_TYPES:
            problems.append(f"{q}: query_type {r.get('query_type')!r}")
        if r.get("split") not in ("dev", "test"):
            problems.append(f"{q}: split {r.get('split')!r}")
        if bool(r.get("golden_contexts")) != bool(r.get("answerable")):
            problems.append(f"{q}: answerable={r.get('answerable')} but {len(r.get('golden_contexts', []))} golden contexts")
        for c in r.get("golden_contexts", []):
            d = by_id.get(c["document_id"])
            if d is None:
                problems.append(f"{q}: unknown document_id {c['document_id']}")
                continue
            if c["span_hash"] != span_hash(c["span"]):
                problems.append(f"{q}: span_hash does not match the span")
            n = len(locate(d, c["span"]))
            if n != 1:
                problems.append(f"{q}: span occurs {n} times in {c['document_id']}")
            if c.get("grade") not in (1, 2, 3):
                problems.append(f"{q}: grade {c.get('grade')!r}")
    return problems
