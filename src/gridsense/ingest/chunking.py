"""Six chunking strategies (EVAL.md § 3 step 1; implementation-plan.md § 6). Token lengths use the embedder's own tokenizer.

Every chunk keeps character offsets into the parsed document text, so golden spans match by text, never by chunk ID.
`embed_text` is what the embedder sees (a section header is prepended for structure-aware and parent-child); `text` is what
the retriever returns as context (the parent section for parent-child).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gridsense.ingest.parsers import Document

CHUNKER_VERSION = "chunking_v1"
STRATEGIES = ("fixed_token", "recursive", "structure_aware", "sentence", "semantic", "parent_child")
Counter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    section_path: tuple[str, ...]
    start: int
    end: int
    text: str          # returned as context
    embed_text: str    # given to the embedder
    parent_id: str | None = None


def _cid(doc: Document, start: int, end: int, strategy: str) -> str:
    h = hashlib.sha256(f"{doc.document_id}|{doc.version}|{strategy}|{start}|{end}".encode()).hexdigest()[:16]
    return f"{doc.document_id}:{start}-{end}:{h}"


def _chunk(doc: Document, start: int, end: int, strategy: str, header: bool = False, ctx: tuple[int, int] | None = None,
           parent_id: str | None = None) -> Chunk:
    path = doc.section_at(start)
    body = doc.text[start:end]
    embed = (f"{doc.title} > {' > '.join(path)}\n{body}" if header and path else body)
    c0, c1 = ctx or (start, end)
    return Chunk(_cid(doc, start, end, strategy), doc.document_id, path, c0, c1, doc.text[c0:c1], embed, parent_id)


def _pieces(text: str, start: int, end: int, seps: tuple[str, ...]) -> list[tuple[int, int]]:
    """Split [start, end) at the first separator that occurs; keep the separator with the left piece."""
    for sep in seps:
        idx = [m.end() for m in re.finditer(re.escape(sep), text[start:end])]
        if idx:
            cuts = [start] + [start + i for i in idx if start + i < end] + [end]
            return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    return [(start, end)]


def _recursive(text: str, start: int, end: int, size: int, count: Counter, seps=("\n\n", "\n", ". ", " ")) -> list[tuple[int, int]]:
    """Recursive splitting by separators, then greedy merging of pieces up to `size` tokens."""
    if count(text[start:end]) <= size:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    for i, (a, b) in enumerate(_pieces(text, start, end, seps)):
        if count(text[a:b]) > size:
            out += _recursive(text, a, b, size, count, seps[1:] if len(seps) > 1 else seps) if len(seps) > 1 else [(a, b)]
        else:
            out.append((a, b))
    merged: list[tuple[int, int]] = []
    for a, b in out:
        if merged and count(text[merged[-1][0]:b]) <= size:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _overlap(spans: list[tuple[int, int]], text: str, overlap_frac: float, count: Counter, size: int) -> list[tuple[int, int]]:
    """Extend each span to the left by about overlap_frac × size tokens, at a word boundary."""
    if overlap_frac <= 0:
        return spans
    want, out = int(round(overlap_frac * size)), []
    for i, (a, b) in enumerate(spans):
        if i == 0:
            out.append((a, b))
            continue
        lo = a
        while lo > spans[i - 1][0] and count(text[lo:a]) < want:
            nxt = text.rfind(" ", spans[i - 1][0], lo - 1)
            if nxt < 0:
                break
            lo = nxt + 1
        out.append((lo, b))
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n{2,}")


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    cuts = [start] + [start + m.end() for m in _SENT.finditer(text[start:end])] + [end]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def markdown_tables(text: str) -> list[dict]:
    """Markdown tables of a parsed document: a line that starts with '|' followed by a separator line (|---|---|), then rows while the lines start with '|'.
    Each table: block span, header line, and the data rows with their character spans (offsets into `text`)."""
    lines, pos = [], 0
    for ln in text.splitlines(keepends=True):
        lines.append((pos, pos + len(ln), ln.rstrip("\n")))
        pos += len(ln)
    tables, i = [], 0
    while i < len(lines) - 1:
        if lines[i][2].lstrip().startswith("|") and _SEPARATOR.match(lines[i + 1][2]) and "|" in lines[i + 1][2]:
            j, rows = i + 2, []
            while j < len(lines) and lines[j][2].lstrip().startswith("|"):
                rows.append(lines[j])
                j += 1
            tables.append({"start": lines[i][0], "end": lines[j - 1][1] if rows else lines[i + 1][1], "header": lines[i][2], "rows": rows})
            i = j
        else:
            i += 1
    return tables


def _chunk_with_tables(doc: Document, strategy: str, size: int, count: Counter, overlap: float) -> list[Chunk]:
    """Plan section 5.4: every table row becomes its own chunk with the table header repeated; the text between tables is chunked as before.
    `text` (what the retriever returns and what golden spans are matched against) = header line + row; `embed_text` also holds the document title and the section path."""
    if strategy != "recursive":
        raise NotImplementedError("table_rows is implemented for the recursive chunker (the frozen winner)")
    t, out, cursor = doc.text, [], 0
    segments = [(tb["start"], tb) for tb in markdown_tables(t)]
    for tb_start, tb in segments + [(len(t), None)]:
        if tb_start > cursor and t[cursor:tb_start].strip():
            spans = _overlap(_recursive(t, cursor, tb_start, size, count), t, overlap, count, size)
            out += [_chunk(doc, a, b, strategy) for a, b in spans if t[a:b].strip()]
        if tb is not None:
            path = doc.section_at(tb["start"])
            prefix = f"{doc.title} > {' > '.join(path)}\n" if path else ""
            for a, b, row in tb["rows"]:
                body = tb["header"] + "\n" + row
                out.append(Chunk(_cid(doc, a, b, "table_row"), doc.document_id, path, a, b, body, prefix + body, None))
            cursor = tb["end"]
    return out


def chunk_document(doc: Document, strategy: str, size: int, count: Counter, overlap: float = 0.0, *,
                   parent_size: int | None = None, embed_fn: Callable[[list[str]], np.ndarray] | None = None,
                   semantic_percentile: float = 20.0, table_rows: bool = False) -> list[Chunk]:
    """Split one document. `size` is the embedded chunk length in tokens (child size for parent_child)."""
    t = doc.text
    if table_rows:
        return _chunk_with_tables(doc, strategy, size, count, overlap)
    if strategy == "fixed_token":
        spans, a = [], 0
        words = [m.start() for m in re.finditer(r"\S+", t)]
        while a < len(words):
            b = a + 1
            while b < len(words) and count(t[words[a]:words[b]]) < size:
                b += 1
            spans.append((words[a], words[b] if b < len(words) else len(t)))
            a = b
        return [_chunk(doc, a, b, strategy) for a, b in _overlap(spans, t, overlap, count, size)]
    if strategy == "recursive":
        spans = _recursive(t, 0, len(t), size, count)
        return [_chunk(doc, a, b, strategy) for a, b in _overlap(spans, t, overlap, count, size)]
    if strategy == "structure_aware":
        out = []
        for s in doc.sections:
            spans = _recursive(t, s.start, s.end, size, count)
            out += [_chunk(doc, a, b, strategy, header=True) for a, b in _overlap(spans, t, overlap, count, size)]
        return out
    if strategy == "sentence":
        spans: list[tuple[int, int]] = []
        for a, b in _sentences(t, 0, len(t)):
            if spans and count(t[spans[-1][0]:b]) <= size:
                spans[-1] = (spans[-1][0], b)
            else:
                spans.append((a, b) if count(t[a:b]) <= size else (a, b))
        out = []
        for a, b in spans:  # a single long sentence is split recursively
            out += _recursive(t, a, b, size, count) if count(t[a:b]) > size else [(a, b)]
        return [_chunk(doc, a, b, strategy) for a, b in out]
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs embed_fn")
        sents = _sentences(t, 0, len(t))
        if len(sents) < 2:
            return [_chunk(doc, 0, len(t), strategy)]
        v = embed_fn([t[a:b] for a, b in sents])
        sim = (v[:-1] * v[1:]).sum(1)
        cut = np.percentile(sim, semantic_percentile)
        spans = [sents[0]]
        for (a, b), s in zip(sents[1:], sim):
            if s >= cut and count(t[spans[-1][0]:b]) <= size:
                spans[-1] = (spans[-1][0], b)
            else:
                spans.append((a, b))
        out = []
        for a, b in spans:
            out += _recursive(t, a, b, size, count) if count(t[a:b]) > size else [(a, b)]
        return [_chunk(doc, a, b, strategy) for a, b in out]
    if strategy == "parent_child":
        psize, out = parent_size or 4 * size, []
        for s in doc.sections:
            for pa, pb in _recursive(t, s.start, s.end, psize, count):
                pid = _cid(doc, pa, pb, "parent")
                for a, b in _recursive(t, pa, pb, size, count):
                    out.append(_chunk(doc, a, b, strategy, header=True, ctx=(pa, pb), parent_id=pid))
        return out
    raise ValueError(f"unknown chunking strategy {strategy!r}; one of {STRATEGIES}")
