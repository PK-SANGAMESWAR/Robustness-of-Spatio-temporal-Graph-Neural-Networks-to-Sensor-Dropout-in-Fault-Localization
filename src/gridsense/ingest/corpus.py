"""Load the GridSense corpus (O7, 2026-10-31) as parsed documents with metadata and a content fingerprint.

Parts (rag-for-dropout/corpus/):
  v1/          model cards: Paper 1 rev 1170 and RESULTS.md (frozen; sha256 in MANIFEST.json)
  feeder_v1/   generated feeder documents (gridsense.ingest.feeder_docs; sha256 in MANIFEST.json)
  notes_v1/    curated notes (front matter; status draft until the author reviews them)
  sop_v1/      one illustrative procedure (approval_status: illustrative)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gridsense.config import TRACK
from gridsense.ingest.parsers import PARSER_VERSION, Document, parse_latex, parse_markdown, split_frontmatter

CORPUS = TRACK / "corpus"
PARTS = ("v1", "feeder_v1", "notes_v1", "sop_v1")


class CorpusIntegrityError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _manifest_docs(part: Path) -> list[dict]:
    man = json.loads((part / "MANIFEST.json").read_text())
    for d in man["documents"]:
        raw = (part / d["file"]).read_bytes()
        if _sha(raw) != d["sha256"]:
            raise CorpusIntegrityError(f"{part.name}/{d['file']}: sha256 differs from MANIFEST.json")
    return man["documents"]


def _md_doc(path: Path, meta: dict, defaults: dict) -> Document:
    front, body = split_frontmatter(path.read_text(encoding="utf-8"))
    m = {**defaults, **meta, **front}
    text, sections = parse_markdown(body)
    title = m.get("title") or (sections[0].path[0] if sections and sections[0].path else path.stem)
    return Document(document_id=m["document_id"], title=title, document_type=m["document_type"],
                    approval_status=m["approval_status"], version=str(m.get("version_tag") or m.get("version", "0")),
                    status=m["status"], text=text, sections=sections, source_file=str(path.relative_to(TRACK)))


def load_corpus(parts: tuple[str, ...] = PARTS, root: Path = CORPUS) -> list[Document]:
    docs: list[Document] = []
    for name in parts:
        part = root / name
        if name == "v1":
            for d in _manifest_docs(part):
                p = part / d["file"]
                base = dict(document_id=d["document_id"], document_type="model_card", approval_status="curated_for_research",
                            version_tag=d["version_tag"], status="frozen")
                if p.suffix == ".tex":
                    text, sections = parse_latex(p.read_text(encoding="utf-8"))
                    docs.append(Document(d["document_id"], "Paper 1 (revision 1170)", "model_card", "curated_for_research",
                                         d["version_tag"], "frozen", text, sections, str(p.relative_to(TRACK))))
                else:
                    docs.append(_md_doc(p, base, {}))
        elif name == "feeder_v1":
            for d in _manifest_docs(part):
                docs.append(_md_doc(part / d["file"], {**d, "status": "generated", "version": name}, {}))
        else:
            for p in sorted(part.glob("*.md")):
                docs.append(_md_doc(p, {}, {"status": "draft"}))
    ids = [d.document_id for d in docs]
    if len(ids) != len(set(ids)):
        raise CorpusIntegrityError(f"duplicate document_id in corpus: {ids}")
    return docs


def corpus_fingerprint(docs: list[Document]) -> str:
    body = [(d.document_id, d.version, d.status, _sha(d.text.encode())) for d in sorted(docs, key=lambda d: d.document_id)]
    return _sha(json.dumps({"parser": PARSER_VERSION, "docs": body}, sort_keys=True).encode())
