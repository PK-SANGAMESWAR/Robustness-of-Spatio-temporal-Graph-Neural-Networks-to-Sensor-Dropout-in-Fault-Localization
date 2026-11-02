"""Row-level table chunks (plan section 5.4): every row is its own chunk with the table header repeated; text between tables is chunked as before."""
from __future__ import annotations

import pytest

from gridsense.eval import golden as G
from gridsense.ingest.chunking import chunk_document, markdown_tables
from gridsense.ingest.parsers import Document, parse_markdown

MD = """# Doc

Intro text before the table.

## Fault buses

| fault bus | nearest (km, hops) |
|---|---|
| 51 | bus 47 (0.229 km, 3) |
| 93 | bus 91 (0.069 km, 1) |

Text after the table. It has a second sentence.
"""


def doc(text=MD):
    t, secs = parse_markdown(text)
    return Document("d", "Doc title", "feeder_documentation", "curated_for_research", "0", "generated", t, secs)


def words(s):
    return len(s.split())


def test_markdown_tables_are_found_with_header_and_rows():
    d = doc()
    (tb,) = markdown_tables(d.text)
    assert tb["header"] == "| fault bus | nearest (km, hops) |" and [r[2] for r in tb["rows"]] == ["| 51 | bus 47 (0.229 km, 3) |", "| 93 | bus 91 (0.069 km, 1) |"]
    assert markdown_tables("no table here | a | b\nanother line") == []


def test_each_row_is_one_chunk_with_the_header_repeated_and_the_golden_row_is_inside():
    d = doc()
    ch = chunk_document(d, "recursive", 50, words, 0.0, table_rows=True)
    rows = [c for c in ch if c.text.startswith("| fault bus |")]
    assert len(rows) == 2 and all(c.text.count("| fault bus | nearest (km, hops) |") == 1 for c in rows)
    assert rows[0].text.endswith("| 51 | bus 47 (0.229 km, 3) |") and rows[1].text.endswith("| 93 | bus 91 (0.069 km, 1) |")
    assert "Doc title" in rows[0].embed_text and "Fault buses" in rows[0].embed_text and rows[0].section_path[-1] == "Fault buses"
    span = {"document_id": "d", "span": "| 51 | bus 47 (0.229 km, 3) |"}
    assert sum(G.matches(c.document_id, c.text, span) for c in ch) == 1                 # the bare golden row is in exactly one chunk
    assert any("Text after the table" in c.text for c in ch) and any("Intro text" in c.text for c in ch)


def test_without_tables_the_result_is_the_same_as_without_the_option():
    d = doc("# T\n\n" + " ".join(f"w{i}" for i in range(120)) + ". End.\n")
    a = chunk_document(d, "recursive", 30, words, 0.1)
    b = chunk_document(d, "recursive", 30, words, 0.1, table_rows=True)
    assert [(c.start, c.end, c.text) for c in a] == [(c.start, c.end, c.text) for c in b]


def test_table_rows_is_limited_to_the_recursive_chunker():
    with pytest.raises(NotImplementedError):
        chunk_document(doc(), "sentence", 50, words, 0.0, table_rows=True)


def test_fingerprint_changes_only_when_table_rows_is_on():
    from gridsense.retrieval.retriever import RetrievalConfig
    a, b = RetrievalConfig(chunker="recursive"), RetrievalConfig(chunker="recursive", table_rows=True)
    assert "table_rows" not in a.index_part() and b.index_part()["table_rows"] is True
