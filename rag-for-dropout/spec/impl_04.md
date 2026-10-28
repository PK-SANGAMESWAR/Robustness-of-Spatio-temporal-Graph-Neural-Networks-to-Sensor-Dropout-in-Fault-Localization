<!-- Source: implementation-plan.md lines 431-530, verbatim. Do not edit. Index: 00_INDEX.md -->

## 5. Document ingestion and preprocessing

The RAG system is only as good as the documents it ingests. Ingestion
must preserve the source text, section structure, and provenance.

### 5.1 Approved source types

Initially allow a controlled set of sources: - Approved grid operating
manuals. - Approved standard operating procedures. - Engineering
guidelines and relevant technical documents. - Internal troubleshooting
documents that have been explicitly approved for use.

Do not ingest arbitrary web pages or unreviewed files into a
safety-sensitive knowledge base without source approval.

**[Rev 2] What the corpus realistically is for this project.** The
STGNN is trained on a simulated feeder; no utility has issued approved
SOPs for it. Do not label anything `approved` that no one approved. Use:

-   Feeder documentation generated from the simulator model: sections,
    lines, buses, phases, µPMU placement, protection devices. This is
    exact and also feeds metadata filters.
-   Curated technical notes written for the project (fault types,
    observability under sensor loss, what an ambiguity set means), each
    with a source reference to the textbook, standard or paper it
    summarizes. Keep it to roughly 30--100 well-written chunks.
-   An example operator procedure, clearly marked as illustrative.

Set `approval_status` to `curated_for_research` (or `illustrative`) and
state in every report that no utility-approved procedure was used. The
approval machinery in this plan stays in place, so swapping in a real
utility's documents later needs no redesign.

### 5.2 Ingestion pipeline

1.  Register a source document.
2.  Assign a stable `document_id`.
3.  Record filename, title, document type, revision, approval status,
    source owner, and effective date where available.
4.  Extract text from PDF, DOCX, or approved HTML.
5.  Preserve page numbers, section headings, tables, warnings, and
    references where feasible.
6.  Detect empty pages, corrupted extraction, duplicated content, and
    OCR-required scans.
7.  Normalize whitespace and obvious extraction artifacts without
    changing technical meaning.
8.  Split text into chunks while retaining section context.
9.  Generate embeddings.
10. Store chunks, vectors, and metadata.
11. Run ingestion checks and produce an audit record.

### 5.3 Metadata schema

Recommended fields per chunk:

``` json
{
  "chunk_id": "stable-id",
  "document_id": "grid-sop-001",
  "document_title": "Approved feeder operating procedure",
  "document_revision": "revision-if-known",
  "approval_status": "approved",
  "effective_date": "date-if-known",
  "document_type": "SOP",
  "section_path": ["Section 4", "Fault isolation"],
  "page_start": 12,
  "page_end": 13,
  "chunk_index": 7,
  "text": "Extracted chunk text...",
  "content_hash": "sha256-of-normalized-source-or-chunk",
  "embedding_model": "exact-model-revision",
  "ingestion_version": "v1"
}
```

Use `null` or omit fields that are genuinely unknown. Never invent a
revision, page number, or approval status. Only documents whose approval
status has been verified should be eligible for operational guidance
retrieval.

### 5.4 PDF and table handling

Grid procedures can contain tables, numbered steps, warnings, and
cross-references. Plain-text extraction can destroy their meaning.

Recommended handling: - Extract text with page boundaries. - Preserve
headings and numbered lists. - Convert simple tables into readable text
with headers repeated for each row group. - Retain page/section
references. - Flag scanned PDFs for OCR rather than silently indexing
empty text. - Manually inspect a sample of extracted documents. - For
complex electrical diagrams, keep the original page reference and route
the user to the source; do not assume text extraction captures the
diagram's full meaning.

Useful Python tools include PyMuPDF or pdfplumber for PDFs and
python-docx for DOCX files. Choose one primary parser per file type and
add fallbacks only where evaluation demonstrates a need.

------------------------------------------------------------------------

