<!-- Source: implementation-plan.md lines 531-673, verbatim. Do not edit. Index: 00_INDEX.md -->

## 6. Chunking strategy: do not choose chunk size by intuition alone

Chunking controls what evidence is retrievable. A chunk that is too
short may lose conditions or definitions; a chunk that is too long may
mix unrelated procedures and reduce retrieval precision.

### 6.1 Strategies to compare

  -----------------------------------------------------------------------
  Strategy          Description       Strength          Risk
  ----------------- ----------------- ----------------- -----------------
  Fixed             Split at a        Simple,           May split a step
  character/token   configured length reproducible      or condition
  size                                baseline          

  Recursive         Prefer            Strong            Still not aware
  separators        paragraphs,       general-purpose   of technical
                    lines, and        baseline          meaning
                    sentence                            
                    boundaries                          

  Structure-aware   Split by          Preserves         Requires reliable
                    headings,         procedure context parsing
                    sections,                           
                    numbered steps,                     
                    and tables                          

  Parent-child      Index small child Precise matching  More metadata and
                    chunks but return with context      retrieval logic
                    the larger parent                   
                    section                             

  Sentence-based    Group complete    Keeps local       Can detach a
                    sentences         meaning           sentence from its
                                                        section

  Semantic chunking Split when        Can preserve      More expensive
                    topic/meaning     coherent topics   and may be
                    changes                             unstable
  -----------------------------------------------------------------------

### 6.2 Recommended starting experiment

Begin with recursive/structure-aware chunking. Compare at least three
configurations, for example: - Small: approximately 250--350 tokens. -
Medium: approximately 450--650 tokens. - Large: approximately 750--1,000
tokens.

**[Rev 2]** These ranges are only valid for an embedding model whose
input window covers them (see §4.4). With MiniLM, run the size
comparison on *parent* sections and keep the embedded child chunks at
≤ 256 tokens.

These are experimental ranges, not universal best settings. Keep a small
overlap, such as 10--15%, only where needed to preserve cross-boundary
context. Do not split a numbered safety procedure into independent
fragments if the preconditions and warnings are in adjacent steps.

For each configuration: 1. Rebuild the index. 2. Run the same labelled
query set. 3. Measure Recall@k, MRR, nDCG@k, citation/source coverage,
and latency. 4. Inspect failure cases manually. 5. Select the
configuration that gives the best evidence retrieval quality for
acceptable cost.

### 6.3 Parent-child recommendation

For long manuals, a useful pattern is: - Store smaller child chunks for
precise matching. - Store a parent section or step group for context. -
Search child chunks. - Return the parent section or child plus adjacent
context to the generation stage. - Preserve the child hit's source
offsets and the parent's page/section metadata.

This helps avoid giving the LLM a fragment that omits an important
prerequisite or warning.

------------------------------------------------------------------------

## 7. Vector store design

Use one vector database initially. Given the user's existing exposure to
Qdrant and Chroma, either can support a prototype.

### 7.1 Chroma versus Qdrant

  -----------------------------------------------------------------------
  Consideration           Chroma                  Qdrant
  ----------------------- ----------------------- -----------------------
  Local prototype         Straightforward         Supports local
                                                  deployment

  Metadata filtering      Supported               Strong payload
                                                  filtering and indexes

  Hybrid retrieval        Depends on surrounding  Dense, sparse, and
                          implementation          hybrid query patterns
                                                  supported

  Operational scaling     Suitable for many       Designed as a
                          prototypes              vector-search service
                                                  with advanced indexing
                                                  options

  Recommendation          Good for a minimal      Preferred if hybrid
                          local demo              search, filtering, and
                                                  a service boundary are
                                                  planned
  -----------------------------------------------------------------------

Choose Qdrant if the plan is to build dense plus lexical/sparse
retrieval, use metadata filters, and keep retrieval as a separately
testable service. Chroma remains a valid choice for a simple first proof
of concept.

### 7.2 Collection design

Use a collection name that includes the corpus and embedding version,
for example:

`gridsense_manuals_minilm_v1`

Keep a collection-level manifest with: - Embedding model and revision. -
Dimension and distance metric. - Chunking strategy and parameters. -
Parser version. - Corpus version. - Index build timestamp. - Evaluation
result for the build.

Create payload indexes for frequently filtered metadata fields such as
`approval_status`, `document_type`, and `document_id`. Always filter to
approved sources for operational guidance.

### 7.3 Stable identifiers and updates

-   Generate deterministic chunk IDs from document ID, revision,
    section, chunk index, and content hash.
-   When a document changes, ingest the new revision as a new version.
-   Do not silently overwrite approved content without an audit trail.
-   Mark obsolete documents as superseded and exclude them by default.
-   Make ingestion idempotent: rerunning it should not create duplicate
    chunks.
-   Store the exact source text and provenance needed to reconstruct
    citations.

------------------------------------------------------------------------

