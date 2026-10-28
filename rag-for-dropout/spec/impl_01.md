<!-- Source: implementation-plan.md lines 1-127, verbatim. Do not edit. Index: 00_INDEX.md -->

# GridSense Implementation Plan

## End-to-end plan for embeddings, RAG, verification, agent orchestration, evaluation, and deployment

**Status:** Proposed implementation plan (revision 2, 2026-10-28)\
**Project:** GridSense --- from STGNN prediction to verified,
human-approved decision support\
**Primary goal:** Reuse the existing dropout-aware STGNN and build a
reliable, testable system around it.\
**Important boundary:** This is a research/decision-support prototype.
It must not directly operate grid equipment or replace approved
engineering procedures.

------------------------------------------------------------------------

## 0. Revision 2 --- review notes (2026-10-28)

This revision corrects the following problems in revision 1 and adds a
full end-to-end evaluation protocol. Changed or new sections are marked
**[Rev 2]**.

  -------------------------------------------------------------------------
  \#   Issue in rev 1                     Fix                     Section
  ---- ---------------------------------- ----------------------- ---------
  R1   The RAG corpus contained only      Add a second retrieval  §8.6
       manuals/SOPs. Manuals can say      path: case-based
       *what to do*, but nothing in the   retrieval over the
       corpus can say *whether this       STGNN's own latent
       prediction is likely right*.       embeddings (latent
                                          kNN).

  R2   `all-MiniLM-L6-v2` truncates input Child chunks ≤ 256       §4.4,
       at 256 word pieces (and was        tokens for MiniLM, or a §6.2
       trained on ≤ 128). The proposed    longer-context model
       450--1,000-token chunks would be   for medium/large
       silently cut, so the chunk-size    chunks.
       experiment would not measure what
       it claims.

  R3   A failed gate sent the case to a   Low confidence is a     §3, §12.1
       manual-review queue with no        separate route that
       report. The operator then gets the still drafts an
       hardest cases with the least       "uncertain" report
       context.                           with a candidate set.
                                          Only invalid input and
                                          failed checks go
                                          straight to review.

  R4   Sensor checks used only the        Add the LOO physics     §11.1
       availability mask. A sensor that   residual check (Stage
       is present but wrong (silent       F) as a gate input
       failure, m = 1) is invisible to    when available.
       every mask-based check.

  R5   "Approved manuals/SOPs" do not     State the corpus        §5.1
       exist for a simulated feeder in a  honestly: curated
       student project. Claiming          technical notes and
       "approved" status would be         simulator/feeder
       fabricated provenance.             documentation, with
                                          `approval_status` =
                                          `curated_for_research`.

  R6   Evaluation covered layers, but not Add end-to-end          §13.7--
       whether the *system* helps: does   protocol: error         §13.13
       the report flag the cases where    detection, rubber-stamp
       the STGNN is wrong?                tests, ablation ladder,
                                          dropout sweep, leakage
                                          controls, judge
                                          validation.

  R7   Embedding-model notes did not      Prefix note added.      §4.5
       mention query/passage prefixes
       required by e5/bge-family models.
  -------------------------------------------------------------------------

Everything else in revision 1 was checked and is kept as written.

------------------------------------------------------------------------

## 1. Executive summary

GridSense combines five capabilities:

1.  **Fault prediction:** an existing PyTorch spatio-temporal GNN
    predicts a candidate fault location from time-series grid
    measurements.
2.  **Deterministic verification:** a non-LLM service checks input
    validity, calibrated confidence, sensor availability, and policy
    rules.
3.  **Evidence retrieval:** an embedding-based retrieval pipeline finds
    relevant passages from approved grid manuals and standard operating
    procedures (SOPs).
4.  **Report drafting:** a LangGraph workflow creates an
    evidence-grounded report for an operator.
5.  **Evaluation and CI:** automated tests check model regressions,
    verification behavior, retrieval quality, citation correctness, and
    approval enforcement.

The central design principle is to **separate prediction, verification,
explanation, and authorization**. An LLM must never be allowed to change
a failed verification decision or authorize a physical grid operation.

### Recommended first version

-   Keep the existing STGNN unchanged behind a stable inference
    interface.
-   Start with a small local embedding model such as
    `sentence-transformers/all-MiniLM-L6-v2` as a baseline, not as an
    assumed final winner.
-   Use Qdrant or Chroma for vector retrieval; select one initially
    rather than maintaining both.
-   Implement dense semantic retrieval first, then compare it with
    lexical/BM25 and hybrid retrieval.
-   Add a reranker only if a measured retrieval evaluation shows that it
    improves the relevant top results.
-   Use deterministic Python code for confidence and sensor gates.
-   Use LangGraph for explicit workflow state, retries, routing, and
    human review.
-   Add pytest tests and a CI workflow before calling the prototype
    complete.

The correct embedding model, chunk size, retrieval method, confidence
threshold, and sensor-coverage policy must be selected through
experiments on a representative evaluation set---not by guessing.

------------------------------------------------------------------------

