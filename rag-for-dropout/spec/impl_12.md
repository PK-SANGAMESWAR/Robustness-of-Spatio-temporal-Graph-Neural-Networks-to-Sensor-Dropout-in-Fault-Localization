<!-- Source: implementation-plan.md lines 1656-1824, verbatim. Do not edit. Index: 00_INDEX.md -->

## 20. First experiment matrix

Do not launch every experiment at once. Use a small controlled matrix.

  ----------------------------------------------------------------------------
  Dimension               Initial candidates           Evaluation
  ----------------------- ---------------------------- -----------------------
  Embedding model         MiniLM baseline + 1--2       Retrieval quality,
                          suitable small alternatives  latency, memory

  Chunk size              Small / medium / large       Recall@k, MRR, nDCG,
                                                       source integrity

  Chunking                Recursive/structure-aware;   Retrieval and context
                          parent-child if needed       preservation

  Retrieval               Dense; lexical; hybrid RRF   Ranking quality and
                                                       exact-identifier recall

  Reranking               Off; one small reranker      Quality gain versus
                                                       latency

  Top-k                   Several candidate counts     Recall versus context
                                                       size and latency

  Confidence gate         Several validation-selected  Coverage, selective
                          thresholds                   risk, false acceptance

  Sensor gate             Availability +               Error detection and
                          critical-pattern policy      rejection rate
                          variants                     

  Report generation       Fixed prompts/schema         Citation support,
                          variants                     completeness,
                                                       abstention

  Runtime                 CPU baseline; selective GPU  Latency, RAM, VRAM
                          use                          

  [Rev 2] Case store k    5 / 10 / 20 neighbours       kNN-vote accuracy,
                                                       error-detection AUROC

  [Rev 2] Latent layer    Penultimate; last GNN layer  Error-detection AUROC
                          (pooled)                     

  [Rev 2] Gate signal     Probability only; + margin;  AURC, selective risk
                          + kNN agreement; combined    at fixed coverage

  [Rev 2] Candidate set   Top-k; conformal (90%)       Coverage, mean set size

  [Rev 2] Dropout level   0 / 10 / 20 / 30 / 40%       All of the above,
                                                       per level
  ----------------------------------------------------------------------------

Keep the final test set untouched until the design has been selected.
Record every configuration so results are reproducible.

------------------------------------------------------------------------

## 21. Definition of done

GridSense v1 is complete when:

-   [ ] The existing STGNN is callable through a documented interface.
-   [ ] Predictions include model version and the necessary
    sensor-availability metadata.
-   [ ] Input validation and deterministic verification gates are
    implemented.
-   [ ] Calibration and selective-risk evaluations are documented.
-   [ ] Approved documents can be ingested with page/section provenance.
-   [ ] Embedding model and chunking configuration are versioned.
-   [ ] Dense retrieval is benchmarked against labelled queries.
-   [ ] Lexical/hybrid retrieval is evaluated before a final choice is
    made.
-   [ ] Reranking is included only if its measured gain justifies its
    cost.
-   [ ] Reports have a validated schema and source references.
-   [ ] Insufficient evidence leads to abstention/manual review.
-   [ ] The LLM cannot override verification or human approval.
-   [ ] Human approvals and rejections are auditable.
-   [ ] CI tests gate behavior, retrieval, citations, and workflow
    invariants.
-   [ ] Documentation includes reproducible commands and limitations.
-   [ ] No component directly operates grid equipment.
-   [ ] **[Rev 2]** Case store built from the training split, versioned
    by checkpoint, with leakage assertions passing.
-   [ ] **[Rev 2]** Error-detection AUROC reported for probability,
    margin, kNN agreement and the combined score.
-   [ ] **[Rev 2]** Ablation ladder A0--A4 reported at every dropout
    level.
-   [ ] **[Rev 2]** Rubber-stamp, evidence-flip, empty-evidence and
    injection tests pass, with pass rates over ≥ 3 generation runs.
-   [ ] **[Rev 2]** Any LLM judge validated against human labels.
-   [ ] **[Rev 2]** Every report states that the corpus is
    research-curated, not utility-approved.

------------------------------------------------------------------------

## 22. Recommended order of work

If time is limited, use this order:

1.  Audit and stabilize the existing STGNN inference interface.
1a. **[Rev 2]** Extract latents, build the case store, and measure
    kNN-vote accuracy and error-detection AUROC (§13.8). This is the
    cheapest experiment and decides whether case evidence is worth
    building around. If kNN agreement adds nothing over calibrated
    probability, simplify the gate and say so.
2.  Define the event, prediction, verification, and report schemas.
3.  Build approved-document ingestion and dense retrieval.
4.  Create the labelled retrieval benchmark before tuning embeddings or
    chunking.
5.  Add lexical/hybrid retrieval and evaluate whether reranking is
    worthwhile.
6.  Implement calibrated confidence and sensor-availability gates.
7.  Add LangGraph orchestration and structured report generation.
8.  Add citation validation and human approval.
9.  Add CI regression tests and end-to-end evaluation.
10. Build the demo UI and polish documentation.

Do not start by building a multi-agent system. First make the prediction
interface, retrieval quality, verification policy, and audit trail
reliable. A single well-controlled report agent is enough for version
one.

------------------------------------------------------------------------

## 23. Official documentation and further reading

Use the current documentation and model cards at implementation time
because APIs and model versions can change.

-   Sentence Transformers quickstart:
    https://sbert.net/docs/quickstart.html
-   Sentence Transformers semantic textual similarity:
    https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
-   Sentence Transformers semantic search:
    https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
-   Hugging Face overview of Sentence Transformers:
    https://huggingface.co/docs/hub/sentence-transformers
-   Qdrant documentation: https://qdrant.tech/documentation/
-   Qdrant hybrid search:
    https://qdrant.tech/documentation/search-tuning/hybrid-search/
-   Qdrant hybrid and multi-stage queries:
    https://qdrant.tech/documentation/hybrid-queries/
-   scikit-learn probability calibration:
    https://scikit-learn.org/stable/modules/calibration.html
-   LangGraph documentation:
    https://docs.langchain.com/oss/python/langgraph/overview
-   FastAPI documentation: https://fastapi.tiangolo.com/
-   pytest documentation: https://docs.pytest.org/
-   GitHub Actions documentation: https://docs.github.com/actions

------------------------------------------------------------------------

## Final recommendation

Build GridSense as a **verified decision-support pipeline**, not as an
LLM that decides whether to trust another model. Embeddings and RAG
should retrieve traceable, approved evidence; the STGNN should predict;
deterministic policy code should decide whether the event is eligible
for report drafting; LangGraph should orchestrate the workflow; and a
human should approve the final report.

The highest-value early deliverable is a reproducible baseline
consisting of **STGNN inference + deterministic verification + evaluated
dense retrieval with source provenance**. Add hybrid retrieval,
reranking, and more elaborate agent behavior only when evaluation
demonstrates that each addition improves the system.