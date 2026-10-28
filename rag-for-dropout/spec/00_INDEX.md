# spec/ — index of the frozen GridSense source documents

The two source documents are frozen. The files in this folder are **verbatim** copies of them, split into parts below
200 lines. Each part starts with a 2-line header. `tests/gridsense/test_docs.py` rebuilds each source file from its parts
and compares the bytes. Do not edit these files. Write corrections in `../PROPOSED_CHANGES.md`.

Source files:
- `../implementation-plan.md` — GridSense implementation plan, revision 2 (2026-10-28), 1,823 lines.
- `../rag-eval.md` — end-to-end evaluation prompt (ASD-STE100), 416 lines.
- `../RAG-arch.png` — architecture figure. Its "0.846 macro-F1" label is wrong for the dropout-aware model (see
  `../PROPOSED_CHANGES.md` P1).

## implementation-plan.md
| Part | Lines | Sections |
|---|---|---|
| `impl_01.md` | 1–127 | Title, §0 revision-2 review notes (R1–R7), §1 executive summary |
| `impl_02.md` | 128–283 | §2 goals, non-goals, acceptance criteria; §3 architecture, component responsibilities |
| `impl_03.md` | 284–430 | §4 embeddings: definition, cosine, model selection, input-length limit, prefixes, baseline code |
| `impl_04.md` | 431–530 | §5 document ingestion: sources, pipeline, metadata schema, PDF/table handling |
| `impl_05.md` | 531–673 | §6 chunking strategies and size experiment, parent-child; §7 vector store (Chroma vs Qdrant, collections, IDs) |
| `impl_06.md` | 674–840 | §8 dense, lexical, hybrid retrieval, query construction, filters, §8.6 latent-kNN case retrieval; §9 reranking |
| `impl_07.md` | 841–1014 | §10 generation, prompt, report schema, citation validation, insufficient evidence; §11 deterministic gate |
| `impl_08.md` | 1015–1192 | §12 LangGraph states, fields, retries, human-in-the-loop; §13.1–13.6 retrieval/gate/report evaluation |
| `impl_09.md` | 1193–1336 | §13.7–13.13 splits, case-retrieval metrics, leakage, rubber-stamp tests, uncertainty, ablation A0–A4, judges |
| `impl_10.md` | 1337–1511 | §14 CI and regression tests; §15 repository structure; §16 API boundaries |
| `impl_11.md` | 1512–1655 | §17 security and audit; §18 T1000 performance; §19 milestones 0–7 |
| `impl_12.md` | 1656–1824 | §20 experiment matrix; §21 definition of done; §22 order of work; §23 references; final recommendation |

## rag-eval.md
| Part | Lines | Sections |
|---|---|---|
| `eval_01.md` | 1–63 | §1 task, §2 rules for all stages, §3 stage 1 repository inspection |
| `eval_02.md` | 64–177 | §4 evaluation layers 0–6 |
| `eval_03.md` | 178–321 | §5 golden datasets, §6 judge and generation configuration, §7 configuration, §8 snapshots and registry |
| `eval_04.md` | 322–417 | §9 baseline vs candidate, §10 tests and cost control, §11 documentation, §12 stages, §13 first response |

## Precedence (when two texts disagree)
1. An author decision recorded in `../claude-rag.md` § Author decisions.
2. An accepted entry in `../PROPOSED_CHANGES.md` (status `accepted`).
3. `rag-eval.md` for evaluation; `implementation-plan.md` for system design.
4. If the conflict is not covered, stop and ask the author.
