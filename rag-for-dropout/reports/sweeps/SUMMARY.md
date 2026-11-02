# Layer 1 sweeps — summary and the final retrieval configuration (stage S3)

Written for: the author. Date 2026-11-02. Golden set v1 (56 reviewed queries), corpus `s3_a1`. All choices were made on the **dev split** (25 answerable and 4 unanswerable queries); the
**test split** was used once per frozen configuration. Sources: `WINNERS.json`, the per-axis reports in this folder, `FROZEN_TEST.md`, snapshots in `eval/snapshots/`. CPU only, 2 threads.

## 1. The final configuration (frozen)
| setting | value | why (report) |
|---|---|---|
| embedder | `all-MiniLM-L6-v2` (free, local) | best quality; OpenAI `text-embedding-3-small` scored lower (`embedder.md`) |
| chunker, size, overlap | recursive, 200 tokens, 10 % overlap | `chunker.md`, `chunk_size.md`, `overlap.md` |
| retrieval | hybrid: dense plus BM25, fused with RRF (k = 60), 20 candidates from each | `retrieval_mode.md`, `candidates.md` |
| reranker | none | the cross-encoder gave no gain (`reranker.md`) |
| top_k | 3 passages | recall at 3, 5, 8 and 10 is the same on dev (`top_k.md`) |
| vector store | **Qdrant (local mode)**, not FAISS | `vector_store.md` |

**Test result (one run, 23 answerable queries; 95 % intervals over queries):** Recall@1 0.74 (0.57–0.91), Recall@5 0.91 (0.78–1.00), MRR@10 0.79 (0.63–0.92), nDCG@5 0.75, Precision@3 0.29, Precision@5 0.21.
Dev numbers are higher (Recall@1 0.80, MRR@10 0.860) because the configuration was chosen on dev: use the test numbers. Chroma gave exactly the same retrieved lists and metrics as Qdrant on test.

## 2. Your questions
**Do chunks of 200 tokens split the answer spans? No.** With the final chunking 0 of 31 dev contexts and 0 of 28 test contexts are split, and no chunker split any dev span at 200 tokens.
A split only happened in the first (rule v1) run with the `fixed_token` chunker at 100 and 150 tokens (4 and 3 spans: `rule_v1/chunk_size.md`). The check is `reachability()`; a unit test shows that it does detect a split.

**Why was Recall@1 low, and what raised it?** The dense baseline had Recall@1 0.60 on dev, and dense-only gets 0.48 on test. Reading the five dev misses of the final configuration by hand (section 4) shows ranking confusions
between similar passages, not lost spans. Hybrid retrieval raised Recall@1 from 0.72 to 0.80 on dev and, on test, by **+0.261 (95 % interval +0.087 to +0.435)**, MRR@10 by +0.145 (+0.045 to +0.255), paired over 23 queries.

**No-answer threshold (fitted on dev only).** tau = 0.490 on the top-1 dense similarity. It flags 4 of 4 unanswerable queries on dev and on test. It also flags 22 % of the answerable test queries (5 of 23). So the flag is right in
4 of 9 cases (0.44): the unanswerable similarities (0.40–0.47 on dev) lie inside the answerable range (from 0.39). With n = 4 unanswerable queries per split, treat tau as a rough guide. bge-small flags fewer answerable queries
(8 %) at the same 100 % on dev, but retrieves worse; a separate embedder for the abstention signal is an open option (not done).

**Precision.** Precision@k cannot exceed (golden contexts of the query) / k, and most queries have one or two, so P@5 can be at most 0.2 to 0.4 (0.21 on test is near that cap). Read Precision@1 (= Recall@1) and MRR:
0.74 and 0.79 on test. LLM-judged contextual precision (DeepEval) is stage S4 (paid; needs the generator and judge models, O1).

**Vector store (your instruction: not FAISS, a good store, precision kept).** FAISS-flat is only the exact reference (a library: no metadata filter, no payload store). Chroma (default and high-recall HNSW) and Qdrant were run through
the real hybrid retriever: **identical top-10 lists for all 29 dev queries and identical precision, recall and MRR** (max metric difference 0). Qdrant won on speed (p95 0.14 ms against 0.48 to 0.79 ms) and memory (100 against 108 MB) and the plan names
it as the preferred store for hybrid search with filters. Limits: the corpus is small, so HNSW cannot show an approximation loss; re-run `store_bench` when the corpus grows. Switching to Chroma is one setting and changes no result.

## 3. All axes (quality Q = mean of MRR@10, Recall@5 and nDCG@5 on dev; winner = best quality, then cheapest within 0.013)
| axis | candidates (Q) | winner |
|---|---|---|
| chunker | parent_child 0.832, recursive 0.821, sentence 0.820, structure_aware 0.814, fixed_token 0.811, semantic 0.652 | recursive (parent_child is best but needs 1,468 context tokens against 643) |
| chunk size | 200: 0.821, 150: 0.736, 100: 0.670 | 200 |
| overlap | 10 %: 0.848, 15 %: 0.839, 0: 0.821 | 10 % |
| embedder | MiniLM 0.848, e5-small 0.806, bge-small 0.790, OpenAI 0.769 | MiniLM |
| retrieval mode | hybrid (k_rrf 20, 60, 100) 0.882, dense 0.848, BM25 0.787 | hybrid, k_rrf 60 (the three values tie exactly) |
| reranker | off 0.882, cross-encoder 0.859 | off |
| candidates | 20 and 50 tie | 20 |
| top_k | 3, 5 and 8 tie in recall | 3 (fewest tokens: 438 against 723 at k = 5, dev) |
| vector store | Qdrant, Chroma (both HNSW settings): identical results | Qdrant |

## 4. The five dev misses of the final configuration, read by hand
| query | first hit | what happened |
|---|---|---|
| R-0003 sequence networks, two phases and earth | rank 2 | sibling note: the single-line-to-ground note ranked first |
| R-0016 targeted against random loss | rank 2 | **valid alternative evidence:** note_06 also says that losing the nearest sensors hurts more; the golden item names note_16. Not changed (no change because of rank) |
| R-0033 class index of fault bus 57 | rank 2 | the heading chunk of the same document ranked first; the answer is a bare table row without column labels |
| R-0021 which signal separates right from wrong predictions | >10 | vocabulary gap: the note says "margin" and "AUROC", the question says "signal" |
| R-0042 which settings were frozen | >10 | the word "frozen" occurs in several RESULTS.md and paper passages; the answer sits in the file's header block |
Two ideas for later, not done: row-level chunks for tables with the header repeated (the plan, section 5.4, asks for it); an abstention signal from a better-separating model.

## 5. Selection rule and what changed (disclosure)
The first run of three axes used **rule v1** (the cheapest candidate inside the 95 % interval of the best). You asked for best plus cheap, so I replaced it by **rule v2**: highest quality first; only a candidate within one query on one metric
(0.013) of the best counts as equal; then the cheapest. The change was made after the dev results of those three axes had been seen. Rule v1 results are in `rule_v1/` (it would have chosen `fixed_token`, 150 tokens and 10 % overlap). All axes were rerun under v2.
Vector store rule: a real store, not the FAISS reference, native filter works, identical to exact search end to end (your instruction).

## 6. Limits and open points
- 25 dev and 23 test answerable queries: the intervals are wide; the test numbers are lower than dev, as expected after selection on dev. Differences between candidates inside 0.013 are ties by construction.
- **11 notes and the SOP are still `draft`** (your review did not mention them); 12 of the 37 corpus documents. Golden items that point to them are reviewed (O34).
- **The OpenAI embedding call sent the text of all 37 corpus documents, including the local-only revision manuscript, to OpenAI.** It cost 0.000626 USD (31,287 tokens; estimate equal to the cost); ledger total 0.00063 USD of 5 USD.
- Golden contexts name one answering span; other passages that also answer (R-0016) count as misses. Several queries use bare table rows as spans (R-0014, R-0015, R-0016, R-0041): accepted in your review; they suit the chunker only if the header sits in the same chunk.
- Snapshots: the first chunker run crashed after writing its snapshots, and rule v1 left more; they are kept (append-only).
- Not done: LLM-judged contextual precision and recall, generator and judge choice (S4, paid, O1); the case store keeps FAISS (it is frozen and part of the gate).
