# Layer 1 sweeps — summary and the final retrieval configuration (stage S3)

Written for: the author. Updated 2026-11-02 after the O34, O39 and O41 decisions. All numbers below use the **54 reviewed queries of golden v2** (24 + 22 answerable and 4 + 4 unanswerable on dev and test). Golden v3 (O39) added R-0055 and R-0056 as reviewed (56 queries); no run was repeated. The 12 near-miss queries are still draft and are not used.
All choices were made on the **dev split**; the **test split** was used once per frozen configuration (`reports/layer1/GOLDEN_TEST_USE.md`). Sources: `WINNERS.json`, the per-axis reports in this folder, `FROZEN_TEST*.md`, `REVIEWED_ONLY.md`,
`PRECISION*.md`, snapshots in `eval/snapshots/`. CPU only, 2 threads.

## 1. The final configuration
| setting | value | why |
|---|---|---|
| embedder | `all-MiniLM-L6-v2` (free, local) | On this corpus, with settings tuned for MiniLM, the other embedders did not perform better. The chunk settings were selected with MiniLM before the embedder sweep (`embedder.md`, `embedder_check.md`) |
| chunking | recursive, 200 tokens, 10 % overlap; **v2: one chunk per table row with the header repeated** | `chunker.md`, `chunk_size.md`, `overlap.md`, `table_rows.md` |
| retrieval | hybrid: dense plus BM25, RRF (k = 60), 20 candidates from each; no reranker; top 3 | `retrieval_mode.md`, `reranker.md`, `candidates.md`, `top_k.md` |
| vector store | **Qdrant (local mode)**, not FAISS; Chroma gives identical results | `vector_store.md` |
Configuration **v2** (table rows) was frozen on dev by the rule that you gave and that I fixed before the run (not worse on dev); `v1` is the same without table rows. **Decision O41 (author, 2026-11-02): keep v2.** It passed the pre-set rule; v1 or v2 is not chosen with test results. The test difference is a known limitation (section 2 and section 7).

## 2. Test results (reviewed queries, one run each; 22 answerable; 95 % intervals over queries)
| | Recall@1 | Recall@5 | MRR@10 | nDCG@5 | P@3 | P@5 | unanswerable flagged / answerable flagged |
|---|---|---|---|---|---|---|---|
| v1 (no table rows) | 0.77 (0.59–0.91) | 0.95 (0.86–1.00) | 0.83 (0.68–0.94) | 0.79 (0.66–0.90) | 0.30 | 0.22 | 1.00 / 0.23 (tau 0.490) |
| v2 (table rows) | 0.73 (0.55–0.91) | 0.91 (0.77–1.00) | 0.80 (0.65–0.93) | 0.75 (0.60–0.88) | 0.27 | 0.20 | 1.00 / 0.23 (tau 0.464) |
Dev: v2 was better (quality Q +0.024, Recall@1 0.88 against 0.79); test: v2 against v1 is Q −0.036 (95 % interval −0.106 to +0.014), Recall@1 −0.045. Both differences lie inside the noise of 22 to 24 queries, but the sign changed. Table-row lookups improved
(R-0033 on dev from rank 2 to 1, R-0034 on test from 4 to 2); R-0036 ("which fault buses also carry a micro-PMU") fell on test from rank 1 to 5, because 111 near-identical row chunks now compete with the narrative passage. I changed nothing. **Known limitation (O41):** on test, v2 minus v1 is Q −0.036 (95 % interval −0.106 to +0.014); the cause of the largest single loss (R-0036) is the competition of 111 near-identical row chunks with the narrative passage.
Hybrid against dense, on test: MRR@10 +0.145 (+0.045 to +0.255) and Recall@1 +0.261 (+0.087 to +0.435) (23 queries incl. the draft R-0055; `FROZEN_TEST.md`).

## 3. Precision (`PRECISION.md`, `PRECISION_final_v2.md`; a passage is relevant if it holds a golden span; M = relevant chunks in the index, 1 or 2 per query)
| | split | P@1 | P@3 | max P@3 | normalized P@3 | normalized P@5 | R-precision |
|---|---|---|---|---|---|---|---|
| v1 | dev | 0.79 | 0.36 | 0.43 | 0.85 (0.73–0.96) | 0.88 (0.75–0.98) | 0.79 (0.65–0.92) |
| v1 | test | 0.77 | 0.30 | 0.41 | 0.75 (0.57–0.91) | 0.91 (0.80–1.00) | 0.66 (0.48–0.82) |
| v2 | dev | 0.88 | 0.36 | 0.42 | 0.88 (0.75–0.98) | 0.88 (0.75–0.98) | 0.85 (0.73–0.96) |
| v2 | test | 0.73 | 0.27 | 0.41 | 0.73 (0.57–0.89) | 0.84 (0.70–0.95) | 0.64 (0.43–0.80) |
Plain P@3 cannot exceed the maximum in the fourth column; normalized P@k removes that cap. The graded precision from your labels (`reports/layer1/LABELLING.md`, 162 passages) is not computed yet.

## 4. Your earlier questions
- **Do 200-token chunks split the answer spans?** No: 0 of 31 dev and 0 of 28 test golden contexts were split by v1; 0 of 27 test contexts by v2. A split happened only with `fixed_token` at 100 and 150 tokens in the first run (`rule_v1/`).
- **Why was Recall@1 low?** Ranking confusions, not lost spans: a sibling note (R-0003), valid alternative evidence (R-0016: note_06 also answers; the golden item names note_16; not changed), a bare table row (R-0033; v2 helps), vocabulary gaps (R-0021, R-0042).
- **No-answer threshold (dev only):** tau is one input. It flags 4 of 4 unanswerable queries and 5 of 22 answerable ones on test, so a flag is right in 4 of 9 cases; with 4 unanswerable queries per split it is a rough guide.
  The S4 decision also uses the generator's "insufficient evidence" output (`S4_PLAN.md` § 4). 12 near-miss queries (draft) are ready for your review.
- **Embedder:** MiniLM minus bge-small, e5-small, OpenAI on dev, paired by query (24): Q +0.060 (−0.017 to +0.136), +0.044 (−0.055 to +0.163), +0.082 (−0.005 to +0.190). Self-retrieval 1.000 for all four; vectors normalized; same chunks in the same order; prefixes correct.
- **Vector store:** FAISS only as the exact reference. Chroma (default and high-recall HNSW) and Qdrant gave identical top-10 lists and metrics for all dev queries (and for the test queries in v1). Qdrant is faster (p95 0.14 against 0.48–0.79 ms) and the plan prefers it for hybrid search with filters.

## 5. All axes (Q = mean of MRR@10, Recall@5, nDCG@5 on dev; winner = best quality, then cheapest within 0.013)
| axis | candidates (Q) | winner |
|---|---|---|
| chunker | parent_child 0.832, recursive 0.821, sentence 0.820, structure_aware 0.814, fixed_token 0.811, semantic 0.652 | recursive (parent_child needs 1,468 context tokens against 643) |
| chunk size | 200: 0.821, 150: 0.736, 100: 0.670 | 200 |
| overlap | 10 %: 0.848, 15 %: 0.839, 0: 0.821 | 10 % |
| embedder | MiniLM 0.848, e5-small 0.806, bge-small 0.790, OpenAI 0.769 (settings tuned for MiniLM) | MiniLM |
| retrieval mode | hybrid (k_rrf 20, 60, 100) 0.882, dense 0.848, BM25 0.787 | hybrid, k_rrf 60 |
| reranker | off 0.882, cross-encoder 0.859 | off |
| candidates, top_k | 20 and 50 tie; k = 3, 5, 8 tie in recall | 20; 3 (fewest tokens) |
| table rows | v2 0.899, v1 0.875 | v2 (your rule) |
Without the two draft queries no winner changes (`REVIEWED_ONLY.md`).

## 6. Selection rule and disclosure
Rule v1 (cheapest inside the 95 % interval) was replaced by rule v2 (best quality, then cheap) at your instruction, after the dev results of three axes had been seen; v1 results: `rule_v1/`. All axes were rerun. The table-row rule and the store rule were written before their runs.

## 7. Limits and open points
- **Known limitation (O41):** v2 was kept although its test result is lower than v1 (section 2). **Future work:** a diversity limit for row chunks (for example, at most n row chunks of one table in the candidate list). It is to be designed and selected on dev only; it is not part of v2.
- 24 dev and 22 test answerable queries: wide intervals; test numbers are lower than dev after selection on dev. Ties are ties.
- **Drafts:** 10 near-miss queries (R-0059 and R-0060 were accepted in O39b; you review the rest; `reports/layer1/NEAR_MISS_CHECK.md`), the SOP (1 of 37 documents). The 11 notes (O34) and the spans of R-0055 and R-0056 (O39) were accepted.
- **The OpenAI embedding call (0.000626 USD) sent the text of all 37 corpus documents, including the local-only revision manuscript, to OpenAI.** No API call and no paid embedder on this corpus again (code refuses). In S4, manuscript chunks would be 24 % of the passages sent (`S4_PLAN.md` § 2).
- Golden contexts name one answering span; other passages that answer count as misses; several queries use bare table rows as spans (accepted in your review; v2 repeats the header).
- Not done: LLM-judged precision (S4, needs your approval and the golden events); graded precision (needs your labels); the case store keeps FAISS (frozen, part of the gate).
