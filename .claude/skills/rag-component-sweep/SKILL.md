---
name: rag-component-sweep
description: Compare GridSense RAG components one variable at a time (chunker, chunk size, embedder, retrieval mode, reranker, top_k, vector store) on the retrieval golden dev split with free deterministic metrics, and write a sweep table.
---

# rag-component-sweep

Use this skill with an axis name: `chunker`, `chunk_size`, `embedder`, `retrieval_mode`, `reranker`, `top_k` or
`vector_store`.

1. Read `rag-for-dropout/EVAL.md` § 3 (sweep order and fixed settings) and `rag-for-dropout/claude-rag.md` § 4.
2. Check the order. The axis before this one in the sweep table must have a recorded winner in
   `rag-for-dropout/reports/sweeps/`. If not, stop and tell the author which step comes first.
3. Fix every other setting at the current winner. Change **only** the axis. Use the same golden set version, the
   `dev` split and the same corpus version for every candidate.
4. Embedder window rule: read `max_seq_length` from the model. Log, per chunk, the token count and whether it is
   truncated. Do not test a chunk size above the window. Apply query and passage prefixes in one function.
5. For each candidate: rebuild the index (new fingerprint), run the free metrics — Recall@k, Precision@k, MRR,
   nDCG@k (k = 1, 3, 5, 10), exact-identifier recall, no-answer behaviour, latency p50/p95, build time, peak RAM.
6. Vector-store axis: run in exact mode first. Ranking parity with FAISS-flat must be 100 %; if not, find the cause
   (normalization, distance, ties) before any other number. Then measure ANN recall vs exact, latency, build time,
   RAM, disk size, re-open time, metadata filter and hybrid support. Quality does not choose the store.
7. Paid candidates (OpenAI embedding, DeepEval contextual metrics): run `rag-cost-check` first and wait for approval.
8. Write `rag-for-dropout/reports/sweeps/<axis>.md` (< 200 lines): one row per candidate, the metrics with 95 % CIs
   (bootstrap over queries), the winner, and the reason. Inspect the 5 worst queries by hand and list them.
9. A difference inside the CI is a tie. On a tie, choose the simpler or cheaper candidate and say so.
10. Write a snapshot per candidate and one `EXPERIMENT_LOG.md` entry for the sweep. Do not run on `test` here.
