# EXPERIMENT_LOG_2.md — entries from E16 (continues EXPERIMENT_LOG.md)

One entry per stage and per run. Append only.

## E16 — 2026-11-02 — O30 review applied: golden set v1, notes, feeder documents accepted
- The author's review was applied. Record: `reports/layer1/O30_REVIEW_RECORD.md`. Scripts: `data/eval/apply_o30_review.py` and `corpus/notes_v1/_review_2026-11-02.py` (both assert the facts before they write).
- Golden retrieval v0 (54 drafts) → **v1: 56 reviewed items**: R-0005 edited (results_md row removed), R-0034 and R-0044 edited (reference answers), R-0055 and R-0056 added, R-0036, R-0047, R-0048 accepted unchanged. Dev 25 answerable + 4 unanswerable; test 23 + 4.
- Notes: 19 reviewed. The 11 notes that the review did not mention stay draft, with the SOP (O34). Found while checking: note_16 had a wrong "indistinguishable" pair, note_20 overstated "the margin is the only threshold" (k = 2 has a probability threshold 0.5424; the margin thresholds are not monotone), note_09 had one unsupported sentence. All corrected. note_07: 25 μPMUs confirmed against the dataset config. 0.846 appears only as the standard model (a test checks every note).
- Feeder documents accepted, not edited (manifest hashes intact). The nearest μPMU to bus 51 is bus 47, 0.229 km, 3 hops; independent check from the raw DSS lines (3 × 0.25 kft = 0.229 km).

## E17 — 2026-11-02 — S3 sweeps (Layer 1, deterministic) and the frozen test runs
- Code: `eval/sweep.py` (axes, selection, frozen runs), `eval/store_bench.py`, `layer1.evaluate()` with reachability (golden spans cut by chunk boundaries), context tokens and memory, a chunk meter, Chroma and Qdrant variants. 184 tests pass.
- Author instructions during the run. (1) "Look for best + cheap, not just cheap": selection rule v1 (the cheapest candidate inside the 95 % interval) was replaced by rule v2 (quality first; equal = within one query on one metric = 0.013; then the cheapest) after the dev results of three axes had been seen. Rule v1 results: `reports/sweeps/rule_v1/`. All axes were rerun. (2) "Use a good vector store, not FAISS (Chroma and others), keep the precision good": FAISS only as the exact reference; end-to-end test with the real hybrid retriever.
- Winners on dev: MiniLM; recursive, 200 tokens, 10 % overlap; hybrid RRF (k 60, 20 candidates); no reranker; top_k 3; Qdrant local. Chroma (default and high-recall HNSW) gives identical top-10 lists and metrics for all 29 dev queries and for all test queries.
- Frozen, one run each on the golden test split (`reports/layer1/GOLDEN_TEST_USE.md`): final: Recall@1 0.74, Recall@5 0.91, MRR@10 0.79, P@3 0.29, P@5 0.21 (23 answerable). final with Chroma: identical. final dense-only: Recall@1 0.48, MRR@10 0.65 (hybrid gain +0.145 MRR@10, 95 % interval +0.045 to +0.255). No-answer (tau 0.490 from dev): 4 of 4 unanswerable flagged, 5 of 23 answerable flagged.
- Low Recall@1: 0 of 31 dev and 0 of 28 test golden contexts are split by the final chunking; the causes were read by hand (sibling note, alternative evidence, table row, vocabulary gap): `reports/sweeps/SUMMARY.md` § 4.
- OpenAI `text-embedding-3-small` as one candidate: 31,287 tokens, 0.000626 USD (estimate 0.000626, cap 0.002); ledger total 0.00063 of 5 USD. The text of all 37 corpus documents, including the local revision manuscript, was sent. It scored lower than MiniLM (MRR@10 0.693 against 0.803 on dev).
- Snapshots: the first chunker run crashed in the report writer after it had written its snapshots, and rule v1 left more; they are kept (append-only). Not committed.
