---
name: rag-golden-review
description: Draft GridSense golden items (retrieval, events, safety) as status draft, then give the author a review table. Never set an item to reviewed; only the author does.
---

# rag-golden-review

Use this skill with a set name (`retrieval`, `events`, `safety`) and a number of items.

1. Read `rag-for-dropout/data/eval/SCHEMA.md` and `rag-for-dropout/EVAL.md` § 11. Read the corpus documents that the
   items use. Do not invent a fact, a number or a section: each `span` must be an exact substring of the source.
2. Drafting rules:
   - Retrieval: mix `paraphrase`, `exact_identifier`, `dropout_observability`, about 15 % `unanswerable`. Write the
     queries as an operator would ask them, not from section titles. Split `dev` / `test` before any sweep runs.
   - Events: pick candidates with the seeded stratified sampler (code), from the `test` split only, at the report-window index
     of `configs/eval.yaml` (O3). If `chosen_index` is null, stop: S2 has not chosen it. Cover every k and all 11 fault types; include many high-confidence errors.
   - Safety: cover every `test_type`, with at least one `aggregate_metric_as_confidence` item.
3. Compute `span_hash` with the normalization of SCHEMA.md. Set `status: draft`, `created`, `created_by: agent:<model>`,
   `reviewed: null`.
4. Append the drafts to the `.jsonl` file. Do not change any item with `status: reviewed`.
5. Give the author a review table: id, type, query or perturbation, golden span (short), source location, and any
   doubt you have. Ask the author to mark each item `accept`, `edit` or `reject`.
6. After the author answers: apply edits, delete rejected drafts, and set `status: reviewed` and the date **only for
   the items the author accepted in writing**. Increase the set version. Record it in `EXPERIMENT_LOG.md`.
7. Never put a draft item into a reported result.
