# IMPLEMENTATION.md — GridSense build guide

Written 2026-10-28 (S0). Only `src/gridsense/__init__.py` exists. Every other module below is **planned**; its tests
skip with `pending: <module>` until it exists (same pattern as `tests/pinns/`). Rules: `claude-rag.md`.

## 1. Principles
1. Wrap the existing code. Do not copy or change `fault_localization/`. Import `masking.py`, `voting.py`, the dataset
   and the model classes. The evaluation uses the real pipeline, not a copy (`rag-eval.md` §1).
2. No expensive work at import. Load models, indexes and API clients inside functions, on first use.
3. One settings module (`gridsense.settings`) reads `rag-for-dropout/.env` and the environment (`OPENAI_API_KEY`,
   `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). It switches DeepEval and Chroma telemetry off and keeps LangSmith
   tracing off until O8. Import it before those libraries. Nothing else reads the environment.
4. One config file: `rag-for-dropout/configs/eval.yaml`. Code reads it through `gridsense.config` (typed, validated).
5. Every LLM call goes through one interface (`gridsense.llm.client`), with a recorded model name, prompt hash,
   token usage, retries and a response cache. A mock client exists for tests.

## 2. Module map (`src/gridsense/`)
| Module | Contents | Stage |
|---|---|---|
| `settings.py`, `config.py` | Environment variables; typed loader for `configs/eval.yaml` | S2 |
| `schemas/` | `event.py`, `prediction.py`, `gate.py`, `retrieval.py`, `report.py` (Pydantic) | S2–S5 |
| `model/predictor.py` | Load checkpoint (read-only), apply dropout, forward pass, soft vote, latent hook | S2 |
| `model/latents.py` | Batch latent extraction to `.npy` memmaps plus a parquet/CSV index | S2 |
| `verification/calibration.py` | cal-A/cal-B split by run; vote share; Platt and isotonic on top-1 share; margin; ECE, Brier | S2 |
| `model/report_window.py` | Accuracy per window index on `val`; the O3 stop rule; writes `chosen_index` to config | S2 |
| `verification/gate.py` | The frozen gate: CONFIDENT / LOW_CONFIDENCE / REJECT (`UNSUPPORTED_DROPOUT_PATTERN`, `INVALID_INPUT`), per-k isotonic map and thresholds, conformal sets, config hash check (built 2026-10-30) | S2 |
| `eval/knn_features.py`, `layer0_final.py`, `gate_fit.py`, `gate_test.py`, `seed_runs.py` | Full-store neighbour features; evaluation with run bootstrap; fit and freeze; the single test run; seeds 43/44 with frozen thresholds | S2 |
| `verification/conformal.py` | Candidate set at a stated coverage, fit on `val` | S2 |
| `cases/store.py`, `cases/build.py` | Case store (FAISS exact): fingerprint, exclusion of own scenario or run, `LeakageError` for val/test runs and shared seeds, `StaleIndexError` for a stale store or another checkpoint (built 2026-10-30: `eval/case_store/seed42_sub10/`) | S2 |
| `cases/evidence.py` | kNN agreement, kNN vote, neighbour error rate, distance to nearest | S2 |
| `ingest/` | `parsers.py` (`parsers_v1`: md headings, LaTeX sections), `corpus.py` (parts v1, feeder_v1, notes_v1, sop_v1; integrity, fingerprint), `chunking.py` (six strategies), `feeder_docs.py` (`feeder_docs_v2`) | S3 (session A, 2026-10-31) |
| ~~`ingest/` (session B)~~ | SUPERSEDED 2026-10-31: `documents.py` (parsers p1, corpus v2), `structure_recursive` chunker; archived (DONE.md section 5) | — |
| `retrieval/` | `embeddings.py` (local + OpenAI; prefixes in one function; cache; ledger), `stores.py` (FAISS, Chroma, Qdrant local = exact), `lexical.py` (BM25, RRF, cross-encoder), `retriever.py` (fingerprinted index; dense, bm25, hybrid; sufficiency signal); `cost.py` (ledger) | S3 (session A) |
| ~~`retrieval/` (session B)~~ | SUPERSEDED 2026-10-31: `index.py` (exact FAISS + dense); archived (DONE.md section 5) | — |
| `llm/client.py` | OpenAI client, mock client, cache, token and cost counters, total-cost ledger (5 USD) | S4 |
| `agent/` | `state.py`, `graph.py` (LangGraph), `prompts/` (versioned files), `report_validation.py` | S5 |
| `review/approval.py` | Human approval log; a model string never changes the state | S5 |
| `eval/sweep.py`, `eval/store_bench.py` | Layer 1 sweeps (8 axes, selection rule v2, frozen test runs with a one-time ledger) and the vector-store benchmark (parity, end-to-end precision, operations); `layer1.evaluate()` and `reachability()` | S3 (done) |
| `eval/` | `metrics.py`, `golden.py` (span matching), `layers/l0.py … l6.py`, `snapshot.py`, `registry.py`, `compare.py`, `cost.py`, `langsmith_sync.py` | S2–S8 |
| `cli.py` | `python -m gridsense <command>` | S2+ |

Tests: `tests/gridsense/` — `test_docs.py` (S0), then one test file per module; mocked LLM and embedder.

## 3. Interfaces (`spec/impl_10.md` §16)
| Interface | Input | Output |
|---|---|---|
| Prediction | window ID, k, mask seed, checkpoint | scores[26] (vote / 25 and raw sum), top-1, top-2 margin, latent[64], model version |
| Gate | prediction, mask, calibration, case evidence, policy | outcome, reason codes, candidate set, policy version |
| Case retrieval | latent, run ID (for LOO), k_nn | neighbours (run ID, label, prediction, k), evidence signals |
| Document retrieval | structured query from the event, filters, config | passages (document_id, section path, span offsets, hash, score), index fingerprint |
| Report | event, gate result, case evidence, passages, prompt version | Pydantic report, citations, validation result, `pending_human_review` |

The structured query comes from validated event fields (bus, fault class, k, dropped μPMUs, gate outcome). The LLM
never writes the bus, the k or the gate outcome.

## 4. Data and corpus
- Windows: `fault_localization.data.dataset.WindowGraphDataset(data_root, split)` on `dataset_fullscale_50` (read only).
- Report window per run: the window index chosen on `val` by the O3 rule (window w has w − 20 post-fault samples;
  `models/common.py:unlabelable_window_mask` already records that w0020 has none). Read the index from config.
- Dropout: `fault_localization.dropout.masking.apply_sensor_dropout_batch`, policy `zero_messages`, mask seeds as in
  `dropout/evaluator.py` (`mask_seed = 10000*k + offset`, batch seed `mask_seed*10000 + batch_idx`). The batch seed
  depends on `batch_idx`, so use the same batch size and loader order as the paper run, or the masks differ. First
  check: Layer 0 macro-F1 at each k must equal `results/dropout_study_aware/degradation_curves.csv` (`f1_per_train_seed`) before any new metric.
- Corpus: `paper/overleaf.tex` (version: O11) and `results/RESULTS.md`. Metadata per chunk: `document_id`,
  `document_type: model_card`, `approval_status: curated_for_research`, `paper_version` (git blob hash + commit),
  `section_path`, `char_start`, `char_end`, `content_hash`, `chunker`, `embedding_model`, `ingestion_version`.

## 5. Configuration keys (`configs/eval.yaml`)
Sections: `model`, `dropout`, `splits`, `calibration`, `gate`, `case_store`, `corpus`, `chunking`, `embedding`,
`vector_store`, `retrieval`, `reranker`, `generation`, `judge`, `golden_sets`, `repeats`, `cost`, `langsmith`,
`snapshots`. A `null` value with `# OPEN: Oxx` means the author has not chosen it. Code refuses to run a stage whose
required keys are `null`.

## 6. Fingerprints (index invalidation, `rag-eval.md` §7)
- Document index: sha256 of (embedding model + revision, prefixes, chunker, chunk size, overlap, parser version,
  corpus content hashes, vector store + distance). Stored in the collection manifest.
- Case store: sha256 of (checkpoint sha256, latent layer, pooling, split, k grid, mask seeds, dataset manifest hash).
- On load: compare with the current config. A mismatch → rebuild or raise `StaleIndexError`. Never warn and continue.

## 7. Snapshot format (`rag-for-dropout/eval/snapshots/<utc>_<label>.json`)
Keys: `git_commit`, `git_dirty`, `label`, `timestamp_utc`, `config_effective`, `index_fingerprint`,
`case_store_fingerprint`, `prompt_hashes`, `rubric_hashes`, `golden_versions`, `judge_model`, `generation_model`,
`deepeval_version`, `metrics` (flat, e.g. `layer0.k10.selective_risk.mean` and `.ci_low`/`.ci_high`), `n_items`,
`n_failures`, `failure_causes`, `tokens`, `cost_usd`, `langsmith_experiment_url`, `items_path`. Cost: every paid call appends to `eval/cost_ledger.jsonl`; the client
refuses a call that would take the total above `cost.limit_usd_total`.
Per-item records go to `eval/items/<snapshot>.jsonl`. Snapshots are append-only; never edit an old one.

## 8. Golden-context matching
A retrieved passage matches a golden context when the normalized passage (lower case, collapsed whitespace) contains
the normalized golden span, and the `document_id` is equal. Chunk IDs are never used for matching, so a chunk-size
change does not break the golden set.

## 9. Commands (planned; available from the stage shown)
```
uv run pytest tests/gridsense -q                                         # S0
uv sync --group rag --inexact      # rag group; pinned list: rag-for-dropout/requirements.txt (uv export)
PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.model.extract --split test --k 0 --latents   # S2
PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.identity 42     # identity gate; exit 1 = stop and report
PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.eval.s2_k0   # report window, latent method, calibration
PYTHONPATH=src uv run python -m gridsense eval --layers 0 --label <name>      # S2
PYTHONPATH=src uv run python -m gridsense ingest --config rag-for-dropout/configs/eval.yaml   # S3
PYTHONPATH=src uv run python -m gridsense sweep --axis chunker --split dev    # S3
PYTHONPATH=src uv run python -m gridsense cost --layers 1 2 3                 # S4
PYTHONPATH=src uv run python -m gridsense compare <baseline.json> <candidate.json>   # S7
PYTHONPATH=src uv run python -m gridsense langsmith push-datasets             # S8, after approval
```

## 10. Resource rules (T1000 8 GB, 15 GB RAM, about 4 GB free on 2026-10-28)
- The S4.4 GPU queue has the GPU until about 6 Nov. All S2–S3 work is CPU.
- Latents (O4) are **float32**: `h` has entries above 65,504, so float16 overflows to inf (found 2026-10-28).
  Sizes: `h` [25, 64] = 6.4 kB per window → test 1.58 GB, val 1.59 GB, 10 % train subsample 0.74 GB on disk
  (memmaps, Git-ignored). A full train store of concatenated `h` = 7.4 GB → ask first (O15).
- Embedders and rerankers run on CPU, in batches. Load one model at a time.
- Measure peak RAM before a full run with a 1 % subset; never size a result from a subset (root `CLAUDE.md`).
