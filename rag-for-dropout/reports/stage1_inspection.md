# Stage 1 — repository inspection (`rag-eval.md` §3)

Date: 2026-10-28. Branch `main`, HEAD `4caa47c` (uncommitted PC-STGNN and PINNS work present). Read-only inspection.
No file outside `rag-for-dropout/` was changed for this report. "Not found" means the component does not exist yet.

| # | Item | Finding |
|---|---|---|
| 1 | STGNN inference entry point, checkpoint, preprocessing | Model build: `fault_localization/experiments/runner.py:_model` (from the run `config.yaml`). Load: `scripts/run_dropout_study.py:load_trained_model` (`torch.load(..., weights_only=True)`, `ckpt["model"]`). Inference: `fault_localization/experiments/evaluate.py:evaluate_loader`. Checkpoints: `results/dropout_aware/rgsage_max/seed_000{42,43,44}/checkpoint_best.pt`. Preprocessing: `fault_localization/data/dataset.py:WindowGraphDataset` (normalizes voltage channels 0–2 only; stats in `processed/norm_stats.json`), `fault_localization/processing/normalization.py` |
| 2 | STGNN output and classes | `RGSAGE.forward` (`models/graphsage.py`) returns node logits [B, 25, 26]. `models/voting.py:soft_vote` returns the sum of node softmaxes (sum = 25) and the argmax. 26 classes: 0 = no fault, c = fault bus `FAULT_BUSES[c-1]` (`config/constants.py`). Bus-level classes only (no section, no fault type output). **Ambiguity set: not found** |
| 3 | Calibration | **Not found** for the STGNN. (`models/zone_refine.py` has a learned zone temperature; it is part of a different model, not a calibration.) |
| 4 | Deterministic gate | **Not found.** SDR-STGNN (`models/sdr_stgnn.py`) has an availability gate inside the network; it is an architecture, not a decision gate. No thresholds, reason codes or outcomes exist |
| 5 | Case store | **Not found.** Latent candidate: the last spatial-layer output `h` [B, 25, 64] in `RGSAGE.forward`, before `NodeClassifier` (`models/common.py`: BN → dropout → ReLU → linear). No vector database in the repo |
| 6 | Document corpus | **Not found.** Approved sources (author, 2026-10-28): `paper/overleaf.tex` (version O11) and `results/RESULTS.md`. No parser, chunker or embedding model is installed |
| 7 | Retriever | **Not found** (no dense, lexical, hybrid or reranker code; no `faiss`, `rank-bm25`, `sentence-transformers`, `chromadb`, `qdrant-client` in `pyproject.toml`) |
| 8 | LangGraph workflow | **Not found** (no `langgraph`, `langchain` dependency) |
| 9 | Report generator | **Not found** (no LLM client; no `openai` dependency; no API key in the environment; no Ollama) |
| 10 | Existing tests, evaluation scripts, metrics, results | `tests/` (Paper 1 / PC-STGNN, 242 tests per root `CLAUDE.md`), `tests/pinns/`. Dropout evaluation: `fault_localization/dropout/evaluator.py:evaluate_at_dropout`, `evaluate_dropout_curve`; `scripts/run_dropout_study.py`. Statistics: `scripts/run_significance_tests.py`. Results: `results/dropout_aware/`, `results/dropout_study_aware/degradation_curves.csv`, `results/RESULTS.md`. Per-window test predictions: `results/dropout_aware/rgsage_max/seed_*/predictions_test.csv` (k = 0 only). No DeepEval, no LangSmith |
| 11 | Configuration files | `config/experiment_base.yaml`, `config/rgsage_max.yaml`, `config/dropout_aware/` (generated), the run `config.yaml` (authoritative for a checkpoint; `train_k_range [0,1,2,5]`, `k_levels [0,1,2,5,10]`, `n_mask_seeds 3`, `zero_messages`). Frozen CLI flags: `--gnn-layers 4 --epochs 30 --grad-clip-norm 1.0 --warmup-epochs 3`. Constants: `fault_localization/config/constants.py` |
| 12 | Data splits | `fault_localization/processing/splits.py:split_events` (stratified by fault bus, event-level 70/15/15), `verify_no_leakage`. Files: `dataset_fullscale_50/splits/{train,val,test}.txt` = 1,155,000 / 248,000 / 247,000 windows; `val_sample.txt` (30,000) is a subset of `val`. No separate calibration split. The split divides runs, not scenarios (all 825 scenarios in train and test). Scenario-held-out splits: `dataset_fullscale_50_scenario/splits/` |

## Facts that change the design (details in `PROPOSED_CHANGES.md`)
1. The figure's 0.846 is the standard model; the dropout-aware model is 0.839 at k = 0 (P1).
2. The dropout grid is k ∈ {0,1,2,5,10}, not 10 % steps (P2, accepted).
3. The soft vote is not a probability (P3, O2).
4. `val` already selected the checkpoint (P4).
5. Near-duplicate neighbours are expected because scenarios are shared (P5).
6. The unit is the window; windows w0000–w0019 of every run are label 0. Window w has w − 20 post-fault samples, so
   w0020 is fault-labelled with none (P6; report window by the O3 rule, closed 2026-10-28).
7. Dropout masks depend on `batch_idx` (`dropout/evaluator.py`), so Layer 0 must use the paper's batch size and order
   to reproduce `degradation_curves.csv` before it adds new metrics.

## Compute and environment on 2026-10-28
- GPU: T1000 8 GB, used by the S4.4 queue (`scripts/run_s4_4_queue.py`, ETA about 6 Nov). NVML is broken.
- CPU: 24 cores; a PINNS Gate-2 CPU job runs under `pinns.cpu_guard`. RAM: 15 GB, about 4 GB available.
- Docker is installed (not needed: Qdrant local mode and Chroma run in-process).
