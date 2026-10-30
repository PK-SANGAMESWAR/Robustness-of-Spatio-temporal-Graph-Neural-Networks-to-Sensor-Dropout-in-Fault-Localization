# claude-rag.md — GridSense RAG track: rules for Claude Code and agents

Imported by the root `CLAUDE.md`. Applies to `rag-for-dropout/`, `src/gridsense/`, `tests/gridsense/` and the
`.claude/skills/rag-*` skills. The root `CLAUDE.md` rules also apply: never invent a number, do not commit, do not reset or
clean, stop and ask before any new GPU run. Written 2026-10-28. Language: ASD-STE100 style.

## 1. Project
GridSense puts a verification layer and a RAG report agent around the Paper-1 **dropout-aware RGSAGE-Max** STGNN:
STGNN → deterministic gate (CONFIDENT / LOW_CONFIDENCE / REJECT) → latent-kNN case retrieval + document retrieval →
LangGraph report → human approval. No component operates grid equipment. Overview: `dropout-rag-intro.md`.
Stage plan: `PLAN.md`. Build guide: `IMPLEMENTATION.md`. Evaluation design: `EVAL.md`. Frozen list: `DONE.md`.

## 2. Source of truth
- `implementation-plan.md` (system design, rev 2) and `rag-eval.md` (evaluation prompt) are **frozen**. Do not edit them.
  Read them through the verbatim parts in `spec/` (index: `spec/00_INDEX.md`). `.claude/settings.json` denies edits.
- Write corrections and gaps in `PROPOSED_CHANGES.md`. Write blocking questions in `OPEN_ITEMS.md`.
- Precedence: (1) author decisions in § 3; (2) accepted entries in `PROPOSED_CHANGES.md`; (3) `rag-eval.md` for
  evaluation, `implementation-plan.md` for design; (4) otherwise stop and ask.

## 3. Author decisions (2026-10-28)
| Item | Decision |
|---|---|
| Model | Dropout-aware RGSAGE-Max, `results/dropout_aware/rgsage_max/seed_000{42,43,44}/checkpoint_best.pt`. Read only |
| LLMs | OpenAI for generation and for the judge, two different models. The author gives model names and prices (O1) |
| Eval stack | DeepEval offline (component and end-to-end, local snapshots). LangSmith for traces, datasets, experiments, online judges |
| Unit | The window, as in the paper. Group by run: leave-one-out excludes all windows of the same run; bootstrap resamples runs |
| Report window | One report per event, from one fixed window index chosen on `val` (O3): the earliest index where top-1 accuracy stops rising. Window w has w − 20 post-fault samples (w0020 has 0, so w ≥ 21). Never chosen on `test`. The golden event set uses the same index |
| Event view | Event-level aggregation is a Layer 0 informational metric only. Never use it in the gate |
| Corpus | Paper 1 and `results/RESULTS.md`, `document_type: model_card`, paper version in the metadata. Other sources: O7 |
| Dropout grid | k ∈ {0, 1, 2, 5, 10} of 25 sensors, the paper protocol (`dropout/masking.py`, `zero_messages`) |
| Long docs | Frozen sources plus verbatim split copies in `spec/`. Every new md file < 200 lines |
| Store version | The scenario-held-out store defines the gate and τ_k; the standard store is secondary only (O20, 2026-10-30) |
| kNN agreement | Kept in the gate: pre-registered rule (`reports/layer0/DECISION_RULE.md`) gave D = +0.0112, CI [+0.0086, +0.0138]. Seeds 43/44 only after the author answers (O18) |
| Gate composition | Three features: isotonic probability, margin (top-1 − top-2), kNN agreement (held-out, 10 neighbours). Combination fitted on cal-A, thresholds per k on cal-B. **Selected on val after the pre-registered test** (O21, 2026-10-30) |
| Calibration, k | Isotonic calibration fitted separately for each k; per-k thresholds. A k or dropout pattern not fitted → REJECT with `UNSUPPORTED_DROPOUT_PATTERN` (O22) |
| Freeze | Full train store (held-out and standard) → refit on cal-A/cal-B → freeze `configs/gate_v1.json` (sha256 recorded, committed) → ONE run on `test`; the gate is not changed afterwards; coverage at k = 10 below 50 % on test is reported, not a stop (O24) |
| Candidate sets | Conformal 90 % per k, fitted on cal-A, checked on cal-B (O23) |
| Seeds 43, 44 | Run now, frozen method, Layer 0 only, no new thresholds (O18) |
| Code home | `src/gridsense/`, `tests/gridsense/`. Docs, configs, golden sets, snapshots in `rag-for-dropout/` |
| System seed | Seed 42. Layer 0 is repeated for seeds 43 and 44 if each takes < 30 min of CPU (O5) |
| Calibration | Vote share = soft vote / 25. Monotone map on the top-1 share, Platt and isotonic compared. No temperature on node logits: predictions stay identical to the paper. Record the top-1 − top-2 margin (O2) |
| Cal split | `val` split 50/50 by run, stratified by scenario: cal-A (fit calibration), cal-B (thresholds, calibrator choice). Checkpoint-selection bias in the limitations |
| Report rule | Earliest w ≥ 21 with `val` accuracy within 1 pp of the max over w ≥ 21 (k = 0). `unlabelable_window_mask` in all Layer 0 metrics |
| Predictions | The pipeline uses its **own CPU predictions** in all layers. Paper predictions are a check only, never pipeline input |
| Identity rule | CPU may differ from the paper only where the vote gap (CPU class − paper class) is < 2e-3. Rule set **after** the data was seen; state it in the limitations. k > 0: macro-F1 within 1e-3 of `degradation_curves.csv`. Any other difference: stop and report |
| Determinism | Re-running an extraction must give byte-identical votes and latents; else stop and report |
| Compute | S2 and S3 on the CPU only; no GPU before the S4.4 queue is complete. `nice -n 19`, 4 torch threads. Record the S4.4 epoch time before and during; if it rises by > 10 %, use 2 threads |
| PCA arm | Standardize, then PCA-256; scaler and PCA fitted on the train subsample only |
| Case latent | Per-node embeddings before the head, `h` [25, 64]. On one 10 % train-run subsample (stratified by scenario) compare (a) concatenation, (b) mean + max pooling, (c) concatenation + PCA-256 fitted on the subsample, by kNN-vote accuracy on `val`. Ask before a full store (O4, O15) |
| Paper version | `corpus/v1/`: frozen copies of `resubmit/1170_revised.tex` and `RESULTS.md`, SHA-256 in `MANIFEST.json`; numbers tested against the checkpoint curve. Camera-ready = corpus v2 (O11) |
| Cost | **5 USD in total** for the track. Stop and ask before more (O9, author update 2026-10-28) |
| Git | Commit only to branch `gridsense-eval`, never to `main` (O10). Commit without a checkout (§ 9) |
| Keys | `rag-for-dropout/.env` (Git-ignored). LangSmith key present and valid (2026-10-28). OpenAI key: later (O16) |

## 4. Hard rules
1. **Never invent a number.** Copy each number from a CSV/JSON or `results/RESULTS.md` and name the source.
   The dropout-aware model is 0.839 macro-F1 at k=0, not 0.846 (0.846 is the standard model).
2. **No aggregate metric as confidence.** A report can use model-card documents to explain behaviour under dropout.
   It must never give macro-F1, retention or AUDC as the confidence of the current prediction. Rubric:
   `rubrics/no_aggregate_metric_as_confidence.md`. Safety set: one trap item at minimum.
3. **The LLM never changes the gate or the approval state.** Gate, evidence score, field consistency and approval are
   Python code. A report string "approved" does not change any state.
4. **Splits.** Train → case store only. `val` → cal-A (calibration fit), cal-B (thresholds), report-window choice,
   retrieval dev queries, prompt iteration.
   `test` → reported numbers only. No test window in the case store; assert it in code. Do not tune on `test`.
5. **Golden items.** An LLM-drafted item has `status: draft`. Only the author sets `status: reviewed`. A draft is never
   in a reported result. Golden contexts use span + content hash, never a chunk ID only.
6. **One variable at a time** in every component sweep. Same golden set version, same split, same seed.
7. **Deterministic metrics first.** Paid (LLM-judge, OpenAI embedding, generation) runs need a cost estimate
   (`rag-cost-check`) and author approval. Stop at the cost limit in `configs/eval.yaml`.
8. **Record failures.** A metric exception is a failure with a cause, never a score of 0 or 1.
9. **Fingerprints.** An index or case store with a stale fingerprint is rebuilt or refused, never used.
10. **No secrets.** API keys come from environment variables through one settings module. No key, path or env value
    goes into code, logs, snapshots, LangSmith traces or Git. Ask before the first LangSmith upload (O8).
11. **Compute.** CPU only until the S4.4 GPU queue ends and the author approves a GPU run. CPU jobs: `nice -n 19`,
    6 workers at most, run through `pinns.cpu_guard` beside a GPU queue. Measure RAM first (about 4 GB free).
12. **Honest claims.** Kappa below the bar → the judge metric is informational. n = 3 seeds → report
    trends as trends (root `CLAUDE.md` § claims). The data is simulated: no claim of field safety.
13. **Do not touch** the items in `DONE.md` § Frozen. Change only the items in `DONE.md` § Agent may change.
14. **md files < 200 lines.** `tests/gridsense/test_docs.py` checks it. Split a long file; do not cut content.

## 5. Verified facts (safe to reuse; source in brackets)
- Dropout-aware RGSAGE-Max macro-F1: k=0 0.839, k=1 0.827, k=2 0.815, k=5 0.774, k=10 0.674; AUDC 0.913;
  retention at k=10 80.3 % [`results/RESULTS.md` §3–4]. Per seed at k=0: 0.8515 / 0.8334 / 0.8314
  [`metrics.json` `test_graph.f1`]. Standard RGSAGE-Max: 0.846 at k=0, 0.048 at k=10 [§1–2].
- Dropout-aware training used k ∈ {0, 1, 2, 5} [`config.yaml` `train_k_range`]. Noise is a null for it [§8].
- Splits (windows): train 1,155,000; val 248,000; test 247,000 (6,175 test runs × 40) [`dataset_fullscale_50/splits/`].
  The split divides runs, not scenarios: all 825 scenarios are in train and test.
- Windows w0000–w0019 of every run have label 0; fault windows start at w0020 [`processing/labels.py`]. Class 0 comes
  from pre-fault windows. Label c ≥ 1 is fault bus `FAULT_BUSES[c-1]` [`config/constants.py`].
- Soft vote = sum of 25 node softmaxes; scores sum to 25, not 1 [`models/voting.py`].
- Gate (2026-10-30): `configs/gate_v1.json`, sha256 `693ff351157d…` (frozen in `b7a80b9`). One run on test: coverage at the 95 % point 0.845 / 0.825 / 0.801 / 0.731 / 0.513 for k = 0 / 1 / 2 / 5 / 10;
  CONFIDENT accuracy 0.949–0.950 [`reports/layer0/test_run_seed42.json`]. Thresholds do not transfer to seeds 43 and 44 (accuracy 0.897–0.932). Thread count: `eval/threads.txt` (2).
- Latent candidates: last spatial-layer output `h` [B, 25, 64] before `NodeClassifier` [`models/graphsage.py`].
- S2 results (seed 42, `val`, k = 0; `reports/s2/`): report window index 25 (accuracy 0.9061; max 0.9097 at w = 32);
  latent method (b) mean + max, 128-d (kNN-vote 0.9635 at k_nn = 10; (a) 0.9627, (c) 0.9629: a statistical tie);
  calibrator isotonic on the top-1 vote share (cal-B ECE 0.0022; Platt 0.0145; raw 0.0585).

## 6. Stop and ask before
- any GPU run; any CPU job longer than 30 min; any paid API call; the first LangSmith upload;
- opening `data/ood_test/` (root `CLAUDE.md`) or using `data/ood_pilot/` numbers;
- setting a golden item to `reviewed`, or choosing any value listed in `OPEN_ITEMS.md`;
- adding a dependency (planned: a `rag` dependency group in `pyproject.toml`, stage S2);
- committing to any branch other than `gridsense-eval`; any commit of `.env` or of another track's files.

## 7. Skills
| Skill | Use |
|---|---|
| `rag-eval-run` | Run one layer or the suite; write a snapshot and a log entry |
| `rag-component-sweep` | Compare chunkers, sizes, embedders, vector stores, retrievers, rerankers, top_k |
| `rag-golden-review` | Draft golden items and give the author a review table |
| `rag-compare` | Baseline vs candidate snapshot → PASS / REVIEW / FAIL |
| `rag-cost-check` | Estimate calls, tokens and cost before a paid run |
| `rag-langsmith-sync` | Push datasets and experiments; set online evaluators |

## 8. Doc map
| File | Contents |
|---|---|
| `dropout-rag-intro.md` | What GridSense is, link to Paper 1, verified numbers, limitations |
| `PLAN.md` | Stages S0–S9, stop points, exit conditions |
| `IMPLEMENTATION.md` | Module map, interfaces, config, fingerprints, snapshot format, commands |
| `EVAL.md` | Layers 0–6, component sweeps, golden sets, judge validation, registry, LangSmith online |
| `DONE.md` | Frozen items, agent-writable areas, completed stages |
| `OPEN_ITEMS.md`, `PROPOSED_CHANGES.md` | Blocking questions; spec corrections |
| `EXPERIMENT_LOG.md` | One entry per stage and per run |
| `reports/stage1_inspection.md` | Stage-1 inspection (`rag-eval.md` §3) |
| `configs/`, `rubrics/`, `data/eval/SCHEMA.md` | Eval config and registry; GEval rubrics; golden-set fields |

## 9. Git workflow (O10)
`main` stays checked out: other tracks have uncommitted work and a running GPU queue. Never `git switch` or
`git checkout` for this track. Commit the GridSense paths to `gridsense-eval` with a temporary index:
```
export GIT_INDEX_FILE=$(mktemp -u) && git read-tree gridsense-eval        # parent = branch tip
git add -- rag-for-dropout src/gridsense tests/gridsense .claude/skills/rag-* CLAUDE.md
T=$(git write-tree) && C=$(git commit-tree "$T" -p gridsense-eval -F msg.txt) && git update-ref refs/heads/gridsense-eval "$C"
rm -f "$GIT_INDEX_FILE"; unset GIT_INDEX_FILE
```
Check before each commit: `git ls-tree -r --name-only gridsense-eval | grep -c '\.env$'` must be 0.
`CLAUDE.md` on `main` keeps its other uncommitted edits; commit only its GridSense lines if it changes again.
