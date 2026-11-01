# DONE.md — frozen items, agent-writable areas, completed stages

Read this file before you change anything in the GridSense track. § 1 is frozen: no agent edits, moves, deletes or
regenerates these items. § 2 lists what an agent may change. § 3 records each completed stage with its evidence.
To unfreeze an item, the author must say so; record the date and reason in § 4.

## 1. Frozen — do not touch
| Item | Why |
|---|---|
| `rag-for-dropout/implementation-plan.md`, `rag-for-dropout/rag-eval.md` | Source spec (author). Edits denied in `.claude/settings.json` |
| `rag-for-dropout/spec/**` | Verbatim split of the source spec; `tests/gridsense/test_docs.py` checks the bytes |
| `rag-for-dropout/RAG-arch.png` | Author figure. Corrections go to `PROPOSED_CHANGES.md` (P1) |
| `dataset_fullscale_50/`, `dataset_fullscale_50_scenario/` | Frozen corpus and splits (root `CLAUDE.md`) |
| `results/**` (checkpoints, metrics, curves, `RESULTS.md`) | Paper-1 results. Read only |
| `fault_localization/**` | Paper-1 and PC-STGNN code. Import it; do not change it |
| `paper/`, `thesis-for-college/`, `PC-STGNN/`, `PINNS/`, `src/pinns/` | Other tracks |
| `data/ood_test/` (sealed), `data/ood_pilot/` (not reportable) | Root `CLAUDE.md` |
| Golden items with `status: reviewed` | Only the author changes a reviewed item; a change makes a new golden version |
| Snapshots in `rag-for-dropout/eval/snapshots/` | Append-only; never edit or delete an old snapshot |
| Accepted baselines (listed in § 3) | A new baseline needs the author's approval |
| `rubrics/*.md` with `status: approved` | The rubric hash is in snapshots |
| `rag-for-dropout/reports/s2/*.json`, `eval/latents/seed42/**` | S2 results and extracted votes/latents (seed 42, checkpoint sha256 in each `extract_meta*.json`); re-extract only with the author's approval |
| `rag-for-dropout/configs/gate_v1.json`, `gate_v1.sha256` | The frozen gate (sha256 `693ff351157dc70e…`). Never edit; a changed gate is a new version `gate_v2` with the author's approval. `gate_test.py` refuses a second run on test |
| `rag-for-dropout/reports/layer0/seed43_layer0_test.json`, `seed44_layer0_test.json` | Seeds 43/44 results (frozen thresholds; not re-run) |
| `rag-for-dropout/reports/layer0/test_run_seed42.json` | Result and lock of the single frozen test run |
| `rag-for-dropout/reports/layer0/DECISION_RULE.md` | Pre-registration (commit `aa7454b`); only dated addenda at the end |
| `rag-for-dropout/eval/case_store/seed42_sub10/**` | Case store; the fingerprint is in `manifest.json`; rebuild only with the author's approval |
| `rag-for-dropout/corpus/v1/**` | Frozen corpus copies; SHA-256 in `MANIFEST.json`, checked by `tests/gridsense/test_corpus.py`. `1170_revised.tex` stays local and Git-ignored (unpublished manuscript, like `/resubmit/`) |
| `rag-for-dropout/.env` | Secrets. Never read aloud, print, copy or commit. Git-ignored |
| Root `CLAUDE.md`, `CLAUDE-PINNS.md` | Only the 2-line GridSense import was added (2026-10-28); other edits need the author |

## 2. Agent may change
| Area | Conditions |
|---|---|
| `src/gridsense/**`, `tests/gridsense/**` | Follow `IMPLEMENTATION.md`; run `uv run pytest tests/gridsense -q` after each change |
| `rag-for-dropout/configs/*.yaml` | Only keys that are not `# OPEN`; an OPEN key needs the author's answer |
| `rag-for-dropout/data/eval/*.jsonl` | Add or edit items with `status: draft` only |
| `rag-for-dropout/rubrics/*.md` | Only while `status: draft` |
| `rag-for-dropout/corpus/notes_v1/`, `corpus/sop_v1/` | Only while `status: draft`; a reviewed note changes only with the author. `corpus/feeder_v*/` is generated: never edit, regenerate into a new folder |
| `rag-for-dropout/reports/**` | Generated reports and sweep tables |
| `rag-for-dropout/eval/snapshots/`, `eval/items/` | Add new files only |
| `PLAN.md`, `IMPLEMENTATION.md`, `EVAL.md`, `claude-rag.md`, `dropout-rag-intro.md` | Keep each < 200 lines; no new number without a source |
| `OPEN_ITEMS.md`, `PROPOSED_CHANGES.md`, `EXPERIMENT_LOG.md`, this file § 3 | Append; close an item only with the author's answer |
| `.claude/skills/rag-*/SKILL.md` | Keep < 200 lines and the frontmatter format |
| `pyproject.toml`, `uv.lock` | Only the `rag` dependency group, after the author approves (S2) |

Never: commit to any branch except `gridsense-eval` (method: `claude-rag.md` § 9), push, reset, clean, delete a frozen item, start a GPU run, start a paid run, or upload to LangSmith
without the author's approval for that action.

## 3. Completed stages
| Stage | Date | Evidence | Approved by author |
|---|---|---|---|
| S0 — support files, skills, scaffold, stage-1 inspection | 2026-10-28 | `EXPERIMENT_LOG.md` E0; `tests/gridsense/test_docs.py` passes | yes (answers to O2–O11, 2026-10-28) |
| S1 — open items O2–O5, O9–O11 closed | 2026-10-28 | `OPEN_ITEMS.md` § Closed; `EXPERIMENT_LOG.md` E1; first commit on `gridsense-eval` | yes |
| S3 part 1 (session A): corpus `s3_a1` (merged notes, feeder_v1, sop_v1), retrieval stack, golden v0 re-checked, smoke run (PRELIMINARY) | 2026-10-31 | `EXPERIMENT_LOG.md` E12; `tests/gridsense/test_s3_retrieval.py` | pending (O30) |
| ~~Layer 1 minimal pipeline (session B): corpus v2, parsers, chunker, MiniLM index, golden draft, baseline (PRELIMINARY)~~ **SUPERSEDED 2026-10-31** by session A's S3 (E12); files in `_quarantine_session2/` | 2026-10-31 | `_quarantine_session2/rag-for-dropout/reports/layer1/BASELINE_REPORT.md`; `EXPERIMENT_LOG.md` E11 | superseded |
| The single run of the frozen gate on `test` (seed 42) | 2026-10-30 | `reports/layer0/test_run_seed42.json`; `EXPERIMENT_LOG.md` E7 | pending |
| Conformal 90 % per k; seeds 43 and 44 (Layer 0, frozen thresholds); `FINAL_REPORT.md` | 2026-10-30 | `reports/layer0/{seed43,seed44}_layer0_test.json`; E8 | pending (O25–O28) |
| Gate frozen: `configs/gate_v1.json` (sha256 in `configs/gate_v1.sha256` and the commit message); full train store (seed 42); leakage checks on it | 2026-10-30 | `EXPERIMENT_LOG.md` E6; freeze commit | pending (author reads the report) |
| Layer 0 step 2 (`val`, both stores), leakage tests, FAISS case store, O6 thresholds | 2026-10-30 | `reports/layer0/STEP2_REPORT.md`; `EXPERIMENT_LOG.md` E5; commit of this entry | pending (O18, O21–O24) |
| S2 — wrapper, k = 0 and k > 0 extraction, report window (25), latent method (b), calibrator (isotonic), identity gate | 2026-10-28 | `reports/s2/S2_REPORT.md`; `EXPERIMENT_LOG.md` E3; k = 10 tolerance open (O17) | O17 closed 2026-10-29 (pass); O18–O20 open |

## 4. Unfreeze log
(none)

## 5. Stage ownership
S3: owned by session A. Session B is closed. Its work is in _quarantine_session2/.
A session must not write to files of a stage that it does not own. Keep `_quarantine_session2/` until the author approves deleting it (2026-10-31).
