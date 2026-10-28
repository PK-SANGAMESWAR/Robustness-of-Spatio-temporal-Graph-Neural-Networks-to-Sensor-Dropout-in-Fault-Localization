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
| `rag-for-dropout/.env` | Secrets. Never read aloud, print, copy or commit. Git-ignored |
| Root `CLAUDE.md`, `CLAUDE-PINNS.md` | Only the 2-line GridSense import was added (2026-10-28); other edits need the author |

## 2. Agent may change
| Area | Conditions |
|---|---|
| `src/gridsense/**`, `tests/gridsense/**` | Follow `IMPLEMENTATION.md`; run `uv run pytest tests/gridsense -q` after each change |
| `rag-for-dropout/configs/*.yaml` | Only keys that are not `# OPEN`; an OPEN key needs the author's answer |
| `rag-for-dropout/data/eval/*.jsonl` | Add or edit items with `status: draft` only |
| `rag-for-dropout/rubrics/*.md` | Only while `status: draft` |
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

## 4. Unfreeze log
(none)
