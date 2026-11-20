# LangSmith status (S8, 2026-11-20)

Written for the author. You said "use langsmith online i have given api key right". I read it as the approval for the first upload (O8) and recorded it in `configs/eval.yaml` (`langsmith.enabled`, `upload_approved`, project `gridsense-eval`) with the scope below.

## What is online now
| Item | Content | Where |
|---|---|---|
| Dataset `gridsense-retrieval` | 48 reviewed retrieval queries: query, reference answer, answerable flag, golden-context references (document id, section, span hash, grade). **No span text.** Payload sha256 `2827cee7c6ad…` | dataset id `25230709-c549-4102-9f77-a6b8b9e10075` |
| Experiment `gridsense-retrieval-v2-dev-c67d2ddb` | The frozen retrieval configuration v2 on the 25 **dev** queries of the dataset. Evaluators ran locally: recall@1/3/5, MRR@10, nDCG@5, evidence flag. Outputs: document id, section, chunk id, score. **No passage text** | `eval/snapshots/20261120T045816Z_s8_langsmith_retrieval_dev.json` (holds the URL) |

Dev numbers (19 answerable queries; recall@1, 3, 5 and MRR@10 all 0.947; evidence flag correct for 0.80 of 25) are a replay on the split where configuration v2 was tuned: **not a result**. The frozen test result is `reports/sweeps/FROZEN_TEST_v2.md`.
The test split was not run again (it was used once per frozen configuration).

## What stayed local
- 10 draft queries and 10 reviewed queries whose golden context comes from the unpublished manuscript (`paper1_rev1170`), `results_md` or the draft SOP.
- In the experiment, chunks of those three sources are withheld: only their score is sent (`[withheld: sensitive source]`).
- The 60 golden events and the 25 safety items (draft), all reports, all passage text, and every trace. Tracing is off (`LANGSMITH_TRACING=false`).

## A mistake I made and fixed
My first experiment run sent document ids and section headings of retrieved chunks, including headings of the unpublished manuscript. I found it by reading the runs back, **deleted that experiment from LangSmith**, withheld the sensitive sources in the code and ran again.
Read back after the rerun: 25 of 25 runs, no sensitive id visible. The local snapshot of the deleted experiment (`…T045736Z_s8_langsmith_retrieval_dev.json`) is kept (snapshots are append-only); its URL no longer works.

## Not done, and why
- **Tracing and online evaluators** would send report text, event facts and passages to LangSmith. I did not enable them. If you want them: say which sources may leave the machine (the corpus notes are generated research material; the manuscript and `RESULTS.md` are not). I would trace only a replay stream with the sensitive sources withheld.
- **Events and safety items** as datasets: only after you review them (`reviewed`); drafts never go up (skill).
- The account also holds an unrelated dataset, `legal-rag-golden-dataset`. I did not touch it.

## Reversal
Set `langsmith.upload_approved: false` in `configs/eval.yaml`: the upload command refuses (tested). To remove the online data, delete the dataset and the experiment in the LangSmith UI.
