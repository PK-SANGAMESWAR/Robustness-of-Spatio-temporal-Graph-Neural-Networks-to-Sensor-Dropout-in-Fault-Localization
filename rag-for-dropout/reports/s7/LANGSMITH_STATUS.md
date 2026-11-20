# LangSmith status (S8, 2026-11-20, updated after the traces)

Written for the author. You said "use langsmith online i have given api key right". I read it as the approval for the first upload (O8) and recorded it in `configs/eval.yaml` (`langsmith.enabled`, `upload_approved`, project `gridsense-eval`) with the scope below.

## Where to look (the first version of this page missed traces; you could not see "tracking")
- **Tracing:** LangSmith > Tracing (Projects) > **`gridsense-eval`**: https://smith.langchain.com/o/<org-id-withheld>/projects/p/b474a882-b602-4311-aad5-dc481b7b20e0 . 30 traces (one per S4 event), 510 runs (one per workflow node).
- **Dataset and experiment:** LangSmith > Datasets & Experiments > `gridsense-retrieval` > Experiments. The experiment also shows in the Tracing project list as `gridsense-retrieval-v2-dev-c67d2ddb`.
- Check that the workspace selector (top left) is the organisation that holds the API key.

## What is online now
| Item | Content | Where |
|---|---|---|
| Dataset `gridsense-retrieval` | 48 reviewed retrieval queries: query, reference answer, answerable flag, golden-context references (document id, section, span hash, grade). **No span text.** Payload sha256 `2827cee7c6ad…` | dataset id `25230709-c549-4102-9f77-a6b8b9e10075` |
| Traces in project `gridsense-eval` | A REPLAY of the 30 S4 events through the LangGraph workflow (`eval/langsmith_replay.py`; snapshot `…T052809Z_s8_langsmith_replay_traces`). The generator is a replay of the stored S4 live output (no API call, cost 0); retrieval, tau, checks and assembly ran for real. 14 traces stop at the human-approval pause, 16 end in manual review (the generator had found the evidence insufficient). **No human decision was made.** Tags: replay, s4-events, draft-events, simulated-data | project id `b474a882-b602-4311-aad5-dc481b7b20e0` |
| Experiment `gridsense-retrieval-v2-dev-c67d2ddb` | The frozen retrieval configuration v2 on the 25 **dev** queries of the dataset. Evaluators ran locally: recall@1/3/5, MRR@10, nDCG@5, evidence flag. Outputs: document id, section, chunk id, score. **No passage text** | `eval/snapshots/20261120T045816Z_s8_langsmith_retrieval_dev.json` (holds the URL) |

Dev numbers (19 answerable queries; recall@1, 3, 5 and MRR@10 all 0.947; evidence flag correct for 0.80 of 25) are a replay on the split where configuration v2 was tuned: **not a result**. The frozen test result is `reports/sweeps/FROZEN_TEST_v2.md`.
The test split was not run again (it was used once per frozen configuration).

## What stayed local
- 10 draft queries and 10 reviewed queries whose golden context comes from the unpublished manuscript (`paper1_rev1170`), `results_md` or the draft SOP.
- In the experiment, chunks of those three sources are withheld: only their score is sent (`[withheld: sensitive source]`).
- The golden events as a dataset and the 25 safety items (draft). Traces: sensitive sources are scrubbed before sending (`scrub` withholds their passages, excerpts and ids, and any string that names them). Read back: 30 traces, 510 runs, no sensitive token. In this replay no retrieved chunk came from a sensitive source, so the scrub withheld nothing; it is tested.
- Live tracing of the running pipeline stays off (`LANGSMITH_TRACING=false`): only replays are traced, by explicit call.

## A mistake I made and fixed
My first experiment run sent document ids and section headings of retrieved chunks, including headings of the unpublished manuscript. I found it by reading the runs back, **deleted that experiment from LangSmith**, withheld the sensitive sources in the code and ran again.
Read back after the rerun: 25 of 25 runs, no sensitive id visible. The local snapshot of the deleted experiment (`…T045736Z_s8_langsmith_retrieval_dev.json`) is kept (snapshots are append-only); its URL no longer works.

## Not done, and why
- **Online evaluators** (LLM judge rules on sampled traces) cost judge calls (about 0.027 USD per report for the full suite; one rubric about 0.004). I have not set them. Tell me which rules you want and the sample rate, and I announce the cost first.
- **Events and safety items** as datasets: only after you review them (`reviewed`); drafts never go up (skill).
- The account also holds an unrelated dataset, `legal-rag-golden-dataset`. I did not touch it.

## A smaller point found while checking
LangSmith copied the value of my `LANGSMITH_TRACING` environment variable (`false`) into the run metadata of a 3-trace trial. I deleted that trial project and removed the variable inside the replay process. The 30 traces carry a git revision tag and library versions, no key and no path.

## Reversal
Set `langsmith.upload_approved: false` in `configs/eval.yaml`: the upload command refuses (tested). To remove the online data, delete the dataset and the experiment in the LangSmith UI.
