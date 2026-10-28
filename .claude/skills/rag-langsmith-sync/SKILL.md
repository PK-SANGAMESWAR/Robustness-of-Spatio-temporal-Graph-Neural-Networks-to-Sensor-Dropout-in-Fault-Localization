---
name: rag-langsmith-sync
description: Push reviewed GridSense golden sets to LangSmith datasets, run offline experiments with the local evaluators, set tracing and online evaluators on sampled traces, with redaction and author approval before the first upload.
---

# rag-langsmith-sync

Use this skill with an action: `push-datasets`, `experiment`, `tracing`, `online-evaluators`.

1. Read `rag-for-dropout/EVAL.md` § 13 and `rag-for-dropout/claude-rag.md` § 4 rule 10.
2. Check `configs/eval.yaml` `langsmith`. If `enabled: false`, `project: null` or `upload_approved: false` (O8),
   stop and ask. Sending data to LangSmith publishes it outside this machine.
3. The API key comes from `LANGSMITH_API_KEY` through `gridsense.settings`. Never print it or write it to a file.
4. Redaction before upload: no API keys, no local absolute paths, no environment values, no user names. Use repo
   relative paths. Check one payload by hand before the first batch.
5. `push-datasets`: push only `status: reviewed` items. Dataset name `gridsense-<set>`; version tag = golden version.
   Never push drafts. Record the dataset ID in `EXPERIMENT_LOG.md`.
6. `experiment`: run the same evaluators as the local suite (deterministic checks + DeepEval metrics wrapped as
   LangSmith evaluators). Run `rag-cost-check` first. Write the experiment URL into the local snapshot. The local
   snapshot is the source of truth; LangSmith is a mirror.
7. `tracing`: one project per environment. Each LangGraph node is a run with metadata (model, index, policy, prompt
   versions). The gate outcome and approval state are recorded, not changed.
8. `online-evaluators`: sample rate from config. LLM judge rules: faithfulness, no-aggregate-confidence. Code rules:
   field consistency, control-command pattern. Feedback per trace. The traffic is a replay stream of `val`/`test`
   events, not a live grid; write this in every result and dashboard description.
9. Report the links and the counts to the author.
