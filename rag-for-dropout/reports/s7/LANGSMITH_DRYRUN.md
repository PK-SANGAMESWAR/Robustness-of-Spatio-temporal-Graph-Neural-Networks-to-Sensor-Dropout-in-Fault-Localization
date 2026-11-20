# LangSmith dry run (S8, offline; nothing was sent)

Written for the author. Dataset `gridsense-retrieval`: **48 examples** would be uploaded (reviewed retrieval queries only). Payload sha256 `2827cee7c6adae53136655722d88a85cfc4203a2a0470ad995d062a1de81110a`.

| Group | Count | Why |
|---|---|---|
| In the payload | 48 | status reviewed; no golden source from the manuscript, `RESULTS.md` or the draft SOP; no span text, only references |
| Excluded: draft | 10 | drafts never go up (near-miss queries) |
| Excluded: sensitive source | 10 | the golden context comes from `paper1_rev1170` (unpublished manuscript), `results_md` or the draft SOP |
| Not part of this package | – | golden events (60, draft), safety items (25, draft), traces, experiments, online evaluators |

Redaction check on the payload (absolute paths, keys, addresses, user names, env names): **no hits**.

Config: `langsmith.enabled` = True, `project` = gridsense-eval, `upload_approved` = True. Tracing stays off (`LANGSMITH_TRACING=false`).

## What the author decides
1. Whether the payload may leave the machine, and the project name.
2. Whether the excluded items may go up (they quote the unpublished manuscript or internal results).
3. Whether golden events and safety items may go up after the author has reviewed them (they hold event facts and passage text).

To approve: set `langsmith.upload_approved: true`, `enabled: true` and `project` in `configs/eval.yaml`, then run `python -m gridsense.eval.langsmith_package upload --confirm-hash 2827cee7c6adae53136655722d88a85cfc4203a2a0470ad995d062a1de81110a`.
Inspect `reports/s7/langsmith_payload_preview.json` first: it is exactly what would be sent.

Excluded sensitive-source items: R-0014 (results_md), R-0015 (results_md), R-0016 (results_md), R-0025 (sop_illustrative_gridsense_review), R-0032 (sop_illustrative_gridsense_review), R-0035 (sop_illustrative_gridsense_review), R-0041 (results_md), R-0042 (results_md), R-0043 (results_md), R-0044 (paper1_rev1170)
