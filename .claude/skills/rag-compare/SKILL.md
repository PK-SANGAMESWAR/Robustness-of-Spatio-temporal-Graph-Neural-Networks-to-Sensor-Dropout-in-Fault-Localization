---
name: rag-compare
description: Compare a GridSense candidate snapshot with a baseline snapshot using configs/registry.yaml and give PASS, REVIEW or FAIL with the reason for each metric, as Markdown and JSON.
---

# rag-compare

Use this skill with two snapshot paths: a baseline and a candidate.

1. Read `rag-for-dropout/EVAL.md` § 12 and `rag-for-dropout/configs/registry.yaml`.
2. Comparability: the golden set versions, the split, the k grid and the metric set must be equal. If not, the
   outcome is FAIL ("not comparable") and you list the differences.
3. Floors: check the baseline floors first. If the baseline misses a floor, tell the author. Do not accept the
   baseline as healthy.
4. Per metric, from the registry:
   - `gate`: candidate must meet the floor → else FAIL.
   - `guardrail`: regression larger than `tolerance` → REVIEW. Overlapping 95 % CIs → REVIEW (not clear).
     `tolerance_source: provisional` → REVIEW.
   - `informational`: report only.
5. Outcome: FAIL if any FAIL; else REVIEW if any REVIEW; else PASS.
6. Run `PYTHONPATH=src uv run python -m gridsense compare <baseline> <candidate>` when the module exists; else write
   NOT RUN with "pending: gridsense.eval.compare".
7. Write `rag-for-dropout/reports/compare/<baseline>_vs_<candidate>.md` and `.json`: one row per metric with the two
   values, the CIs, the delta, the role, the rule that applied and the result.
8. A PASS does not make the candidate the new baseline. Only the author accepts a baseline (`DONE.md` § 3).
