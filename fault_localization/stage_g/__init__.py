"""Stage G (``PC-STGNN/08_STAGE_G_eval.md``) code: scoring that saves predictions, the one-run
sealed-set pipeline, and helpers for the model-mismatch study.

Everything here is new code.  Files that the running S4.4 queue or the G-TF2 watcher import
(``scripts/score_s4_4_gate4_run.py``, ``experiments/runner.py``, ``models/pc_stgnn.py``,
``models/common.py``, ``physics/residual.py``) are imported read-only and are not edited.
"""
