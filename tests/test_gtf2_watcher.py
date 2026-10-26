"""The G-TF2 watcher: queue-state logic, locks and the step plan (files only; no GPU, no subprocess, no queue)."""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_gtf2_after_queue as W  # noqa: E402


def _patch(monkeypatch, tmp_path, lock_pid=None, hist=None, stop=False):
    monkeypatch.setattr(W, "QUEUE_LOCK", tmp_path / "queue.lock")
    monkeypatch.setattr(W, "QUEUE_PROGRESS", tmp_path / "queue_progress.json")
    monkeypatch.setattr(W, "QUEUE_STOP", tmp_path / "queue_STOPPED.json")
    if lock_pid is not None:
        (tmp_path / "queue.lock").write_text(str(lock_pid))
    if hist is not None:
        (tmp_path / "queue_progress.json").write_text(json.dumps(hist))
    if stop:
        (tmp_path / "queue_STOPPED.json").write_text("{}")


DONE = [{"stage": "queue_started", "pid": 1}, {"stage": "queue_complete"}]


def test_gtf2_watcher_waits_while_the_queue_process_is_alive(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, lock_pid=os.getpid(), hist=DONE[:1])
    assert W.queue_state() == "running"


def test_gtf2_watcher_starts_only_after_queue_complete_and_exit(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, lock_pid=999999999, hist=DONE)           # lock PID dead, queue_complete written
    assert W.queue_state() == "complete"
    _patch(monkeypatch, tmp_path, lock_pid=os.getpid(), hist=DONE)         # complete but the process is still alive
    assert W.queue_state() == "running"


def test_gtf2_watcher_does_not_start_if_the_queue_stopped_or_died(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, lock_pid=999999999, hist=DONE, stop=True)
    assert W.queue_state() == "stopped"                                    # the STOPPED marker wins over everything
    (tmp_path / "queue_STOPPED.json").unlink()
    (tmp_path / "queue_progress.json").write_text(json.dumps([{"stage": "queue_started"}, {"stage": "training_done"}]))
    assert W.queue_state() == "died"                                       # process gone, no queue_complete
    (tmp_path / "queue_progress.json").write_text(json.dumps([{"stage": "queue_complete"}, {"stage": "queue_started"}]))
    assert W.queue_state() == "died"                                       # a queue_complete of an OLDER run does not count
    (tmp_path / "queue_progress.json").unlink()
    assert W.queue_state() == "died"


def test_gtf2_watcher_gpu_lock_allows_one_holder_and_replaces_a_stale_lock(monkeypatch, tmp_path):
    lock = tmp_path / "gpu_job.lock"
    lock.write_text(str(os.getppid()))                                     # a live process holds it
    assert W._lock_take(lock) is False
    lock.write_text("999999999")                                           # stale
    assert W._lock_take(lock) is True and lock.read_text() == str(os.getpid())


def test_gtf2_watcher_plan_is_gtf2_then_residual_only_and_thresholds_are_cpu():
    plan = W.steps()
    names = [n for n, _, _ in plan]
    assert names == ["gtf2_thresholds", "gtf2_score_42", "gtf2_score_43", "gtf2_score_44", "gtf2_summarize",
                     "gtf2r_score_42", "gtf2r_score_43", "gtf2r_score_44", "gtf2r_summarize"]
    gpu = {n: g for n, _, g in plan}
    assert gpu["gtf2_thresholds"] is False and gpu["gtf2_summarize"] is False and gpu["gtf2r_summarize"] is False
    assert all(gpu[n] for n in names if "_score_" in n)
    assert all("--gate" in a for _, a, _ in plan) and all("full" not in a for _, a, _ in plan)
