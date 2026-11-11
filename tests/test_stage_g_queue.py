"""Stage G queue: order, approval gate, lock, resume guard, collapse guard, dry run. No GPU, nothing is started.

The real queue is never run here: ``execute_step`` and the patch step are replaced by fakes, and every state file lives in a
temporary directory. The approval file of the real queue (``results/stage_g/STAGE_G_APPROVED``) is never created.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_stage_g_queue as Q  # noqa: E402


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "GPU_LOCKS", [tmp_path / "no_s44.lock", tmp_path / "no_watcher.lock"])
    p = Q.QueuePaths(tmp_path / "state")
    return p


def _approve(paths):
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.approval.write_text("approved by the author (unit test)\n")


@pytest.fixture()
def fake_exec(monkeypatch):
    calls = []

    def run(step, paths_):
        calls.append(step["id"])
        return {"status": "skipped_blocked", "reason": "x"} if step["kind"] == "blocked" else {"note": "fake"}
    monkeypatch.setattr(Q, "execute_step", run)
    monkeypatch.setattr(Q, "run_patch_and_tests", lambda p: {"patch": "fake", "tests": "fake"})
    monkeypatch.setattr(Q, "missing_inputs", lambda s: [])
    return calls


# ------------------------------------------------------------------------------ the plan
def test_order_is_the_registered_order_and_the_sealed_set_is_not_in_it():
    ids = [s["id"] for s in Q.build_steps()]
    assert ids[0] == "patch"
    first = lambda prefix: next(i for i, x in enumerate(ids) if x.startswith(prefix))      # noqa: E731
    assert first("patch") < first("mismatch_p1/") < first("b2a/train") < first("retrain_imp5/train") < first("s3/") < first("s4/") < first("s5/")
    assert ids.index("b2a/train") < ids.index("b2a/score_val") < ids.index("retrain_imp5/train")
    assert len([i for i in ids if i.startswith("mismatch_p1/")]) == 9                       # 3 tables x 3 R5 seeds
    assert len(set(ids)) == len(ids)
    assert not any("ood" in i for i in ids)
    steps = {s["id"]: s for s in Q.build_steps()}
    assert steps["s5/silent_failures"]["kind"] == "blocked"
    assert steps["b2a/train"]["seed"] == 42 and steps["retrain_imp5/train"]["config"].endswith("retrain_imp5.yaml")
    for t in Q.TABLES:
        assert steps[f"mismatch_p1/{t}/42"]["table_dir"] == f"operator_mismatch/{t}/operator" and steps[f"mismatch_p1/{t}/42"]["grid"] == "mismatch"
    assert steps["s3/r6i/42"]["shuffle_seed"] == 6 and steps["s3/b2a/42"]["split"] == "test"


def test_cost_estimate_uses_the_measured_rates():
    steps = {s["id"]: s for s in Q.build_steps()}
    assert steps["mismatch_p1/imp5/42"]["est_gpu_h"] == pytest.approx(2.98, abs=0.01)     # 24 forwards x 248,000 x 1.80 ms
    assert steps["b2a/train"]["est_gpu_h"] == pytest.approx(26.16)
    assert steps["b2a/score_val"]["est_gpu_h"] == pytest.approx(7.44, abs=0.01)
    assert steps["s3/r5/42"]["est_gpu_h"] == pytest.approx(7.41, abs=0.01)
    assert Q.total_gpu_hours(Q.build_steps()) == pytest.approx(sum(s["est_gpu_h"] for s in Q.build_steps()))


# ------------------------------------------------------------------------------ refusals
def test_refuses_without_the_approval_file_and_writes_nothing(paths):
    assert Q.run_queue(paths, Q.build_steps()) == 2
    assert not paths.root.exists()


def test_the_real_approval_file_is_not_created_by_anything_here():
    approval = Q.QueuePaths().approval
    if approval.exists():
        assert "Approved by the user" in approval.read_text()


def test_refuses_after_a_stop_marker_until_the_author_clears_it(paths, fake_exec):
    _approve(paths)
    paths.stop.write_text("{}")
    assert Q.run_queue(paths, Q.build_steps()) == 2 and fake_exec == []


def test_refuses_when_another_gpu_job_or_this_lock_is_alive(paths, fake_exec, monkeypatch, tmp_path):
    _approve(paths)
    (tmp_path / "no_s44.lock").write_text(str(os.getpid()))                   # a live PID
    assert Q.run_queue(paths, Q.build_steps()) == 2 and fake_exec == []
    (tmp_path / "no_s44.lock").unlink()
    paths.lock.write_text(str(os.getpid()))
    assert Q.run_queue(paths, Q.build_steps()) == 2 and fake_exec == []
    paths.lock.write_text("999999999")                                         # stale lock: dead PID
    assert Q.run_queue(paths, Q.build_steps()) == 0


# ------------------------------------------------------------------------------ the loop
def test_full_fake_run_writes_markers_complete_and_releases_the_lock(paths, fake_exec):
    _approve(paths)
    steps = Q.build_steps()
    assert Q.run_queue(paths, steps) == 0
    assert fake_exec == [s["id"] for s in steps if s["kind"] != "patch"]
    done = {p.stem for p in paths.done_dir.glob("*.json")}
    assert "s5__silent_failures" not in done and "patch" in done and len(done) == len(steps) - 1
    assert paths.complete.exists() and not paths.lock.exists() and paths.heartbeat.exists()
    log = json.loads(paths.progress.read_text())
    assert log[0]["stage"] == "queue_started" and log[-1]["stage"] == "queue_complete"
    assert Q.run_queue(paths, steps) == 2                                      # completed: refuses to run again


def test_resume_skips_done_steps_and_a_changed_spec_stops_the_queue(paths, fake_exec):
    _approve(paths)
    steps = Q.build_steps()
    paths.done_dir.mkdir(parents=True)
    for s in steps[:3]:
        (paths.done_dir / (s["id"].replace("/", "__") + ".json")).write_text(json.dumps({"spec_hash": Q.spec_hash(s)}))
    assert Q.run_queue(paths, steps) == 0
    assert steps[1]["id"] not in fake_exec and steps[3]["id"] in fake_exec
    # changed spec
    paths2 = Q.QueuePaths(paths.root.parent / "state2")
    _approve(paths2)
    paths2.done_dir.mkdir(parents=True)
    (paths2.done_dir / "mismatch_p1__tap_p1__42.json").write_text(json.dumps({"spec_hash": "0" * 64}))
    fake_exec.clear()
    assert Q.run_queue(paths2, steps) == 3
    stop = json.loads(paths2.stop.read_text())
    assert "spec changed" in stop["detail"] and stop["step"] == "mismatch_p1/tap_p1/42"


def test_collapse_stops_the_queue_with_a_marker_and_later_steps_do_not_run(paths, monkeypatch):
    _approve(paths)
    ran = []

    def run(step, p):
        ran.append(step["id"])
        if step["id"] == "b2a/train":
            raise Q.Collapsed("b2a/train: collapse guard fired")
        return {"note": "fake"}
    monkeypatch.setattr(Q, "execute_step", run)
    monkeypatch.setattr(Q, "run_patch_and_tests", lambda p: {"patch": "fake", "tests": "fake"})
    monkeypatch.setattr(Q, "missing_inputs", lambda s: [])
    assert Q.run_queue(paths, Q.build_steps()) == 3
    assert json.loads(paths.stop.read_text())["reason"] == "collapsed" and ran[-1] == "b2a/train"
    assert "retrain_imp5/train" not in ran and not paths.complete.exists() and not paths.lock.exists()
    assert Q.run_queue(paths, Q.build_steps()) == 2                            # not restarted without the author


def test_an_exception_stops_the_queue_and_missing_inputs_are_an_error(paths, monkeypatch):
    _approve(paths)
    monkeypatch.setattr(Q, "run_patch_and_tests", lambda p: {"patch": "fake", "tests": "fake"})
    monkeypatch.setattr(Q, "missing_inputs", lambda s: ["x/checkpoint_best.pt"] if s["id"] == "mismatch_p1/tap_p1/42" else [])
    assert Q.run_queue(paths, Q.build_steps()) == 3
    assert "inputs missing" in json.loads(paths.stop.read_text())["detail"]
    paths.stop.unlink()
    monkeypatch.setattr(Q, "missing_inputs", lambda s: [])
    monkeypatch.setattr(Q, "execute_step", lambda s, p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert Q.run_queue(paths, Q.build_steps()) == 3 and json.loads(paths.stop.read_text())["reason"] == "exception"


# ------------------------------------------------------------------------------ training-step guards
def _train_step(tmp_path, name="b2a"):
    return {"id": f"{name}/train", "kind": "train", "config": f"config/stage_g/{name}.yaml", "seed": 42,
            "run_dir": str(tmp_path / "run"), "est_gpu_h": 26.16}


def _fake_run_dir(tmp_path, collapsed, mutate=None):
    from fault_localization.experiments.config import load_config, validate_config
    cfg = validate_config(load_config("config/stage_g/b2a.yaml"))
    cfg["seed"] = 42
    saved = json.loads(json.dumps(cfg))
    if mutate:
        mutate(saved)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(saved))
    (run / "metrics.json").write_text(json.dumps({"best_validation_graph_f1": 0.5,
                                                   "collapse_abort": {"epoch": 3, "triggered": collapsed}}))
    return run


def test_a_completed_collapsed_run_stops_the_queue(tmp_path):
    _fake_run_dir(tmp_path, collapsed=True)
    with pytest.raises(Q.Collapsed):
        Q.run_train(_train_step(tmp_path))


def test_a_completed_healthy_run_is_resumed_not_retrained(tmp_path):
    _fake_run_dir(tmp_path, collapsed=False)
    assert Q.run_train(_train_step(tmp_path))["note"].startswith("resumed")


def test_a_saved_config_that_differs_from_the_frozen_one_stops(tmp_path):
    _fake_run_dir(tmp_path, collapsed=False, mutate=lambda c: c["pc_loss"].__setitem__("lambda2", 0.2))
    with pytest.raises(Q.Stop, match="differs"):
        Q.run_train(_train_step(tmp_path))


def test_a_partial_run_directory_stops(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(Q.Stop, match="partial"):
        Q.run_train(_train_step(tmp_path))


def test_spec_hash_follows_the_config_file(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n")
    step = {"id": "x/train", "kind": "train", "config": str(cfg), "seed": 1, "run_dir": "r", "est_gpu_h": 1.0}
    h1 = Q.spec_hash(step)
    cfg.write_text("a: 2\n")
    assert Q.spec_hash(step) != h1


# ------------------------------------------------------------------------------ patch state and dry run
def test_patch_state_is_known():
    assert Q.patch_state() in {"applied", "applicable"}


def test_dry_run_uses_no_gpu_takes_no_lock_and_creates_no_approval(paths, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert Q.main(["--dry-run", "--state-dir", str(paths.root)]) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not paths.approval.exists() and not paths.lock.exists() and not paths.stop.exists()
    rep = json.loads(paths.dry_report.read_text())
    assert rep["checks"]["sealed_set_in_queue"] is False and rep["checks"]["b2a_cpu_forward_backward"]["head_has_grad"]
    assert any("STAGE_G_APPROVED" in b for b in rep["checks"]["preconditions_a_real_run_would_report"])
    assert all(v["shape_matches_nominal"] and v["complete"] and v["n_nan"] == 0 for v in rep["checks"]["mismatch_tables"].values())
