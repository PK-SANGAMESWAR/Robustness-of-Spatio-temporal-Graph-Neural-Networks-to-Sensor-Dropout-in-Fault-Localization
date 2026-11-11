"""Stage G GPU queue (``PC-STGNN/stage_g_plan_draft.md`` section 5). **Written, not started.**

    uv run python scripts/run_stage_g_queue.py --dry-run     # no GPU, no lock, no approval needed; prints the plan and checks inputs
    uv run python scripts/run_stage_g_queue.py               # the real run

The real run refuses to start unless ALL of these hold:
  * the file ``results/stage_g/STAGE_G_APPROVED`` exists (the author creates it; this script never does),
  * no ``queue_STOPPED.json`` from an earlier stop is present (the author removes it),
  * no other GPU job is alive: the S4.4 queue lock, the G-TF2 watcher lock, this queue's own lock.

Order (the sealed set is NOT in this queue; it has its own one-run pipeline, ``scripts/run_stage_g_ood.py``):
  0. apply ``patches/stage_g/b2a_runner_hooks.patch`` and run the unit tests (on failure the patch is reverted and the queue stops);
  1. mismatch Phase 1: R5 seeds 42/43/44 x 3 perturbed tables, validation split, 8 cells;
  2. B2a: train (seed 42, epoch-3 collapse guard) and score on validation;
  3. mismatch retrain: R5 on the impedance +-5 % table (seed 42, collapse guard) and score on validation;
  4. test-split grid: S3 random dropout x noise, S4 persistent / contiguous / targeted, S5 silent failures.
     S5 is a registered step that is *blocked*: G-TF2 on the test split has no pre-registration (see ``blocked_reason``).

Safety (same pattern as ``run_s4_4_queue.py``): every step has a done-marker keyed on a hash of its spec (resume skips it; a changed
spec stops the queue); a lock file; a heartbeat file; the S4.3c collapse guard on both training runs (a collapsed run is recorded and
stops the queue; it is not retried); any exception stops the queue with a marker. Nothing restarts itself.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

if "--dry-run" in sys.argv:                      # before torch is imported anywhere
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PATCH = REPO / "patches/stage_g/b2a_runner_hooks.patch"
PATCH_TESTS = ["tests/test_b2a_rhat.py", "tests/test_b2a_hooks.py", "tests/test_stage_g_queue.py",
               "tests/test_pc_stgnn.py", "tests/test_s4_4_remaining_arms.py", "tests/test_stage_g_scoring.py"]
GPU_LOCKS = [REPO / "results/s4_4/queue.lock", REPO / "results/stage_f/gtf2/watcher.lock"]
TABLES = ("tap_p1", "tap_m1", "imp5")
SEEDS = (42, 43, 44)
TRAIN_ROOT = "results/stage_g/train"
# measured rates (PC-STGNN/stage_g_plan_draft.md section 5): GPU hours
H_TRAIN_R5, H_SCORE_FULL_VAL = 26.16, 7.44
C_R5_MS, C_R0_MS = 1.80, 0.325                   # ms per window-forward
N_VAL, N_TEST, N_FWD_FULL, N_FWD_MISMATCH, N_FWD_POLICY = 248_000, 247_000, 60, 24, 12


@dataclass
class QueuePaths:
    root: Path = REPO / "results/stage_g"

    @property
    def approval(self): return self.root / "STAGE_G_APPROVED"
    @property
    def lock(self): return self.root / "queue.lock"
    @property
    def progress(self): return self.root / "queue_progress.json"
    @property
    def stop(self): return self.root / "queue_STOPPED.json"
    @property
    def heartbeat(self): return self.root / "queue_heartbeat.json"
    @property
    def done_dir(self): return self.root / "queue_done"
    @property
    def complete(self): return self.root / "queue_complete.json"
    @property
    def dry_report(self): return self.root / "queue_dry_run.json"


class Stop(RuntimeError):
    """The queue stops with a marker and exit code 3; nothing is retried."""


class Collapsed(Stop):
    pass


# ---------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------


def _fwd_hours(c_ms: float, n_windows: int, n_fwd: int) -> float:
    return c_ms * 1e-3 * n_windows * n_fwd / 3600


def _score(step_id, run_dir, seed, split, grid, out_name, c_ms, *, policy="random", table=None, shuffle=None,
           max_windows=None) -> dict:
    n = N_TEST if split == "test" else N_VAL
    n_fwd = {"full": N_FWD_FULL, "mismatch": N_FWD_MISMATCH, "policy": N_FWD_POLICY}[grid]
    return {"id": step_id, "kind": "score", "run_dir": run_dir, "seed": seed, "split": split, "grid": grid,
            "policy": policy, "table_dir": table, "shuffle_seed": shuffle, "out_name": out_name,
            "est_gpu_h": round(_fwd_hours(c_ms, n, n_fwd), 2)}


def build_steps() -> list[dict]:
    steps: list[dict] = [{"id": "patch", "kind": "patch", "est_gpu_h": 0.0}]
    r5 = lambda s: f"results/s4_4/r5/pcstgnn/seed_{s:05d}"           # noqa: E731
    r0n = lambda s: f"results/s4_4/r0n/pcstgnn/seed_{s:05d}"         # noqa: E731
    r2 = lambda s: f"results/s4_4/r2/pcstgnn/seed_{s:05d}"           # noqa: E731
    # 1. mismatch Phase 1 (inference only)
    for t in TABLES:
        for s in SEEDS:
            steps.append(_score(f"mismatch_p1/{t}/{s}", r5(s), s, "val", "mismatch", f"r5_{t}", C_R5_MS,
                                table=f"operator_mismatch/{t}/operator"))
    # 2. B2a, 3. retrain
    for name, cfg in (("b2a", "config/stage_g/b2a.yaml"), ("retrain_imp5", "config/stage_g/retrain_imp5.yaml")):
        run_dir = f"{TRAIN_ROOT}/{name}/pcstgnn/seed_00042"
        steps.append({"id": f"{name}/train", "kind": "train", "config": cfg, "seed": 42, "run_dir": run_dir,
                      "est_gpu_h": H_TRAIN_R5 + 0.0})
        steps.append(_score(f"{name}/score_val", run_dir, 42, "val", "full", name, C_R5_MS))
    # 4. test split
    for s in SEEDS:                                                     # S3
        steps.append(_score(f"s3/r0n/{s}", r0n(s), s, "test", "full", "r0n", C_R0_MS))
    for s in SEEDS:
        steps.append(_score(f"s3/r5/{s}", r5(s), s, "test", "full", "r5", C_R5_MS))
    for s in SEEDS:
        steps.append(_score(f"s3/r2/{s}", r2(s), s, "test", "full", "r2", C_R0_MS))
    for s in SEEDS:                                                     # R6^i: R5 weights, shuffled table, 8 cells
        steps.append(_score(f"s3/r6i/{s}", r5(s), s, "test", "mismatch", "r6i", C_R5_MS, shuffle=6))
    steps.append(_score("s3/b2a/42", f"{TRAIN_ROOT}/b2a/pcstgnn/seed_00042", 42, "test", "full", "b2a", C_R5_MS))
    for pol in ("persistent", "contiguous", "targeted"):               # S4
        for s in SEEDS:
            steps.append(_score(f"s4/{pol}/r5/{s}", r5(s), s, "test", "policy", f"r5_{pol}", C_R5_MS, policy=pol))
            steps.append(_score(f"s4/{pol}/r0n/{s}", r0n(s), s, "test", "policy", f"r0n_{pol}", C_R0_MS, policy=pol))
    steps.append({"id": "s5/silent_failures", "kind": "blocked", "est_gpu_h": 0.0,
                  "blocked_reason": "G-TF2 on the test split is outside its pre-registration (stage_f_gtf2_preregistration.md lists "
                                    "the test split as 'not in scope') and scripts/run_gtf.py scores validation half B only. It needs "
                                    "its own pre-registration and code before it can run."})
    return steps


def spec_hash(step: dict) -> str:
    spec = dict(step)
    if step["kind"] == "train":
        spec["config_sha256"] = hashlib.sha256((REPO / step["config"]).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def total_gpu_hours(steps) -> float:
    return round(sum(s["est_gpu_h"] for s in steps), 2)


# ---------------------------------------------------------------------------------------------
# State files, lock, heartbeat
# ---------------------------------------------------------------------------------------------


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_alive(lock: Path) -> bool:
    try:
        os.kill(int(Path(lock).read_text().split()[0]), 0)
        return True
    except (FileNotFoundError, ValueError, IndexError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def append_progress(paths: QueuePaths, entry: dict) -> None:
    entry = {**entry, "utc": utc()}
    data = json.loads(paths.progress.read_text()) if paths.progress.exists() else []
    data.append(entry)
    paths.progress.write_text(json.dumps(data, indent=1) + "\n")


def write_stop(paths: QueuePaths, reason: str, detail: str, step_id: str) -> None:
    paths.stop.write_text(json.dumps({"reason": reason, "detail": detail, "step": step_id, "utc": utc()}, indent=2) + "\n")


def preconditions(paths: QueuePaths, *, dry_run: bool) -> list[str]:
    bad = []
    if not paths.approval.exists():
        bad.append(f"{paths.approval.relative_to(REPO) if paths.approval.is_relative_to(REPO) else paths.approval} is missing (the author creates it)")
    if paths.stop.exists():
        bad.append(f"{paths.stop.name} exists: an earlier stop has not been cleared by the author")
    if paths.complete.exists():
        bad.append(f"{paths.complete.name} exists: the queue already completed")
    if _pid_alive(paths.lock):
        bad.append("this queue's lock is held by a live process")
    for lock in GPU_LOCKS:
        if _pid_alive(lock):
            bad.append(f"a GPU job is alive: {lock.name}")
    return bad


class Heartbeat:
    def __init__(self, paths: QueuePaths, period: float = 60.0):
        self.paths, self.period, self.current = paths, period, "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _write(self):
        self.paths.heartbeat.write_text(json.dumps({"pid": os.getpid(), "utc": utc(), "step": self.current}) + "\n")

    def _loop(self):
        while not self._stop.is_set():
            self._write()
            self._stop.wait(self.period)

    def start(self):
        self._write()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._write()


# ---------------------------------------------------------------------------------------------
# Patch step
# ---------------------------------------------------------------------------------------------


def patch_state() -> str:
    """'applied', 'applicable' or 'conflict' for ``patches/stage_g/b2a_runner_hooks.patch``."""
    def ok(*args):
        return subprocess.run(["git", "-C", str(REPO), "apply", *args, str(PATCH)], capture_output=True).returncode == 0
    if ok("--check", "-R"):
        return "applied"
    return "applicable" if ok("--check") else "conflict"


def run_patch_and_tests(paths: QueuePaths) -> dict:
    state = patch_state()
    if state == "conflict":
        raise Stop("the B2a patch does not apply to runner.py (it changed since the patch was written)")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    r = subprocess.run([sys.executable, "-m", "pytest", *PATCH_TESTS, "-q", "-m", "not slow", "-p", "no:cacheprovider"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise Stop("unit tests failed before applying the B2a patch:\n" + (r.stdout + r.stderr)[-1500:])
    if state == "applicable":
        subprocess.run(["git", "-C", str(REPO), "apply", str(PATCH)], check=True)
    return {"patch": "applied now" if state == "applicable" else "was already applied", "tests": r.stdout.strip().splitlines()[-1]}


# ---------------------------------------------------------------------------------------------
# Step execution (heavy imports happen here, after the patch)
# ---------------------------------------------------------------------------------------------


_DATASETS: dict = {}


def _dataset(split: str):
    from fault_localization.data.dataset import WindowGraphDataset
    if split not in _DATASETS:
        _DATASETS[split] = WindowGraphDataset("dataset_fullscale_50", split)
    return _DATASETS[split]


def run_train(step: dict) -> dict:
    import yaml
    import importlib
    import run_s4_4_remaining_queue as rq
    from fault_localization.experiments.config import load_config, validate_config
    runner = importlib.import_module("fault_localization.experiments.runner")
    cfg = validate_config(load_config(step["config"]))
    cfg["seed"] = step["seed"]
    run_dir = Path(step["run_dir"])
    if (run_dir / "metrics.json").exists():
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        diffs = rq._dict_diff(cfg, saved)
        if diffs:
            raise Stop("a completed run exists whose saved config differs from the frozen config: " + "; ".join(diffs[:6]))
        note = "resumed: completed training found"
    elif run_dir.exists():
        raise Stop(f"{run_dir} exists without metrics.json (partial run); the author decides")
    else:
        rq._run_with_heartbeat(run_dir, runner.run, cfg)
        note = "trained"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics.get("collapse_abort", {}).get("triggered"):
        raise Collapsed(f"{step['id']}: collapse guard fired {metrics['collapse_abort']}")
    return {"note": note, "best_validation_graph_f1": metrics.get("best_validation_graph_f1")}


def _check_network_only_equals_r5(summary_cells: dict, seed: int, grid_cells) -> None:
    ref = json.loads((REPO / f"results/s4_4/r5/pcstgnn/seed_{seed:05d}/scoring_full_grid.json").read_text())["cells"]
    for k, s in grid_cells:
        a = summary_cells[f"k{k}_sigma{s:g}"]["f1_network_only_mean"]
        b = ref[f"k{k}_sigma{float(s)}"]["f1_network_only_mean"]
        if abs(a - b) > 1e-9:
            raise Stop(f"self-check failed: network-only F1 differs from R5 at k={k} sigma={s}: {a} vs {b}")


def run_score(step: dict) -> dict:
    import torch
    from fault_localization.stage_g import scoring as S
    ds = _dataset(step["split"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = REPO / step["run_dir"] if not Path(step["run_dir"]).is_absolute() else Path(step["run_dir"])
    table = REPO / step["table_dir"] if step["table_dir"] else None
    model, _cfg = S.load_model(run_dir, ds.edge_index, device, table_dir=table, shuffle_seed=step["shuffle_seed"])
    res = S.score_dataset(model, ds, S.GRIDS[step["grid"]], device=device, policy=step["policy"], progress_every=500)
    meta = {"arm": step["out_name"], "seed": step["seed"], "split": step["split"], "grid": step["grid"],
            "policy": step["policy"], "table_dir": step["table_dir"], "shuffle_seed": step["shuffle_seed"],
            "checkpoint_sha256": S.sha256_file(run_dir / "checkpoint_best.pt"), "data_root": "dataset_fullscale_50",
            "resistances": [0.1, 1.0, 10.0], "queue_step": step["id"]}
    out_dir = REPO / "results/stage_g" / step["split"] / step["out_name"]
    path = S.write_summary(out_dir, f"seed_{step['seed']:05d}", res, meta, resistances=(0.1, 1.0, 10.0))
    if step["id"].startswith("mismatch_p1/"):
        summary = json.loads(Path(path).read_text())
        _check_network_only_equals_r5(summary["cells"], step["seed"], S.GRIDS["mismatch"])
    S.release(model)
    return {"summary": str(path.relative_to(REPO)), "seconds": res.seconds}


def execute_step(step: dict, paths: QueuePaths) -> dict:
    kind = step["kind"]
    if kind == "train":
        return run_train(step)
    if kind == "score":
        return run_score(step)
    if kind == "blocked":
        return {"status": "skipped_blocked", "reason": step["blocked_reason"]}
    raise ValueError(kind)


def missing_inputs(step: dict) -> list[str]:
    """Files a step needs *before* it runs. Outputs of earlier queue steps are not listed."""
    need = []
    if step["kind"] == "score":
        produced_by_queue = step["run_dir"].startswith(TRAIN_ROOT)
        if not produced_by_queue and not (REPO / step["run_dir"] / "checkpoint_best.pt").exists():
            need.append(f"{step['run_dir']}/checkpoint_best.pt")
        if step["table_dir"] and not (REPO / step["table_dir"] / "mu.npy").exists():
            need.append(f"{step['table_dir']}/mu.npy")
    if step["kind"] == "train" and not (REPO / step["config"]).exists():
        need.append(step["config"])
    return need


# ---------------------------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------------------------


def run_queue(paths: QueuePaths, steps: list[dict]) -> int:
    bad = preconditions(paths, dry_run=False)
    if bad:
        print("REFUSED: " + "; ".join(bad), file=sys.stderr)
        return 2
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.done_dir.mkdir(parents=True, exist_ok=True)
    paths.lock.write_text(f"{os.getpid()}\n")
    atexit.register(lambda: paths.lock.unlink(missing_ok=True))
    hb = Heartbeat(paths)
    hb.start()
    append_progress(paths, {"stage": "queue_started", "pid": os.getpid(), "n_steps": len(steps), "approval": paths.approval.read_text()[:200]})
    try:
        for i, step in enumerate(steps, 1):
            hb.current = step["id"]
            marker = paths.done_dir / (step["id"].replace("/", "__") + ".json")
            h = spec_hash(step)
            if marker.exists():
                if json.loads(marker.read_text())["spec_hash"] != h:
                    raise Stop(f"{step['id']}: done-marker exists but the step spec changed (frozen settings differ)")
                append_progress(paths, {"stage": "skipped_done", "step": step["id"], "position": i})
                continue
            append_progress(paths, {"stage": "step_started", "step": step["id"], "position": i})
            miss = missing_inputs(step)
            if miss:
                raise Stop(f"{step['id']}: inputs missing: {miss}")
            t0 = time.perf_counter()
            if step["kind"] == "patch":
                result = run_patch_and_tests(paths)
            else:
                result = execute_step(step, paths)
            seconds = time.perf_counter() - t0
            append_progress(paths, {"stage": "step_done", "step": step["id"], "position": i, "seconds": seconds, **{
                k: v for k, v in result.items() if k in ("status", "reason", "note", "tests", "patch")}})
            if result.get("status") != "skipped_blocked":
                marker.write_text(json.dumps({"spec_hash": h, "utc": utc(), "seconds": seconds, "result": result}, indent=2) + "\n")
        paths.complete.write_text(json.dumps({"utc": utc(), "n_steps": len(steps)}) + "\n")
        append_progress(paths, {"stage": "queue_complete"})
        return 0
    except Collapsed as exc:
        write_stop(paths, "collapsed", str(exc), hb.current)
        append_progress(paths, {"stage": "queue_stopped", "reason": "collapsed", "step": hb.current})
        print(f"QUEUE STOPPED (collapse guard): {exc}", file=sys.stderr)
        return 3
    except Stop as exc:
        write_stop(paths, "stop", str(exc), hb.current)
        append_progress(paths, {"stage": "queue_stopped", "reason": "stop", "step": hb.current})
        print(f"QUEUE STOPPED: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:                                   # noqa: BLE001
        write_stop(paths, "exception", traceback.format_exc()[-2000:], hb.current)
        append_progress(paths, {"stage": "queue_stopped", "reason": "exception", "step": hb.current, "error": repr(exc)})
        print(f"QUEUE STOPPED (exception): {exc!r}", file=sys.stderr)
        return 3
    finally:
        hb.stop()
        paths.lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------------


def dry_run(paths: QueuePaths, steps: list[dict]) -> int:
    """No GPU (CUDA_VISIBLE_DEVICES is empty), no lock, no approval file created. Checks and prints the plan."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report: dict = {"utc": utc(), "n_steps": len(steps), "estimated_gpu_hours": total_gpu_hours(steps), "checks": {}}
    checks = report["checks"]
    checks["preconditions_a_real_run_would_report"] = preconditions(paths, dry_run=True)
    checks["patch_state"] = patch_state()
    checks["order"] = [s["id"] for s in steps]
    checks["sealed_set_in_queue"] = any("ood" in s["id"] for s in steps)
    checks["blocked_steps"] = [s["id"] for s in steps if s["kind"] == "blocked"]
    miss = {s["id"]: missing_inputs(s) for s in steps if missing_inputs(s)}
    checks["missing_inputs"] = miss
    checks["spec_hashes_computed"] = len({spec_hash(s) for s in steps}) == len(steps)
    # configs validate and the tables load on CPU; a B2a forward/backward runs on CPU with the real table
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from fault_localization.experiments.config import load_config, validate_config
    for name in ("b2a", "retrain_imp5"):
        validate_config(load_config(f"config/stage_g/{name}.yaml"))
    checks["configs_validate"] = True
    from fault_localization.physics import residual as R, table as T
    from fault_localization.physics.observation import load_norm_stats
    edge = np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
    stats = load_norm_stats("dataset_fullscale_50")
    nominal = T.load_table(REPO / "operator")
    for t in TABLES:
        op = REPO / "operator_mismatch" / t / "operator"
        tbl = T.load_table(op)
        grid = json.loads((op / "grid.json").read_text())
        checks.setdefault("mismatch_tables", {})[t] = {"shape_matches_nominal": tuple(tbl["mu"].shape) == tuple(nominal["mu"].shape),
                                                       "complete": bool(grid.get("complete")), "n_nan": int(grid.get("n_nan", -1)),
                                                       "n_non_converged": int(grid.get("n_non_converged", -1))}
    from fault_localization.models.pc_stgnn_rhat import PCSTGNNRHat, b2a_batch_loss
    tab = R.flatten_table(nominal["mu"], nominal["mu0"])
    torch.manual_seed(0)
    model = PCSTGNNRHat(edge, tab, stats, use_fusion=True, aggregator="max", num_layers=4, sigma_eps=7.255549105131452e-4)
    b = 4
    x = torch.randn(b, 25, 20, 6)
    x[..., 3:] = 1.0
    batch = {"x": x, "x_clean": x.clone(), "drop_mask": torch.zeros(b, 25, 3), "y": torch.tensor([3, 0, 7, 11]),
             "window_id": [f"e{i}_w00{21 + i}" for i in range(b)], "metadata": [{"resistance": r} for r in ("1.0", "1.0", "10.0", "0.1")]}
    parts = b2a_batch_loss(model, batch, torch.device("cpu"))
    parts["total"].backward()
    checks["b2a_cpu_forward_backward"] = {"loss_finite": bool(torch.isfinite(parts["total"])),
                                          "n_parameters": int(sum(p.numel() for p in model.parameters())),
                                          "head_has_grad": bool(model.rhat_head.weight.grad is not None)}
    checks["approval_file_created_by_dry_run"] = paths.approval.exists()
    checks["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.dry_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_steps": report["n_steps"], "estimated_gpu_hours": report["estimated_gpu_hours"],
                      "real_run_would_refuse_because": checks["preconditions_a_real_run_would_report"],
                      "patch_state": checks["patch_state"], "missing_inputs": miss, "blocked": checks["blocked_steps"],
                      "sealed_set_in_queue": checks["sealed_set_in_queue"],
                      "b2a_cpu_forward_backward": checks["b2a_cpu_forward_backward"]}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state-dir", default=None, help="tests only")
    args = ap.parse_args(argv)
    paths = QueuePaths(Path(args.state_dir)) if args.state_dir else QueuePaths()
    steps = build_steps()
    if args.dry_run:
        return dry_run(paths, steps)
    return run_queue(paths, steps)


if __name__ == "__main__":
    raise SystemExit(main())
