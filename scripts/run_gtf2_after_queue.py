"""Start G-TF2, then the residual-only arm, after the S4.4 queue has finished. Reads files only while it waits.

Waiting (no GPU, no torch import, no write to the queue's files): poll ``results/s4_4/queue.lock`` (the queue's PID) and
``results/s4_4/queue_progress.json``. The watcher starts the GPU work only when ALL of these hold:

  1. ``results/s4_4/queue_STOPPED.json`` does not exist (a stopped queue means the author decides; the watcher exits);
  2. the queue wrote the stage ``queue_complete`` after its last ``queue_started`` and its process has exited;
  3. the readiness guard ``run_gtf.check_ready_gtf2()`` returns ready (pre-registration and ``07_STAGE_F_gate.md``
     committed and unmodified; S5.1 raw scores and the three R5 checkpoints present);
  4. it holds the GPU-job lock ``results/stage_f/gtf2/gpu_job.lock`` (one GPU job at a time; refuses while the queue's lock
     belongs to a live process).

Steps, each a subprocess of ``scripts/run_gtf.py``; the first failure stops the chain (nothing is retried):
  thresholds (CPU) -> G-TF2 score seeds 42, 43, 44 (GPU) -> G-TF2 summarize -> residual-only score seeds 42, 43, 44 (GPU)
  -> residual-only summarize.

``--dry-run`` evaluates the conditions and the guard and prints the plan. It never launches a step and hides the GPU.
The watcher never stops, restarts or edits the queue.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUEUE_PROGRESS = REPO / "results/s4_4/queue_progress.json"
QUEUE_STOP = REPO / "results/s4_4/queue_STOPPED.json"
QUEUE_LOCK = REPO / "results/s4_4/queue.lock"
OUT = REPO / "results/stage_f/gtf2"
WATCH_LOCK = OUT / "watcher.lock"
GPU_LOCK = OUT / "gpu_job.lock"
PROGRESS = OUT / "watcher_progress.json"
LOG = OUT / "watcher.log"
PY = REPO / ".venv/bin/python"
GTF = REPO / "scripts/run_gtf.py"
SEEDS = (42, 43, 44)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def queue_state() -> str:
    """'stopped' | 'running' | 'complete' | 'died'. Reads files only."""
    if QUEUE_STOP.exists():
        return "stopped"
    pid = read_pid(QUEUE_LOCK)
    if pid is not None and pid_alive(pid):
        return "running"
    try:
        hist = json.loads(QUEUE_PROGRESS.read_text())
    except (OSError, ValueError):
        return "died"
    last_start = max((i for i, e in enumerate(hist) if e.get("stage") == "queue_started"), default=-1)
    done = any(e.get("stage") == "queue_complete" for e in hist[last_start + 1:])
    return "complete" if (done and last_start >= 0) else "died"


def steps() -> list[tuple[str, list[str], bool]]:
    """(name, argv after the interpreter, uses_gpu)"""
    g = str(GTF)
    plan = [("gtf2_thresholds", [g, "--gate", "gtf2", "thresholds"], False)]
    plan += [(f"gtf2_score_{s}", [g, "--gate", "gtf2", "score", "--seed", str(s)], True) for s in SEEDS]
    plan += [("gtf2_summarize", [g, "--gate", "gtf2", "summarize"], False)]
    plan += [(f"gtf2r_score_{s}", [g, "--gate", "gtf2r", "score", "--seed", str(s)], True) for s in SEEDS]
    plan += [("gtf2r_summarize", [g, "--gate", "gtf2r", "summarize"], False)]
    return plan


def _lock_take(path: Path) -> bool:
    """Atomic create; a lock whose PID is dead is stale and is replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = read_pid(path)
            if pid is not None and pid_alive(pid) and pid != os.getpid():
                return False
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        atexit.register(lambda: path.exists() and read_pid(path) == os.getpid() and path.unlink())
        return True
    return False


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def progress(entry: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    hist.append({**entry, "utc": utc()})
    PROGRESS.write_text(json.dumps(hist, indent=2) + "\n")


def readiness() -> tuple[bool, list[str]]:
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
    import run_gtf as G                              # imported late: the wait loop does not load torch
    return G.check_ready_gtf2()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks while the queue runs")
    args = ap.parse_args()
    if args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # the dry run never touches the GPU
    state = queue_state()
    if args.dry_run:
        ok, problems = readiness()
        print(json.dumps({"queue_state": state, "queue_stop_marker": QUEUE_STOP.exists(),
                          "queue_lock_pid": read_pid(QUEUE_LOCK),
                          "queue_lock_pid_alive": (lambda p: p is not None and pid_alive(p))(read_pid(QUEUE_LOCK)),
                          "readiness_guard_ready": ok, "readiness_problems": problems,
                          "python_exists": PY.exists(), "script_exists": GTF.exists(),
                          "gpu_lock_held_by_live_process": (lambda p: p is not None and pid_alive(p))(read_pid(GPU_LOCK)),
                          "would_start_now": state == "complete" and ok,
                          "plan": [{"step": n, "gpu": g, "cmd": "python " + " ".join(a)} for n, a, g in steps()]}, indent=2))
        return 0
    if not _lock_take(WATCH_LOCK):
        print("another watcher is running", file=sys.stderr); return 1
    log(f"watcher started (pid {os.getpid()}); queue state: {state}")
    progress({"stage": "watcher_started", "pid": os.getpid(), "queue_state": state})
    while True:
        state = queue_state()
        if state == "running":
            time.sleep(args.poll); continue
        break
    if state != "complete":
        log(f"queue state '{state}': G-TF2 is NOT started. The author decides what happens next.")
        progress({"stage": "not_started", "queue_state": state}); return 2
    log("queue_complete seen and the queue process has exited")
    ok, problems = readiness()
    if not ok:
        log(f"readiness guard NOT ready: {problems}. G-TF2 is not started.")
        progress({"stage": "not_started", "reason": "guard_not_ready", "problems": problems}); return 3
    qpid = read_pid(QUEUE_LOCK)
    if qpid is not None and pid_alive(qpid):
        log("the queue lock belongs to a live process: not starting"); return 4
    if not _lock_take(GPU_LOCK):
        log("another GPU job holds the GPU-job lock: not starting")
        progress({"stage": "not_started", "reason": "gpu_lock_held"}); return 5
    env = dict(os.environ)
    for name, argv, gpu in steps():
        if QUEUE_STOP.exists():                      # a queue stop marker appearing now still blocks any new GPU step
            log("queue stop marker appeared: stopping the chain"); progress({"stage": "chain_stopped", "before": name}); return 6
        log(f"step {name}: start")
        progress({"stage": "step_started", "step": name, "gpu": gpu})
        t0 = time.time()
        rc = subprocess.run([str(PY), *argv], cwd=REPO, env=env).returncode
        progress({"stage": "step_done" if rc == 0 else "step_failed", "step": name, "returncode": rc, "seconds": time.time() - t0})
        if rc != 0:
            log(f"step {name} failed (rc {rc}): chain stopped, nothing retried")
            return 7
    log("chain complete")
    progress({"stage": "chain_complete"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
