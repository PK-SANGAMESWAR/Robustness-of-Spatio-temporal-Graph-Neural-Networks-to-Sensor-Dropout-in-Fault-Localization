"""Sample the S4.4 queue's CPU-time rate (by PID) and epoch times, so S2 jobs can be backed off (author rule 2026-10-28).

Rule: if the S4.4 epoch time rises by more than 10 % while S2/S3 jobs run, reduce torch threads from 4 to 2.
Epoch time is too slow a signal (about 43 min), so this also logs the queue process's CPU rate every INTERVAL s.
Usage: .venv/bin/python -m gridsense.eval.s44_monitor <queue_pid> <epoch_csv> [seconds]
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone

from gridsense.config import TRACK

TICKS = os.sysconf("SC_CLK_TCK")
INTERVAL = 30
LOG = TRACK / "eval" / "logs" / "s44_monitor.csv"


def cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / TICKS  # utime + stime


def read_bytes(pid: int) -> int:
    """rchar of the queue process: bytes it read. During S4.4 scoring there is no epoch time, so its read rate is the throughput proxy."""
    with open(f"/proc/{pid}/io") as f:
        return int(f.readline().split()[1])


def epoch_times(path: str) -> list[float]:
    return [float(r["train_seconds"]) for r in csv.DictReader(open(path))]


def main() -> None:
    pid, epoch_csv = int(sys.argv[1]), sys.argv[2]
    total = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "queue_cpu_pct", "n_epochs_done", "last_epoch_seconds", "gridsense_jobs", "queue_read_MBps"])
        last, t_last, t_end = cpu_seconds(pid), time.time(), time.time() + total
        last_rd = read_bytes(pid)
        while time.time() < t_end:
            time.sleep(INTERVAL)
            now, t, rd = cpu_seconds(pid), time.time(), read_bytes(pid)
            eps = epoch_times(epoch_csv)
            jobs = sum(1 for p in os.listdir("/proc") if p.isdigit() and _is_gs_job(p))
            w.writerow([datetime.now(timezone.utc).strftime("%H:%M:%S"), round(100 * (now - last) / (t - t_last), 1),
                        len(eps), round(eps[-1], 1) if eps else "", jobs, round((rd - last_rd) / 1e6 / (t - t_last), 2)])
            f.flush()
            last, t_last, last_rd = now, t, rd


def _is_gs_job(pid: str) -> bool:
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError:
        return False
    return len(cmd) > 2 and cmd[1:3] == [b"-m", b"gridsense.model.extract"] or (len(cmd) > 2 and cmd[1:3] == [b"-m", b"gridsense.eval.s2_k0"])


if __name__ == "__main__":
    main()
