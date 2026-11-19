#!/usr/bin/env bash
# Waits for results/stage_g/ood/COMPLETED.json, then runs the pre-approved Gate 6 analysis (14, Amendment 9), builds T8 / F-G3,
# and tries the T12 latency step (refuses if the machine is not quiet). Analysis files only; never touches the sealed set or its outputs.
# Start:  setsid nohup bash scripts/post_sealed_watch.sh > results/stage_g/post_sealed.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OOD=results/stage_g/ood
echo "[$(date -Is)] watching for $OOD/COMPLETED.json"
while [ ! -f "$OOD/COMPLETED.json" ]; do
  if [ -f "$OOD/FAILED.json" ]; then echo "[$(date -Is)] FAILED.json present: stop, author decides"; exit 3; fi
  sleep 120
done
echo "[$(date -Is)] COMPLETED.json found"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 19 uv run python scripts/run_gate6_analysis.py --dir "$OOD" || { echo "gate6 analysis failed"; exit 4; }
nice -n 19 uv run python scripts/build_stage_g_t8.py || { echo "T8 failed"; exit 5; }
sleep 180   # let the 1-min load average fall before the latency step
uv run python scripts/build_stage_g_t12.py --latency --device cuda || echo "T12 latency not measured (refused or failed)"
echo "[$(date -Is)] done"
