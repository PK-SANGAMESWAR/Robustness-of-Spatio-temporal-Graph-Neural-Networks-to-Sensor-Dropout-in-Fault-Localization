#!/usr/bin/env bash
# Sequential CPU chain AFTER the gate is frozen and committed (O24 step 4, O18; author 2026-10-30).
# One job at a time (RAM: an out-of-memory kill could hit the S4.4 queue). nice 19; torch threads per python process: eval/threads.txt, default 4.
# Step 1 computes the seed-42 test neighbours; step 2 is the ONE run of the frozen gate on test (the script refuses a second run);
# the loop prepares and evaluates seeds 43 and 44 with the frozen thresholds (Layer 0 only).
set -u
cd /home/ubuntu/Fault-Localization
export PYTHONPATH=src:.
PY=".venv/bin/python"
run() { echo "=== $(date -u +%H:%M:%S) $*"; nice -n 19 "$@" || { echo "FAILED: $*"; exit 1; }; }
for k in 0 1 2 5 10; do run $PY -m gridsense.eval.knn_features --seed 42 --split test --k $k; done   # one job per k: a thread change (eval/threads.txt) applies to the next job
run $PY -m gridsense.eval.gate_test
for s in 43 44; do
  run $PY -m gridsense.model.extract --seed $s --split train --k 0 --latents --latent-repr b_mean_max
  run $PY -m gridsense.cases.build --seed $s --tag train
  run $PY -m gridsense.model.extract --seed $s --split val --k 0 1 2 5 10 --offsets 1
  run $PY -m gridsense.model.extract --seed $s --split test --k 0 1 2 5 10 --offsets 1 --latents --latent-repr b_mean_max
  for k in 0 1 2 5 10; do run $PY -m gridsense.eval.knn_features --seed $s --split test --k $k; done
  run $PY -m gridsense.eval.seed_runs $s
done
echo "=== chain done $(date -u +%H:%M:%S)"
