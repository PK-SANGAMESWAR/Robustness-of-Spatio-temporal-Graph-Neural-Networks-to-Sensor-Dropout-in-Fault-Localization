"""ONE run of the frozen gate on `test` (O24 step 4; author 2026-10-30). Seed 42, full train store, held-out neighbours (O20).

Refuses to run (a) if the config is not identical to the version committed on gridsense-eval, (b) if the store fingerprint differs from the
one frozen in the config, (c) if this run already exists. The gate must not be changed after the run; the result file doubles as the lock.
Run: nice -n 19 env PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.gate_test
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys

from gridsense.config import ROOT, TRACK
from gridsense.eval.gate_fit import CONFIG
from gridsense.eval.layer0_final import evaluate, load_split
from gridsense.verification.gate import Gate

RESULT = TRACK / "reports" / "layer0" / "test_run_seed42.json"


def main() -> None:
    if RESULT.exists():
        sys.exit(f"{RESULT.name} exists: the frozen gate was already run once on test and must not be run again")
    committed = subprocess.run(["git", "show", "gridsense-eval:rag-for-dropout/configs/gate_v1.json"], cwd=ROOT, capture_output=True)
    if committed.returncode != 0 or committed.stdout != CONFIG.read_bytes():
        sys.exit("configs/gate_v1.json is not committed on gridsense-eval (or differs from the committed file): freeze first")
    gate = Gate.load(CONFIG)
    manifest = json.loads((TRACK / "eval/case_store/seed42_full/store_manifest.json").read_text())
    if manifest["fingerprint"] != gate.cfg["store"]["fingerprint"]:
        sys.exit("case store fingerprint differs from the frozen one")
    data = load_split(42, "test")
    res = evaluate(gate, data, None, boot=True)
    cov10 = res["per_k"]["10"]["gate_acc95"]["coverage"]
    res["frozen_gate"] = {"config_sha256": gate.cfg["config_sha256"], "base_commit": gate.cfg["base_commit"], "operating_point": "acc95 (accuracy >= 95 % on CONFIDENT, thresholds from cal-B)",
                          "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run_once": True,
                          "coverage_at_k10_acc95": cov10, "coverage_at_k10_below_50_percent": bool(cov10 < 0.5)}
    RESULT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"test run written: coverage at k = 10 (95 % point) = {cov10:.4f}", flush=True)


if __name__ == "__main__":
    main()
