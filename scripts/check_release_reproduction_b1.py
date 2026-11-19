"""S7.4 release check, rungs B1 and B1ᵃ: recompute both training-free baselines on CPU into a new folder and compare with the stored audit values.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 19 uv run python scripts/check_release_reproduction_b1.py

B1 (delta-V nearest signature): ``run_b1_deltav_baseline.py`` with its OUT_DIR redirected to ``results/stage_g/release_check/b1`` (no cached v_nominal, so it is
recomputed from the training split's pre-fault steps). B1ᵃ (matched filter): ``run_physics_matched_filter.py --out results/stage_g/release_check/b1a``.
Nothing under ``results/pcstgnn/audit`` or ``results/physics_baseline`` is written. Writes ``release_check/b1_b1a_check.json``.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/stage_g/release_check"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("b1mod", REPO / "scripts/run_b1_deltav_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT_DIR = OUT / "b1"
    sys.argv = ["run_b1_deltav_baseline.py", "--device", "cpu"]
    assert mod.main() == 0
    new = json.loads((OUT / "b1/b1_full_eval.json").read_text())["b1_delta_v"]
    old = json.loads((REPO / "results/pcstgnn/audit/s2_3_b1/b1_full_eval.json").read_text())["b1_delta_v"]
    vn_new = np.load(OUT / "b1/v_nominal.npy")
    vn_old = np.load(REPO / "results/pcstgnn/audit/s2_3_b1/v_nominal.npy")
    b1 = {"f1_by_k": {k: {"new": new["f1_by_k"][k], "stored": old["f1_by_k"][k], "abs_diff": abs(new["f1_by_k"][k] - old["f1_by_k"][k])} for k in old["f1_by_k"]},
          "v_nominal_max_abs_diff": float(np.abs(vn_new - vn_old).max())}
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_physics_matched_filter.py"), "--out", str(OUT / "b1a"), "--device", "cpu"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:                           # the script may not take --device; retry without it
        r = subprocess.run([sys.executable, str(REPO / "scripts/run_physics_matched_filter.py"), "--out", str(OUT / "b1a")],
                           cwd=REPO, capture_output=True, text=True)
    r.check_returncode()
    mf_new = json.loads((OUT / "b1a/matched_filter.json").read_text())
    mf_old = json.loads((REPO / "results/physics_baseline/matched_filter.json").read_text())
    b1a = {"tau": {"new": mf_new["tau"], "stored": mf_old["tau"]}, "val_f1_k0": {"new": mf_new["val_f1_k0"], "stored": mf_old["val_f1_k0"]},
           "curve_max_abs_diff": max(abs(mf_new["curve"][c]["f1_mean"] - mf_old["curve"][c]["f1_mean"]) for c in mf_old["curve"] if c in mf_new["curve"]),
           "n_cells_compared": sum(1 for c in mf_old["curve"] if c in mf_new["curve"])}
    res = {"B1": b1, "B1a": b1a}
    (OUT / "b1_b1a_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
