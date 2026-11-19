"""S7.1 error attribution on the test split (CPU). Writes results/stage_g/analysis/error_attribution.json.

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/run_error_attribution.py

Uses the saved per-window predictions of ``results/stage_g/test`` and the identifiability matrices of
``results/pcstgnn/audit/s2_1_identifiability``. Logic and definitions: ``fault_localization/stage_g/attribution.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fault_localization.stage_g import attribution as A  # noqa: E402


def main() -> int:
    models = {"R0ⁿ": "r0n", "R5": "r5", "R2": "r2"}
    res = A.run(REPO / "results/stage_g/test", models)
    summary = {}
    for label, by_k in res.items():
        for k, per_seed in by_k.items():
            keys = ["error_rate", "frac_errors_in_indist_pairs", "chance_frac", "frac_indist_class_windows_wrong",
                    "frac_other_class_windows_wrong", "n_wrong", "n_errors_in_indist_pairs"]
            summary[f"{label}|{k}"] = {key: [float(np.mean([o[key] for o in s["offsets"]])) for s in per_seed] for key in keys}
    out = {"threshold_D": A.D_THRESHOLD, "split": "test", "sigma": 0, "seeds": [42, 43, 44], "n_offsets": 3,
           "source": "results/stage_g/test/*/seed_*_predictions.npz; results/pcstgnn/audit/s2_1_identifiability/D_global_k{k}.npy",
           "per_seed_means_over_offsets": summary, "detail": res}
    p = REPO / "results/stage_g/analysis/error_attribution.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    for key, v in summary.items():
        print(key, {m: [round(x, 4) for x in v[m]] for m in ("error_rate", "frac_errors_in_indist_pairs", "chance_frac")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
