"""Check that the sealed-set windowing code reproduces the processed pilot set exactly (CPU, pilot only).

Re-makes windows, labels and metadata from ``data/ood_pilot/raw_sim`` with
``fault_localization.stage_g.ood_pipeline.build_eval_root`` into a temporary folder and compares every
array and every CSV row with ``data/ood_pilot/processed``.  Never reads ``data/ood_test/``.

    nice -n 19 uv run python scripts/check_pilot_windows_equal.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.stage_g import ood_pipeline as P  # noqa: E402

PILOT = REPO / "data/ood_pilot"
OUT = REPO / "results/stage_g/pilot_windows_check.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        stats = P.build_eval_root(PILOT / "raw_sim", Path(tmp) / "eval", REPO / "dataset_fullscale_50", 1100)
        new, ref = Path(tmp) / "eval/processed", PILOT / "processed"
        n_files = n_bad = 0
        for f in sorted((ref / "windows").glob("*.npz")):
            n_files += 1
            with np.load(new / "windows" / f.name) as a, np.load(f) as b:
                if not (np.array_equal(a["windows"], b["windows"]) and np.array_equal(a["window_ids"], b["window_ids"])):
                    n_bad += 1
        csv_equal = {}
        for name in ("labels.csv", "metadata.csv"):
            csv_equal[name] = (new / name).read_bytes() == (ref / name).read_bytes()
        all_equal = (Path(tmp) / "eval/splits/all.txt").read_bytes() == (PILOT / "splits/all.txt").read_bytes()
    report = {"n_event_files_compared": n_files, "n_event_files_different": n_bad, "csv_byte_identical": csv_equal,
              "all_txt_byte_identical": all_equal, "stats": stats}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if (n_bad == 0 and all(csv_equal.values()) and all_equal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
