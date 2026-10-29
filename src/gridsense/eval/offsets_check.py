"""O17 (author, 2026-10-29): k = 10 on `test`, seed 42, batch 32, with 5 mask offsets (3..7, none of the paper's 0..2).

Method: macro-F1 per offset (same call as identity.py); mean and sample SD (ddof = 1) over the 5 offsets; the stored value
(results/dropout_study_aware/degradation_curves.csv, seed 42, k = 10, a mean of the paper's 3 offsets) passes if
|stored - mean| <= 2 SD. The SD is the SD of ONE offset; the stored value is a mean of 3, so SD/sqrt(3) is reported as a
stricter, informational comparison. The tolerance of 1e-3 is not changed.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.metrics import f1_score

from gridsense.config import ROOT, TRACK

OFFSETS = [3, 4, 5, 6, 7]
CURVES = ROOT / "results" / "dropout_study_aware" / "degradation_curves.csv"


def run(seed: int = 42, k: int = 10) -> dict:
    d = TRACK / "eval" / "latents" / f"seed{seed}" / "test"
    y = np.load(d / "labels.npy").astype(int)
    f1 = [float(f1_score(y, np.load(d / f"votes_k{k}_o{o}.npy").argmax(1), labels=list(range(26)), average="macro", zero_division=0))
          for o in OFFSETS]
    row = next(r for r in csv.DictReader(CURVES.open()) if r["model"] == "rgsage_max" and int(r["k"]) == k)
    stored = json.loads(row["f1_per_train_seed"])[[42, 43, 44].index(seed)]
    mean, sd = float(np.mean(f1)), float(np.std(f1, ddof=1))
    z = (stored - mean) / sd
    return {"seed": seed, "k": k, "offsets": OFFSETS, "f1_per_offset": f1, "mean": mean, "sd_ddof1": sd,
            "stored_value": stored, "stored_minus_mean": stored - mean, "z": z, "pass_within_2sd": abs(z) <= 2.0,
            "info_sd_of_3_offset_mean": sd / np.sqrt(3), "info_z_vs_sd_over_sqrt3": (stored - mean) / (sd / np.sqrt(3)),
            "info_earlier_batch32_offsets_0_1_2": json.loads((TRACK / "reports" / "s2" / "identity_curve_seed42.json").read_text())["per_k"][str(k)]["f1_per_offset"],
            "method": "5 new mask offsets, batch 32, k = 10, test; pass if |stored - mean| <= 2 SD (SD of one offset, ddof = 1); tolerance 1e-3 unchanged"}


if __name__ == "__main__":
    res = run()
    out = TRACK / "reports" / "layer0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "o17_k10_offsets_seed42.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["pass_within_2sd"] else 1)
