"""S7.4 release check: reproduce a training-free T6 row from the released operator table alone.

    OMP_NUM_THREADS=2 nice -n 19 uv run python scripts/check_release_reproduction.py [--out results/stage_g/release_check]

Recomputes PSM (argmin of rho, zero learned parameters) at k = 0 on the validation split with ``operator/{mu,mu0}.npy`` and the
canonical sigma_eps, and compares the macro-F1 with the value stored by the audit run (``results/pcstgnn/audit/s2_4_psm/psm_full_eval.json``).
Writes only under ``--out`` (never into the audit directory). Also records the sha256 of the operator files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results/stage_g/release_check"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import run_psm_full_eval as P
    from fault_localization.physics import table as T
    from fault_localization.physics.residual import flatten_table

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = json.loads((REPO / "results/pcstgnn/audit/s2_4_psm/psm_full_eval.json").read_text())["validation"]["f1_by_k"]["0"]
    P.K_LEVELS = [0]
    t0 = time.time()
    tbl = T.load_table(REPO / "operator")
    table = flatten_table(tbl["mu"], tbl["mu0"])
    res = P.eval_val_split(REPO / "dataset_fullscale_50", table, torch.device("cpu"), checkpoint_path=out / "partial_k0.json")
    f1 = float(res["f1_by_k"][0])
    rec = {"check": "PSM macro-F1, validation, k=0, from operator/ alone", "recomputed": f1, "stored": stored, "abs_diff": abs(f1 - stored),
           "match_1e-9": bool(abs(f1 - stored) < 1e-9), "sigma_eps": P.CANONICAL_SIGMA_EPS, "device": "cpu",
           "operator_sha256": {n: sha(REPO / "operator" / n) for n in ("mu.npy", "mu0.npy", "grid.json", "mask.npy")},
           "seconds": time.time() - t0}
    (out / "release_check.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("recomputed", "stored", "abs_diff", "match_1e-9", "seconds")}, indent=1))
    return 0 if rec["match_1e-9"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
