"""S7.4: manifest of the assets a reader needs to check the PC-STGNN paper (``results/stage_g/release/MANIFEST.json``).

    nice -n 19 uv run python scripts/build_release_manifest.py

Lists path, size and sha256 of the signature table, the perturbed tables, the OOD generator and its seeds file, the silent-failure
injector, the pre-registration, the frozen model list and the result tables, and records the git commit of the pre-registration and
of the current head. It copies nothing. The sealed OOD data is listed by its recorded directory hash only (never opened here).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/stage_g/release"

GROUPS = {
    "signature_table (operator/)": ["operator/mu.npy", "operator/mu0.npy", "operator/mask.npy", "operator/grid.json", "operator/build_log.csv"],
    "perturbed_tables (Stage G mismatch study)": ["operator_mismatch/tap_p1", "operator_mismatch/tap_m1", "operator_mismatch/imp5",
                                                   "scripts/build_mismatch_tables.py"],
    "ood_generator_and_seeds": ["scripts/generate_ood_sets.py", "data/ood_test.sha256"],
    "silent_failure_injector": ["fault_localization/dropout/silent.py", "scripts/run_silent_detection.py"],
    "preregistration_and_stage_records": ["PC-STGNN/14_PREREGISTRATION.md", "PC-STGNN/08_STAGE_G_eval.md", "PC-STGNN/stage_f_gtf2_preregistration.md",
                                          "PC-STGNN/stage_g_nonsealed_record.md", "config/stage_g/frozen_models.json"],
    "result_tables_and_figures": ["results/stage_g/tables", "results/stage_g/figures", "results/stage_g/analysis", "results/stage_g/release_check/release_check.json"],
    "paper": ["pcstgnn-paper/pcstgnn_paper.tex", "pcstgnn-paper/numbers.tex", "combined-paper/combined_paper.tex", "combined-paper/numbers.tex"],
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def main() -> int:
    out: dict = {"git_head": git("rev-parse", "--short", "HEAD"),
                 "prereg_last_commit": git("log", "-1", "--format=%h %cs", "--", "PC-STGNN/14_PREREGISTRATION.md"),
                 "prereg_uncommitted": bool(git("status", "--porcelain", "--", "PC-STGNN/14_PREREGISTRATION.md")),
                 "sealed_set": {"directory_hash_recorded": (REPO / "data/ood_test.sha256").read_text().split()[0],
                                "note": "data/ood_test is not opened by this script"}, "groups": {}, "missing": []}
    for g, paths in GROUPS.items():
        files = []
        for rel in paths:
            p = REPO / rel
            if not p.exists():
                out["missing"].append(rel)
                continue
            for f in ([p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file() and "raw" not in x.parts and x.suffix != ".npz")):
                files.append({"path": str(f.relative_to(REPO)), "bytes": f.stat().st_size, "sha256": sha(f)})
        out["groups"][g] = files
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print({g: len(f) for g, f in out["groups"].items()}, "missing:", out["missing"], "prereg:", out["prereg_last_commit"],
          "uncommitted" if out["prereg_uncommitted"] else "clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
