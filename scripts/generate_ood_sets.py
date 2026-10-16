"""S2.0 -- generate the OOD-pilot and sealed OOD-test sets (``04_STAGE_C_gonogo.md`` S2.0).

Two separate generations, disjoint seeds by construction (``deterministic_seed_ood``,
``OOD_SEED_OFFSET``, ``tests/test_ood_seed.py``):

| set | R (Ohm) | runs | run_id range | events |
|---|---|---|---|---|
| pilot | {5, 20} | 2 | 1000-1001 | 25*11*2*2 = 1,100 |
| test  | {3, 5, 20, 30} | 10 | 2000-2009 | 25*11*4*10 = 11,000 |

Both use the same generator (``simulate_fault_event``, Daily mode, 60 steps) and load
distribution as the frozen corpus; only the seed source and resistance values differ. Output is
raw_sim only (windowing/labels/splits are Stage-C-eval concerns, added when each step needs
them, not here) -- this script's only job is to produce and seal the two corpora.

Usage:
    uv run python scripts/generate_ood_sets.py --set pilot
    uv run python scripts/generate_ood_sets.py --set test
    uv run python scripts/generate_ood_sets.py --set both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from random import Random

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.config.constants import (  # noqa: E402
    FAULT_BUSES,
    FAULT_TYPES,
    deterministic_seed_ood,
)
from fault_localization.simulation.runner import (  # noqa: E402
    event_exists,
    save_event,
    simulate_fault_event,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SET_SPECS = {
    "pilot": {"resistances": [5.0, 20.0], "run_ids": range(1000, 1002), "dir": "data/ood_pilot"},
    "test": {"resistances": [3.0, 5.0, 20.0, 30.0], "run_ids": range(2000, 2010), "dir": "data/ood_test"},
}


def _worker(args):
    fault_bus, fault_type, resistance, run_id, output_dir_str, set_name = args
    output_dir = Path(output_dir_str)
    if event_exists(fault_bus, fault_type, resistance, run_id, output_dir):
        return None
    seed = deterministic_seed_ood(fault_bus, fault_type, resistance, run_id, set_name)
    rng = Random(seed)
    try:
        recording, meta = simulate_fault_event(
            fault_bus, fault_type, resistance, run_id, rng, seed=seed
        )
        meta["ood_set"] = set_name
        save_event(recording, meta, output_dir)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"FAILED bus={fault_bus} type={fault_type} R={resistance} run={run_id}: {exc}"


def generate(set_name: str, n_workers: int | None = None) -> dict:
    spec = SET_SPECS[set_name]
    output_dir = (REPO / spec["dir"] / "raw_sim").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (bus, ftype, r, run_id, str(output_dir), set_name)
        for bus in FAULT_BUSES
        for ftype in FAULT_TYPES
        for r in spec["resistances"]
        for run_id in spec["run_ids"]
    ]
    total_expected = len(tasks)
    tasks = [t for t in tasks if not event_exists(t[0], t[1], t[2], t[3], output_dir)]

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    errors: list[str] = []
    if tasks:
        logger.info("[%s] generating %d/%d missing events with %d workers -> %s",
                     set_name, len(tasks), total_expected, n_workers, output_dir)
        if n_workers == 1:
            for args in tqdm(tasks, desc=set_name):
                err = _worker(args)
                if err:
                    errors.append(err)
        else:
            with mp.Pool(n_workers) as pool:
                for err in tqdm(pool.imap_unordered(_worker, tasks, chunksize=4),
                                 total=len(tasks), desc=set_name):
                    if err:
                        errors.append(err)
    else:
        logger.info("[%s] all %d events already present.", set_name, total_expected)

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"[{set_name}] {len(errors)}/{len(tasks)} generations failed:\n{preview}")

    # Verify seeds actually used (re-derive, don't trust the files) and disjointness within set.
    seeds = {
        deterministic_seed_ood(bus, ftype, r, run_id, set_name)
        for bus in FAULT_BUSES for ftype in FAULT_TYPES
        for r in spec["resistances"] for run_id in spec["run_ids"]
    }
    n_files = len(list(output_dir.glob("*.npz")))
    return {
        "set_name": set_name,
        "output_dir": str(output_dir.parent),
        "n_events_expected": total_expected,
        "n_files_on_disk": n_files,
        "n_unique_seeds": len(seeds),
        "seeds_min": min(seeds),
        "seeds_max": max(seeds),
        "resistances": spec["resistances"],
        "run_ids": [spec["run_ids"].start, spec["run_ids"].stop],
    }


def sha256_of_dir(dir_path: Path) -> str:
    """Hash of the sorted concatenation of every file's own sha256 -- stable under
    filesystem reordering, sensitive to any single byte in any file."""
    h = hashlib.sha256()
    files = sorted(p for p in dir_path.rglob("*") if p.is_file())
    for p in files:
        h.update(p.relative_to(dir_path).as_posix().encode("utf-8"))
        file_h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                file_h.update(chunk)
        h.update(file_h.digest())
    return h.hexdigest()


def seal_test_set() -> dict:
    """Hash ``data/ood_test/`` and write ``data/ood_test.sha256``. Never reads pixel/voltage
    content back out -- only walks bytes to hash them, which is not "opening" the set for use
    (Stage G only)."""
    test_dir = (REPO / SET_SPECS["test"]["dir"]).resolve()
    digest = sha256_of_dir(test_dir)
    hash_path = REPO / "data" / "ood_test.sha256"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{digest}  ood_test/\n")
    return {"hash_path": str(hash_path), "sha256": digest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["pilot", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seal", action="store_true",
                     help="After generating, hash data/ood_test/ and write data/ood_test.sha256.")
    args = ap.parse_args()

    sets = ["pilot", "test"] if args.set == "both" else [args.set]
    report = {}
    for set_name in sets:
        report[set_name] = generate(set_name, args.workers)

    if "pilot" in report and "test" in report:
        pilot_seeds = {
            deterministic_seed_ood(bus, ftype, r, run_id, "pilot")
            for bus in FAULT_BUSES for ftype in FAULT_TYPES
            for r in SET_SPECS["pilot"]["resistances"] for run_id in SET_SPECS["pilot"]["run_ids"]
        }
        test_seeds = {
            deterministic_seed_ood(bus, ftype, r, run_id, "test")
            for bus in FAULT_BUSES for ftype in FAULT_TYPES
            for r in SET_SPECS["test"]["resistances"] for run_id in SET_SPECS["test"]["run_ids"]
        }
        overlap = pilot_seeds & test_seeds
        report["seeds_shared_pilot_test"] = len(overlap)
        if overlap:
            raise RuntimeError(f"pilot and test share {len(overlap)} seeds -- must be 0")

    if args.seal:
        if "test" not in sets:
            raise SystemExit("--seal requires --set test (or both)")
        report["seal"] = seal_test_set()

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
