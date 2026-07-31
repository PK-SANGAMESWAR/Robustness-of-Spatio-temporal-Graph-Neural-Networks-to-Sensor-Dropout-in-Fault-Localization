"""Build a scaled-up dataset for the MEDIUM_DATASET trend test (docs/MEDIUM-DATASET.md).

Reuses the existing simulation + windowing pipeline unchanged. Raw fault-event
simulations are generated into one shared, growing pool
(``dataset/raw_sim/``) keyed by (fault_bus, fault_type, resistance, run_id) —
the same deterministic seeding the tiny run already used — so runs already
simulated for a smaller scale are never re-simulated for a larger one.

For each requested scale, a self-contained data root (``dataset_<tag>/``) is
created with ``raw_sim/`` populated as symlinks into the shared pool, filtered
to ``run_id < runs_per_scenario``. ``fault_localization.processing.build_dataset``
is then run unmodified against that root to produce ``processed/`` and
``splits/``.

Usage:
    uv run python scripts/generate_trend_dataset.py --tag medium_A --runs 20
    uv run python scripts/generate_trend_dataset.py --tag medium_B --runs 40
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Allow running as a script from the project root
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# NOTE: OpenDSS's "Compile" command chdir's into the directory containing the
# compiled .dss file as a side effect (a pre-existing quirk of this codebase,
# not introduced here). Every path below is therefore resolved to an absolute
# path against _PROJECT_ROOT *before* any compile_feeder() call happens,
# so a stray chdir can never silently redirect output into the wrong place.
_SHARED_RAW_POOL = _PROJECT_ROOT / "dataset" / "raw_sim"
_FILENAME_RE = re.compile(r"^(?P<bus>\d+)_(?P<type>[A-Z]+)_(?P<res>[0-9p]+)_(?P<run>\d{4})\.npz$")


def _run_id_of(path: Path) -> int:
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized raw event filename: {path.name}")
    return int(m.group("run"))


def grow_shared_pool(runs_per_scenario: int, n_workers: int | None = None) -> None:
    """Ensure the shared raw pool has at least `runs_per_scenario` runs/scenario.

    Deliberately does NOT call compile_feeder() here: each per-event worker
    (simulate_fault_event) compiles its own feeder instance already, and
    calling it in this (parent) process first would chdir the parent before
    _SHARED_RAW_POOL / data_root paths get used — see the module-level note.
    """
    from fault_localization.simulation.feeder import ensure_feeder_files
    from fault_localization.simulation.runner import stage_full_generation

    ensure_feeder_files()

    logger.info("Growing shared raw pool %s to %d runs/scenario …", _SHARED_RAW_POOL, runs_per_scenario)
    stage_full_generation(
        runs_per_scenario=runs_per_scenario,
        output_dir=_SHARED_RAW_POOL,
        n_workers=n_workers,
    )


def materialize_scale_dataset(tag: str, runs_per_scenario: int) -> Path:
    """Symlink the filtered subset of the shared pool into dataset_<tag>/raw_sim/."""
    data_root = _PROJECT_ROOT / f"dataset_{tag}"
    raw_dir = data_root / "raw_sim"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pool_files = sorted(_SHARED_RAW_POOL.glob("*.npz"))
    if not pool_files:
        raise FileNotFoundError(f"Shared raw pool {_SHARED_RAW_POOL} is empty")

    kept = 0
    for src in pool_files:
        if _run_id_of(src) >= runs_per_scenario:
            continue
        dst = raw_dir / src.name
        if dst.is_symlink() or dst.exists():
            kept += 1
            continue
        dst.symlink_to(src.resolve())
        kept += 1

    expected = None
    from fault_localization.config.constants import NUM_SCENARIOS
    expected = NUM_SCENARIOS * runs_per_scenario
    if kept != expected:
        raise RuntimeError(
            f"dataset_{tag}: expected {expected} raw events (runs_per_scenario={runs_per_scenario}), "
            f"found {kept} — shared pool may not have been grown far enough."
        )
    logger.info("dataset_%s/raw_sim: %d symlinked events (runs_per_scenario=%d).", tag, kept, runs_per_scenario)
    return data_root


def build_scale_dataset(tag: str, runs_per_scenario: int, feeder_config: str = "default") -> Path:
    from fault_localization.processing.build_dataset import build
    from fault_localization.validation.dataset_checks import strict_benchmark_validation

    data_root = materialize_scale_dataset(tag, runs_per_scenario)
    build(data_root, feeder_config=feeder_config)

    # Pre-flight manifest gate: abort loudly here rather than discover a
    # miscount after hours of training. Asserts exact event/window counts,
    # 26-class coverage, 20/20 per-event balance, and clean splits.
    summary = strict_benchmark_validation(data_root, runs_per_scenario=runs_per_scenario)
    logger.info("✓ Pre-flight manifest check passed for '%s': %s", tag, summary)
    return data_root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="Scale tag, e.g. medium_A, medium_B")
    p.add_argument("--runs", type=int, required=True, help="runs_per_scenario for this scale")
    p.add_argument("--workers", type=int, default=None, help="Multiprocessing workers for raw generation")
    p.add_argument("--skip-generation", action="store_true", help="Assume shared pool already covers --runs; only (re)build processed/splits")
    args = p.parse_args()

    if not args.skip_generation:
        grow_shared_pool(args.runs, n_workers=args.workers)
    data_root = build_scale_dataset(args.tag, args.runs)
    logger.info("Scale '%s' ready at %s (runs_per_scenario=%d).", args.tag, data_root, args.runs)


if __name__ == "__main__":
    main()
