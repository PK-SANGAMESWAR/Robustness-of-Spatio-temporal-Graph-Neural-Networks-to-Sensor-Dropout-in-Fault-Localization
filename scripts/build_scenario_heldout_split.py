"""Build a scenario-held-out split as a sibling of the frozen corpus.

WHY
---
The frozen split (``processing/splits.py``) is taken at the event level and
stratified by fault bus, but the events of one scenario are its 50 independent
runs, which differ only in the load multiplier drawn per run.  Every one of the
825 (fault bus, fault type, resistance) configurations is therefore present in
all three splits, and test accuracy measures generalization to unseen
*operating conditions* of a configuration the model has already seen, not to an
unseen fault configuration.

This script builds the complementary partition: whole scenarios are assigned to
one split, so the test set contains (fault type, resistance) configurations that
never appear in training.  The assignment is made *within each fault bus*, so
all 25 fault classes remain represented in every split and the label space is
unchanged; what is withheld is the combination of fault type and resistance at
that bus.

The frozen corpus is never modified.  A sibling directory is created whose
``processed/`` is a symlink to the original and whose ``splits/`` is new, so
both partitions can be trained against by pointing ``--data-root`` at one or the
other, with byte-identical windows underneath.

Normalization statistics are deliberately NOT recomputed: ``norm_stats.json`` is
a global per-channel mean and standard deviation over hundreds of millions of
samples, and two 70 % subsets of the same corpus agree on it to several decimal
places.  Reusing the frozen statistics keeps the split the only variable that
changes (``.rules/10-integrity.md``), at the cost of a negligible dependence on
values computed over a superset of the new training split.

Usage:
    uv run python scripts/build_scenario_heldout_split.py \
        --data-root dataset_fullscale_50 \
        --out-root dataset_fullscale_50_scenario
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset_fullscale_50")
    p.add_argument("--out-root", default="dataset_fullscale_50_scenario")
    p.add_argument("--seed", type=int, default=20260817)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    src = (repo / args.data_root).resolve()
    dst = (repo / args.out_root).resolve()

    meta_path = src / "processed" / "metadata.csv"
    labels_path = src / "processed" / "labels.csv"
    if not meta_path.exists():
        print(f"missing {meta_path}")
        return 1

    # window_id -> scenario key, grouped so whole scenarios move together.
    by_scenario: dict[tuple, list[str]] = defaultdict(list)
    by_bus: dict[str, set[tuple]] = defaultdict(set)
    with meta_path.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["fault_bus"], row["fault_type"], row["resistance"])
            by_scenario[key].append(row["window_id"])
            by_bus[row["fault_bus"]].add(key)

    rng = random.Random(args.seed)
    assign: dict[tuple, str] = {}
    for bus in sorted(by_bus):
        scen = sorted(by_bus[bus])
        rng.shuffle(scen)
        n = len(scen)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)
        for s in scen[:n_train]:
            assign[s] = "train"
        for s in scen[n_train:n_train + n_val]:
            assign[s] = "val"
        for s in scen[n_train + n_val:]:
            assign[s] = "test"

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key, wids in by_scenario.items():
        splits[assign[key]].extend(wids)

    # Every class must survive in every split, or the comparison is not like for
    # like: a missing class silently changes what a 26-class macro-F1 averages.
    labels = {r["window_id"]: int(r["label"]) for r in
              csv.DictReader(labels_path.open())}
    for name, wids in splits.items():
        present = {labels[w] for w in wids}
        missing = set(range(26)) - present
        if missing:
            print(f"ERROR: split {name} is missing classes {sorted(missing)}")
            return 1

    # No scenario may straddle two splits — that is the whole point.  ``assign``
    # is keyed by scenario so this holds by construction; the check guards
    # against a future edit that groups by something coarser.
    if len(assign) != len(by_scenario):
        print("ERROR: scenario assignment is not one-to-one with scenarios")
        return 1

    (dst / "splits").mkdir(parents=True, exist_ok=True)
    link = dst / "processed"
    if not link.exists():
        link.symlink_to(src / "processed", target_is_directory=True)

    for name, wids in splits.items():
        wids_sorted = sorted(wids)
        (dst / "splits" / f"{name}.txt").write_text("\n".join(wids_sorted) + "\n")
        n_scen = sum(1 for k, v in assign.items() if v == name)
        print(f"{name:<6}: {len(wids_sorted):>9,} windows  "
              f"{len(wids_sorted)//40:>7,} events  {n_scen:>4} scenarios")

    counts = defaultdict(int)
    for key, split in assign.items():
        counts[split] += 1
    print(f"\nScenarios: {sum(counts.values())} total, "
          f"partitioned {counts['train']}/{counts['val']}/{counts['test']}")
    print(f"Wrote {dst}/splits/  (processed/ symlinked to the frozen corpus)")
    print("\nTrain with:  --data-root", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
