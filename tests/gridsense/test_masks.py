"""k > 0 masks equal the paper's seeded generator (S2, author 2026-10-28). Needs the test split and checkpoint 42."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytestmark = [pytest.mark.slow]


@pytest.mark.skipif(not (ROOT / "dataset_fullscale_50" / "splits" / "test.txt").exists(), reason="pending: dataset")
def test_masks_identical_to_paper_generator_small() -> None:
    from gridsense.eval.mask_check import run

    res = run(n_batches=6)
    assert res["mask_calls_identical"], "mask calls differ from the paper's evaluate_dropout_curve"
    assert res["n_mask_calls_paper"] == 6 * 4 * 3
    assert res["max_f1_abs_diff"] <= 1e-3
