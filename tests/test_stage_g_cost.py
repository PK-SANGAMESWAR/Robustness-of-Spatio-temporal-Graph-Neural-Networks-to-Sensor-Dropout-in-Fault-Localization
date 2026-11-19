"""T12 cost helpers (``fault_localization/stage_g/cost.py``). No GPU, no sealed data."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fault_localization.stage_g import cost as C

REPO = Path(__file__).resolve().parents[1]


def test_count_params_and_flops_linear():
    m = torch.nn.Linear(4, 3)                      # 15 parameters
    assert C.count_params(m) == 15
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        m(torch.zeros(2, 4))
    assert fc.get_total_flops() == 2 * 2 * 4 * 3   # multiply-add = 2 FLOPs


def test_residual_flops_counts_table_only_when_present():
    class Plain(torch.nn.Module):
        pass

    class WithTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("phys_table", torch.zeros(26, 10, 75))

    assert C.residual_flops(Plain()) == 0
    assert C.residual_flops(WithTable()) == 4 * 26 * 10 * 75


def test_training_hours_and_best_epoch(tmp_path):
    (tmp_path / "metrics_by_epoch.csv").write_text(
        "epoch,train_loss,train_seconds,val_graph_f1\n1,1,3600,0.5\n2,1,3600,0.9\n3,1,3600,0.8\n")
    t = C.training_hours(tmp_path)
    assert t["hours"] == pytest.approx(3.0) and t["best_epoch"] == 2 and t["epochs"] == 3


def test_scoring_rate_ms(tmp_path):
    (tmp_path / "scoring_full_grid.json").write_text(
        '{"cells": {"a": {}, "b": {}}, "n_mask_seeds": 3, "n_windows": 1000, "scoring_wallclock_seconds": 6.0}')
    assert C.scoring_rate_ms(tmp_path) == pytest.approx(1e3 * 6.0 / (1000 * 6))
    assert C.scoring_rate_ms(tmp_path / "none") is None


def test_quiet_reasons_flags_running_sealed_run(tmp_path):
    ood = tmp_path / "results/stage_g/ood"
    ood.mkdir(parents=True)
    (ood / "STARTED.json").write_text("{}")
    assert any("sealed OOD run" in r for r in C.quiet_reasons(tmp_path, max_load=1e9))
    (ood / "COMPLETED.json").write_text("{}")
    assert C.quiet_reasons(tmp_path, max_load=1e9) == []
    assert any("load average" in r for r in C.quiet_reasons(tmp_path, max_load=-1.0))


@pytest.mark.skipif(not (REPO / "results/s4_4/r5/pcstgnn/seed_00042/checkpoint_best.pt").exists(), reason="checkpoint absent")
def test_real_r5_has_one_more_parameter_than_r0n_and_same_backbone_flops():
    r0n = C.load_any("results/s4_4/r0n/pcstgnn/seed_00042", torch.device("cpu"))
    r5 = C.load_any("results/s4_4/r5/pcstgnn/seed_00042", torch.device("cpu"))
    assert C.count_params(r5) - C.count_params(r0n) == 1           # the scalar beta (14, Amendment 5)
    assert C.backbone_flops(r5) == C.backbone_flops(r0n)
    assert C.residual_flops(r5) > 0 and C.residual_flops(r0n) == 0
