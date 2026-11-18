"""Ablation ladder (rag-eval.md § 10): assembled from existing Layer 0 JSON; no model run, no test split use."""
from __future__ import annotations

import json

import pytest

from gridsense.eval import ablation_ladder as AL


def test_ladder_numbers_are_copied_from_the_layer0_files_and_a0_is_by_hand():
    d = AL.build()
    val = json.loads((AL.L0 / "layer0_val_seed42.json").read_text())
    test = json.loads((AL.L0 / "test_run_seed42.json").read_text())
    for k in AL.KS:
        acc = val["per_k"][k]["stgnn"]["acc_cal_b_post"]
        a0 = d["val"][k]["A0"]
        assert a0["coverage"] == 1.0 and a0["error_recall"] == 0.0 and a0["selective_risk"] == pytest.approx(1 - acc) and a0["false_acceptance"] == pytest.approx(1 - acc)
        c = val["gates"]["p+agreement_heldout"]["0.95"]["per_k"][k]["cal_b"]
        assert d["val"][k]["A2"]["coverage"] == c["coverage"] and d["val"][k]["A2"]["false_acceptance"] == pytest.approx(c["errors_accepted"] / c["n_windows"])
        g = test["per_k"][k]["gate_acc95"]
        assert d["test_frozen_gate"][k]["gate_v1 coverage"] == g["coverage"] and d["test_frozen_gate"][k]["gate_v1 selective_risk"] == pytest.approx(1 - g["accuracy_confident"])


def test_ladder_says_a3_is_not_run_and_a1_a2_were_never_run_on_test():
    md = AL.markdown(AL.build())
    assert "A3 (+ silent-failure check): **NOT RUN**" in md and "A1 and A2 were never run on `test`" in md and "NOT a like-for-like" in md
    assert "A4 (+ retrieval and LLM report)" in md
