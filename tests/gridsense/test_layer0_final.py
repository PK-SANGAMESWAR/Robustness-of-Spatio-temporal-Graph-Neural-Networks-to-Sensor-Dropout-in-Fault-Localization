"""layer0_final.evaluate on synthetic data: bookkeeping, determinism, CI sanity, unsupported k (O24). No real data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gridsense.eval.gate_fit import fit_k  # noqa: E402
from gridsense.eval.layer0_final import KData, SplitData, evaluate  # noqa: E402
from gridsense.verification.gate import Gate  # noqa: E402

R, W = 80, 19


def _kdata(rng, scale: float) -> KData:
    n = R * W
    logits = rng.normal(size=(n, 26)) * scale
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(26, p=p) for p in probs])
    agree = np.clip(np.round(probs.max(1) + rng.normal(0, 0.15, n), 1), 0, 1)
    knn = {"heldout_agree_10": agree, "standard_agree_10": np.clip(agree + 0.1, 0, 1), "heldout_nbrerr_10": rng.random(n), "heldout_nnsim": rng.random(n),
           "standard_nnsim": rng.random(n), "standard_nbrerr_10": rng.random(n)}
    for s in ("heldout", "standard"):
        for kk in (5, 10, 20):
            knn[f"{s}_vote_{kk}"] = np.where(rng.random(n) < 0.8, y, rng.integers(0, 26, n)).astype(np.int8)
    return KData(probs * 25.0, y, knn)


def _setup():
    rng = np.random.default_rng(0)
    runs = np.repeat([f"r{i:03d}" for i in range(R)], W)
    data = SplitData(0, "val", [f"{r}_w{21 + j:04d}" for r in sorted(set(runs)) for j in range(W)], runs,
                     {0: _kdata(rng, 1.6), 10: _kdata(rng, 1.2)}, {0: 0.9, 10: 0.7})
    A = np.isin(runs, [f"r{i:03d}" for i in range(0, R, 2)])
    per_k = {str(k): fit_k(data.by_k[k], A, ~A) for k in data.by_k}
    return Gate({"supported_patterns": ["random"], "per_k": per_k}), data, A


def test_evaluate_bookkeeping_and_determinism() -> None:
    gate, data, A = _setup()
    a, b = evaluate(gate, data, ~A), evaluate(gate, data, ~A)
    assert a == b                                                         # deterministic (fixed bootstrap seed)
    for k, r in a["per_k"].items():
        n = r["gate_acc95"]["n_windows"]
        assert n == int((~A).sum()) and sum(r["gate_acc95"]["outcomes"].values()) == n
        g = r["gate_acc95"]
        lo, hi = g["coverage_ci"]
        assert lo <= g["coverage"] <= hi
        assert 0 <= r["conformal_90"]["coverage_all_post_fault"] <= 1
        if g["n_windows"] and g["coverage"] > 0:
            assert g["accuracy_confident"] >= 0.95 - 0.02                 # threshold accuracy on cal-B itself (selection set)
        assert "logistic_combination_p+margin+agreement" in r["error_detection"]
        e = r["error_detection"]["isotonic_prob"]
        assert e["auroc_ci"][0] <= e["auroc"] <= e["auroc_ci"][1] and 0.5 < e["auroc"] <= 1


def test_k_without_parameters_is_reported_as_rejected() -> None:
    gate, data, A = _setup()
    gate.per_k.pop(10)
    r = evaluate(gate, data, ~A, boot=False)
    assert r["per_k"]["10"]["gate"].startswith("REJECT: UNSUPPORTED_DROPOUT_PATTERN") and "gate_acc95" in r["per_k"]["0"]
