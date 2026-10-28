"""k > 0 check (author, S2): the dropout masks must come from the paper's seeded generator.

Runs the paper's own `evaluate_dropout_curve` and `gridsense.model.extract` on the same first N test batches, records
every call to `apply_sensor_dropout_batch` (k, seed, dropped nodes per sample) in both runs, and compares the call
sequences exactly. Also compares macro-F1 per (k, mask offset) on those windows. CPU only.
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from gridsense.config import TRACK
from gridsense.data.splits import DATA_ROOT
from gridsense.model.extract import extract
from gridsense.model.predictor import lean_dataset, load_model

K_LEVELS = [1, 2, 5, 10]
N_OFFSETS = 3


def _spy(module, calls: list):
    orig = module.apply_sensor_dropout_batch

    def wrapped(x, k, seed, policy="zero_messages", edge_index=None):
        out = orig(x, k=k, seed=seed, policy=policy, edge_index=edge_index)
        calls.append((int(k), int(seed), [list(map(int, d)) for d in out[2]]))
        return out

    module.apply_sensor_dropout_batch = wrapped
    return lambda: setattr(module, "apply_sensor_dropout_batch", orig)


def run(n_batches: int = 40, seed: int = 42) -> dict:
    import fault_localization.dropout.evaluator as paper_eval
    import fault_localization.dropout.masking as masking
    from fault_localization.data.dataset import make_dataloader

    torch.set_num_threads(4)
    ds = lean_dataset("test", DATA_ROOT)
    model, _ = load_model(seed, ds.edge_index, "cpu")
    loader = make_dataloader(ds, 32, shuffle=False)

    paper_calls: list = []
    restore = _spy(paper_eval, paper_calls)
    try:
        paper = paper_eval.evaluate_dropout_curve(model, list(itertools.islice(loader, n_batches)), torch.device("cpu"),
                                                  dropout_levels=K_LEVELS, n_mask_seeds=N_OFFSETS)
    finally:
        restore()

    mine_calls: list = []
    restore = _spy(masking, mine_calls)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = extract(seed, "test", K_LEVELS, N_OFFSETS, latents=False, run_fraction=None, limit_batches=n_batches, out_root=Path(tmp))
            y = np.load(out / "labels.npy").astype(int)
            # Same call as the paper's evaluate_dropout_curve: no `labels=`, so the macro average is over the classes present in
            # y_true or the predictions (a 1,280-window subset has few). On the full test split all 26 classes are present.
            f1 = {(k, o): float(f1_score(y, np.load(out / f"votes_k{k}_o{o}.npy").argmax(1), average="macro", zero_division=0))
                  for k in K_LEVELS for o in range(N_OFFSETS)}
    finally:
        restore()

    calls_equal = paper_calls == mine_calls
    f1_rows = []
    for k in K_LEVELS:
        for o in range(N_OFFSETS):
            f1_rows.append({"k": k, "offset": o, "f1_paper_function": paper[k]["f1_per_seed"][o], "f1_gridsense": f1[(k, o)],
                            "abs_diff": abs(paper[k]["f1_per_seed"][o] - f1[(k, o)])})
    return {"n_batches": n_batches, "n_windows": n_batches * 32, "n_mask_calls_paper": len(paper_calls), "n_mask_calls_gridsense": len(mine_calls),
            "mask_calls_identical": calls_equal, "first_call": {"k": paper_calls[0][0], "seed": paper_calls[0][1]},
            "seed_formula": "(10000*k + offset)*10000 + batch_idx", "f1": f1_rows, "max_f1_abs_diff": max(r["abs_diff"] for r in f1_rows),
            "passes": calls_equal and max(r["abs_diff"] for r in f1_rows) <= 1e-3}


if __name__ == "__main__":
    res = run(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    (TRACK / "reports" / "s2").mkdir(parents=True, exist_ok=True)
    (TRACK / "reports" / "s2" / "mask_check_seed42.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "f1"}, indent=2))
    sys.exit(0 if res["passes"] else 1)
