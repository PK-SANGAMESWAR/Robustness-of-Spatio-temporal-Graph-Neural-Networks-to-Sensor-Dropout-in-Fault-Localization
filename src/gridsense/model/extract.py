"""Extract soft votes and per-node latents with the paper's evaluation loop (S2).

Mirrors fault_localization/dropout/evaluator.py:evaluate_dropout_curve: same loader (EventChunkBatchSampler,
shuffle=False), batch size 32, mask seed (10000*k + offset)*10000 + batch_idx, policy zero_messages, soft vote.
Run on CPU: `nice -n 19 env PYTHONPATH=src:. uv run python -m gridsense.model.extract ...` (4 torch threads).
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from gridsense.config import ROOT, TRACK
from gridsense.data.splits import DATA_ROOT, split_events, stratified_run_subsample
from gridsense.model.latent_repr import METHODS, load_scaler_pca, represent
from gridsense.model.predictor import LatentCapture, checkpoint_sha256, lean_dataset, load_model

OUT_ROOT = TRACK / "eval" / "latents"


def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def extract(seed: int, split: str, k_levels: list[int], n_offsets: int, latents: bool, run_fraction: float | None,
            batch_size: int = 32, threads: int = 4, limit_batches: int | None = None, out_root: Path = OUT_ROOT,
            latent_repr: str = "full", offset_start: int = 0) -> Path:
    from fault_localization.data.dataset import make_dataloader
    from fault_localization.dropout.masking import apply_sensor_dropout_batch
    from fault_localization.models.voting import soft_vote

    torch.set_num_threads(threads)
    keep = None
    tag = split
    if run_fraction is not None:
        keep = set(stratified_run_subsample(split_events(split), run_fraction, seed=42))
        tag = f"{split}_sub{int(round(run_fraction * 100))}"
    ds = lean_dataset(split, DATA_ROOT, keep_events=keep)
    model, cfg = load_model(seed, ds.edge_index, "cpu")
    cap = LatentCapture(model)
    loader = make_dataloader(ds, batch_size, shuffle=False)
    n = len(ds) if limit_batches is None else min(len(ds), limit_batches * batch_size)
    combos = [(k, off) for k in k_levels for off in (range(offset_start, offset_start + n_offsets) if k > 0 else [0])]

    out = out_root / f"seed{seed}" / tag
    out.mkdir(parents=True, exist_ok=True)
    votes = {c: np.lib.format.open_memmap(out / f"votes_k{c[0]}_o{c[1]}.npy", "w+", np.float32, (n, 26)) for c in combos}
    pca = load_scaler_pca(out_root / f"seed{seed}" / "scaler_pca256_train_sub10.npz") if latent_repr == "c_concat_pca256" else None
    shape = {"full": (n, 25, 64), "a_concat": (n, 1600), "b_mean_max": (n, 128), "c_concat_pca256": (n, 256)}[latent_repr]
    hs = ({c: np.lib.format.open_memmap(out / f"h_k{c[0]}_o{c[1]}.npy", "w+", np.float32, shape) for c in combos}
          if latents else {})
    labels = np.lib.format.open_memmap(out / "labels.npy", "w+", np.int8, (n,))
    window_ids: list[str] = []
    edge_index = getattr(model, "edge_index", None)

    t0 = time.time()
    pos = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            x = batch["x"]
            b = x.shape[0]
            labels[pos:pos + b] = batch["y"].numpy()
            window_ids.extend(batch["window_id"])
            for (k, off) in combos:
                if k == 0:
                    xd = x
                else:
                    xd, _, _ = apply_sensor_dropout_batch(x, k=k, seed=(10000 * k + off) * 10000 + batch_idx,
                                                          policy="zero_messages", edge_index=edge_index)
                scores, _ = soft_vote(model(xd))
                votes[(k, off)][pos:pos + b] = scores.numpy()
                if latents:
                    hh = cap.h.numpy()
                    hs[(k, off)][pos:pos + b] = hh if latent_repr == "full" else represent(hh, latent_repr, pca)
            pos += b
            if batch_idx % 1000 == 0:
                print(f"[{tag}] batch {batch_idx} windows {pos}/{n} {time.time() - t0:.0f}s", flush=True)
    if pos != n:
        raise RuntimeError(f"wrote {pos} windows, expected {n}")
    for arr in [*votes.values(), *hs.values(), labels]:
        arr.flush()
    ids_file = out / "window_ids.txt"
    new_ids = "\n".join(window_ids) + "\n"
    if ids_file.exists() and ids_file.read_text() != new_ids:
        raise RuntimeError(f"{ids_file} differs from this run's window order; refusing to overwrite")
    ids_file.write_text(new_ids)
    meta = {"seed": seed, "split": split, "tag": tag, "n_windows": n, "k_levels": k_levels, "n_mask_offsets": n_offsets, "offset_start": offset_start,
            "combos": [list(c) for c in combos], "batch_size": batch_size, "policy": "zero_messages",
            "loader": "EventChunkBatchSampler(shuffle=False)", "run_fraction": run_fraction, "run_subsample_seed": 42,
            "latents": latents, "latent_repr": latent_repr,
            "latent": "model.classifier input h [25, 64], float32 (float16 overflows: |h| > 65504); repr per latent_repr",
            "checkpoint_sha256": checkpoint_sha256(seed), "torch_threads": threads, "device": "cpu",
            "wall_seconds": round(time.time() - t0, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "git_commit": _git_commit(), "limit_batches": limit_batches}
    meta_name = "extract_meta.json" if k_levels == [0] else f"extract_meta_k{'-'.join(map(str, k_levels))}" + (f"_o{offset_start}" if offset_start else "") + ".json"
    (out / meta_name).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("tag", "n_windows", "wall_seconds", "peak_rss_mb")}), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--k", type=int, nargs="+", default=[0])
    p.add_argument("--offsets", type=int, default=3, help="mask seeds per k > 0 (paper: 3)")
    p.add_argument("--offset-start", type=int, default=0, help="first mask offset (paper: 0,1,2; O17 uses 3..7)")
    p.add_argument("--latents", action="store_true")
    p.add_argument("--latent-repr", choices=["full", *METHODS], default="full", help="what to store for --latents (O4: b_mean_max)")
    p.add_argument("--run-fraction", type=float, default=None, help="whole-run stratified subsample (O15)")
    p.add_argument("--batch-size", type=int, default=32, help="paper protocol: 32; other values only for the mask-realisation diagnostic")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = p.parse_args()
    extract(a.seed, a.split, a.k, a.offsets, a.latents, a.run_fraction, batch_size=a.batch_size, threads=a.threads,
            limit_batches=a.limit_batches, out_root=a.out_root, latent_repr=a.latent_repr, offset_start=a.offset_start)


if __name__ == "__main__":
    main()
