"""One-run sealed-set pipeline, tested on a **fake sealed directory made of pilot events**.

Nothing here touches ``data/ood_test/``.  The pilot (``data/ood_pilot/``) is the throwaway Gate-2 set; its
numbers are not reported.  CPU only.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data/ood_pilot"
sys.path.insert(0, str(REPO / "scripts"))

from fault_localization.stage_g import ood_pipeline as P  # noqa: E402

N_EVENTS = 6


@pytest.fixture()
def fake_sealed(tmp_path):
    """A stand-in 'sealed' directory with six pilot events and its hash file."""
    sealed = tmp_path / "sealed" / "ood_fake"
    (sealed / "raw_sim").mkdir(parents=True)
    for f in sorted((PILOT / "raw_sim").glob("*.npz"))[:N_EVENTS]:
        shutil.copy2(f, sealed / "raw_sim" / f.name)
    hash_file = tmp_path / "sealed" / "ood_fake.sha256"
    hash_file.write_text(f"{P.sha256_of_dir(sealed)}  ood_fake/\n")
    return sealed, hash_file


@pytest.fixture()
def manifest(tmp_path):
    out = tmp_path / "frozen_models.json"
    P.freeze_manifest([{"name": "r0n", "seed": 42, "run_dir": "results/s4_4/r0n/pcstgnn/seed_00042"}], out)
    return out


def _cfg(tmp_path, fake_sealed, manifest, **kw):
    sealed, hash_file = fake_sealed
    base = dict(sealed_dir=sealed, hash_file=hash_file, out_dir=tmp_path / "out", manifest=manifest,
                expected_events=N_EVENTS, go_ahead="test go-ahead (unit test)", device="cpu", max_windows=48,
                check_git=False, check_locks=False)
    base.update(kw)
    return P.PipelineConfig(**base)


# --------------------------------------------------------------------------- hash
def test_hash_function_equals_the_one_that_sealed_the_set(tmp_path):
    import generate_ood_sets as G
    d = tmp_path / "d"
    (d / "a").mkdir(parents=True)
    (d / "a" / "x.bin").write_bytes(b"123")
    (d / "y.bin").write_bytes(b"abc" * 100000)
    assert P.sha256_of_dir(d) == G.sha256_of_dir(d)


# --------------------------------------------------------------------------- windows and labels
def test_windows_labels_metadata_equal_the_pilot_processed_files(tmp_path, fake_sealed):
    sealed, _ = fake_sealed
    stats = P.build_eval_root(sealed / "raw_sim", tmp_path / "eval", REPO / "dataset_fullscale_50", N_EVENTS)
    assert stats["n_events"] == N_EVENTS and stats["n_windows"] == 40 * N_EVENTS and stats["n_no_fault"] == 20 * N_EVENTS
    proc_new, proc_ref = tmp_path / "eval/processed", PILOT / "processed"
    for f in sorted((sealed / "raw_sim").glob("*.npz")):
        ev = f.stem
        with np.load(proc_new / "windows" / f"{ev}.npz") as a, np.load(proc_ref / "windows" / f"{ev}.npz") as b:
            assert np.array_equal(a["windows"], b["windows"]), ev
            assert np.array_equal(a["window_ids"], b["window_ids"]), ev

    def rows(path):
        with open(path) as fh:
            return {r["window_id"]: r for r in csv.DictReader(fh)}
    for name in ("labels.csv", "metadata.csv"):
        new, ref = rows(proc_new / name), rows(proc_ref / name)
        assert set(new) <= set(ref)
        assert all(new[w] == ref[w] for w in new)
    ids = (tmp_path / "eval/splits/all.txt").read_text().split()
    assert ids == sorted(ids) and len(ids) == 40 * N_EVENTS
    for name in P.GRAPH_FILES:                                   # graph and normalisation are the training corpus's
        assert (proc_new / name).read_bytes() == (REPO / "dataset_fullscale_50/processed" / name).read_bytes()


def test_eval_root_must_not_be_inside_the_sealed_directory(tmp_path, fake_sealed):
    sealed, _ = fake_sealed
    with pytest.raises(P.Refused):
        P.build_eval_root(sealed / "raw_sim", sealed / "derived", REPO / "dataset_fullscale_50", N_EVENTS)
    with pytest.raises(P.Refused):
        P.build_eval_root(sealed / "raw_sim", tmp_path / "e2", REPO / "dataset_fullscale_50", N_EVENTS + 1)


# --------------------------------------------------------------------------- refusals
def test_refuses_without_go_ahead_and_touches_nothing(tmp_path, manifest):
    missing = tmp_path / "does_not_exist"
    cfg = P.PipelineConfig(sealed_dir=missing, hash_file=tmp_path / "nope.sha256", out_dir=tmp_path / "out",
                           manifest=manifest, go_ahead=None, check_git=False, check_locks=False, device="cpu")
    with pytest.raises(P.Refused, match="go-ahead"):
        P.run(cfg)
    assert not missing.exists() and not (tmp_path / "out").exists()


def test_smoke_mode_refuses_the_real_sealed_path(tmp_path, manifest):
    cfg = P.PipelineConfig(sealed_dir=P.SEALED_DIR, hash_file=None, out_dir=tmp_path / "out", manifest=manifest,
                           smoke=True, check_git=False, check_locks=False, device="cpu")
    with pytest.raises(P.Refused, match="real sealed"):
        P.run(cfg)
    assert not (tmp_path / "out").exists()


def test_refuses_when_a_checkpoint_changed_since_the_freeze(tmp_path, fake_sealed, manifest):
    data = json.loads(manifest.read_text())
    data["entries"][0]["checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(P.Refused, match="changed since freeze"):
        P.run(_cfg(tmp_path, fake_sealed, manifest))
    assert not (tmp_path / "out").exists()


def test_refuses_when_a_gpu_job_holds_its_lock(tmp_path, fake_sealed, manifest, monkeypatch):
    lock = tmp_path / "queue.lock"
    lock.write_text(str(os.getpid()))
    monkeypatch.setattr(P, "LOCK_FILES", [lock])
    with pytest.raises(P.Refused, match="GPU job"):
        P.run(_cfg(tmp_path, fake_sealed, manifest, check_locks=True))


def test_hash_mismatch_stops_before_anything_is_written(tmp_path, fake_sealed, manifest):
    sealed, hash_file = fake_sealed
    (sealed / "raw_sim" / "tamper.txt").write_text("changed after sealing")
    with pytest.raises(P.Refused, match="hash"):
        P.run(_cfg(tmp_path, fake_sealed, manifest))
    assert not (tmp_path / "out").exists()


def test_unfrozen_or_uncommitted_prereg_is_refused(tmp_path, fake_sealed, manifest):
    needles = {tmp_path / "prereg.md": ["Gate 6"]}
    (tmp_path / "prereg.md").write_text("no amendment here")
    cfg = _cfg(tmp_path, fake_sealed, manifest, check_git=True, prereg_text=needles)
    bad = P.preflight(cfg)
    assert any("does not contain" in b or "relative" in b.lower() for b in bad) or bad


# --------------------------------------------------------------------------- the whole run
def test_one_run_then_refuses_to_run_again(tmp_path, fake_sealed, manifest):
    cfg = _cfg(tmp_path, fake_sealed, manifest)
    done = P.run(cfg)
    out = tmp_path / "out"
    for name in ("STARTED.json", "COMPLETED.json", "log_row.md", "eval_root_stats.json",
                 "r0n_seed00042_scoring.json", "r0n_seed00042_predictions.npz"):
        assert (out / name).exists(), name
    assert not (out / "FAILED.json").exists()
    row = (out / "log_row.md").read_text()
    sealed, hash_file = fake_sealed
    digest = hash_file.read_text().split()[0]
    assert digest in row and "test go-ahead" in row and row.count("yes") >= 1 and "opened (Stage G, once)" in row
    summ = json.loads((out / "r0n_seed00042_scoring.json").read_text())
    assert summ["n_windows"] == 48 and set(summ["cells"]) == {
        "k0_sigma0", "k0_sigma0.0005", "k0_sigma0.002", "k0_sigma0.01", "k5_sigma0", "k10_sigma0"}
    pred = np.load(out / "r0n_seed00042_predictions.npz")
    assert pred["pf__k0_s0.01_o2"].shape == (48,) and pred["window_ids"].shape == (48,)
    completed = json.loads((out / "COMPLETED.json").read_text())
    assert "r0n_seed00042_predictions.npz" in completed["outputs"] and done["n_models"] == 1
    assert not os.access(out / "COMPLETED.json", os.W_OK)          # outputs are read-only
    with pytest.raises(P.Refused, match="COMPLETED"):
        P.run(cfg)
    with pytest.raises(P.Refused, match="COMPLETED"):
        P.run(_cfg(tmp_path, fake_sealed, manifest, resume=True))


def test_started_without_completed_needs_resume(tmp_path, fake_sealed, manifest):
    out = tmp_path / "out"
    out.mkdir()
    (out / "STARTED.json").write_text("{}")
    with pytest.raises(P.Refused, match="resume"):
        P.run(_cfg(tmp_path, fake_sealed, manifest))


def test_freeze_manifest_never_overwrites(tmp_path):
    out = tmp_path / "m.json"
    P.freeze_manifest([{"name": "r0n", "seed": 42, "run_dir": "results/s4_4/r0n/pcstgnn/seed_00042"}], out)
    with pytest.raises(FileExistsError):
        P.freeze_manifest([{"name": "r0n", "seed": 42, "run_dir": "results/s4_4/r0n/pcstgnn/seed_00042"}], out)
    assert len(json.loads(out.read_text())["entries"][0]["checkpoint_sha256"]) == 64
