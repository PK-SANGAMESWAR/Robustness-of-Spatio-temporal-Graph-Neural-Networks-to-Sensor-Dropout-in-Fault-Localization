"""One-run pipeline for the sealed OOD test set (``PC-STGNN/08_STAGE_G_eval.md`` S6.1, amended 2026-11-02).

In order, in one run:

1. refuse unless every precondition holds (author go-ahead given, no completed run, pre-registration
   files committed, frozen-model manifest matches the checkpoints, no GPU job running);
2. verify the sealed directory against ``data/ood_test.sha256``;
3. write ``STARTED.json`` (from here on a rerun needs ``--resume``);
4. make windows, labels and metadata from the raw simulation **outside** the sealed directory,
   with the same functions that built the validation split (``processing.windowing``,
   ``processing.labels``) and the training corpus's graph and normalisation statistics;
5. score every model of the manifest on the OOD grid and save every prediction;
6. write the log row for ``14_PREREGISTRATION.md`` section 6 (the date is the run's own clock);
7. write ``COMPLETED.json`` (sha256 of every output) and make the outputs read-only.

After ``COMPLETED.json`` exists the pipeline refuses to run again.  The sealed directory is
read-only data: nothing is written into it.  No function here is called at import time and no
test touches the real sealed path; tests and the pilot smoke run use other directories
(``smoke=True`` refuses the real sealed path).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEALED_DIR = REPO / "data" / "ood_test"
HASH_FILE = REPO / "data" / "ood_test.sha256"
DEFAULT_OUT = REPO / "results" / "stage_g" / "ood"
CORPUS_ROOT = REPO / "dataset_fullscale_50"
MANIFEST = REPO / "config" / "stage_g" / "frozen_models.json"
PREREG = REPO / "PC-STGNN" / "14_PREREGISTRATION.md"
STAGE_G = REPO / "PC-STGNN" / "08_STAGE_G_eval.md"
#: strings the committed pre-registration must already contain (amendments of 2026-11-02)
REQUIRED_TEXT = {PREREG: ["Gate 6", "grounded"], STAGE_G: ["Gate 6", "grounded"]}
LOCK_FILES = [REPO / "results/s4_4/queue.lock", REPO / "results/stage_f/gtf2/watcher.lock"]
GRAPH_FILES = ["edge_index.npy", "edge_attr.npy", "node_mapping.json", "norm_stats.json"]
WINDOWS_PER_EVENT = 40
EXPECTED_SEALED_EVENTS = 11000


class Refused(RuntimeError):
    """A precondition failed; nothing was written."""


@dataclass
class PipelineConfig:
    sealed_dir: Path = SEALED_DIR
    hash_file: Path | None = HASH_FILE
    out_dir: Path = DEFAULT_OUT
    corpus_root: Path = CORPUS_ROOT
    manifest: Path = MANIFEST
    expected_events: int | None = EXPECTED_SEALED_EVENTS
    go_ahead: str | None = None
    resume: bool = False
    smoke: bool = False                   # pilot / test mode: relaxed checks, real sealed path refused
    device: str = "cuda"
    max_windows: int | None = None        # smoke only
    grid: str = "ood"
    check_git: bool = True
    check_locks: bool = True
    prereg_text: dict = field(default_factory=lambda: dict(REQUIRED_TEXT))

    @property
    def is_real_sealed(self) -> bool:
        return Path(self.sealed_dir).resolve() == SEALED_DIR.resolve()


# ---------------------------------------------------------------------------------------
# Hash (same algorithm as scripts/generate_ood_sets.py::sha256_of_dir; test asserts equality)
# ---------------------------------------------------------------------------------------


def sha256_of_dir(dir_path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in Path(dir_path).rglob("*") if p.is_file()):
        h.update(p.relative_to(dir_path).as_posix().encode("utf-8"))
        fh_ = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                fh_.update(chunk)
        h.update(fh_.digest())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> dict:
    utc = dt.datetime.now(dt.timezone.utc)
    ist = utc.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    return {"ist": ist.strftime("%Y-%m-%d %H:%M IST"), "utc": utc.strftime("%Y-%m-%dT%H:%M:%SZ")}


def _rel(path) -> Path:
    try:
        return Path(path).resolve().relative_to(REPO)
    except ValueError:
        return Path(path)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


# ---------------------------------------------------------------------------------------
# Frozen-model manifest
# ---------------------------------------------------------------------------------------


def freeze_manifest(entries: list[dict], out_path: Path) -> dict:
    """Write the list of models to score, each with the sha256 of its checkpoint. Never overwrites.

    ``entries``: ``{"name": "r5", "seed": 42, "run_dir": "results/s4_4/r5/pcstgnn/seed_00042"}``.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; a frozen manifest is not overwritten")
    rows = []
    for e in entries:
        ckpt = REPO / e["run_dir"] / "checkpoint_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        rows.append({**e, "checkpoint_sha256": _sha256_file(ckpt)})
    manifest = {"frozen": _now(), "git_head": _git("rev-parse", "HEAD"), "entries": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not data.get("entries"):
        raise Refused(f"manifest {path} has no entries")
    return data["entries"]


# ---------------------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------------------


def _pid_alive(lock: Path) -> bool:
    try:
        pid = int(Path(lock).read_text().strip().split()[0])
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, IndexError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def preflight(cfg: PipelineConfig) -> list[str]:
    """Return the list of failed preconditions. Touches nothing inside the sealed directory."""
    bad: list[str] = []
    out = Path(cfg.out_dir)
    if cfg.smoke and cfg.is_real_sealed:
        bad.append("smoke mode must not point at the real sealed directory")
    if not cfg.smoke:
        if not cfg.go_ahead or not cfg.go_ahead.strip():
            bad.append("no author go-ahead given (--author-go-ahead)")
        if cfg.hash_file is None:
            bad.append("a hash file is required outside smoke mode")
    if (out / "COMPLETED.json").exists():
        bad.append("COMPLETED.json exists: the sealed set was opened once already; this pipeline refuses to run again")
    elif (out / "STARTED.json").exists() and not cfg.resume:
        bad.append("STARTED.json exists without COMPLETED.json: a previous run did not finish; use --resume")
    if not Path(cfg.manifest).exists():
        bad.append(f"frozen-model manifest missing: {cfg.manifest}")
    else:
        try:
            for e in load_manifest(cfg.manifest):
                ckpt = REPO / e["run_dir"] / "checkpoint_best.pt"
                if not ckpt.exists():
                    bad.append(f"checkpoint missing: {ckpt}")
                elif _sha256_file(ckpt) != e["checkpoint_sha256"]:
                    bad.append(f"checkpoint changed since freeze: {e['name']} seed {e['seed']}")
        except Refused as exc:
            bad.append(str(exc))
    if cfg.check_git and not cfg.smoke:
        for f, needles in cfg.prereg_text.items():
            rel = _rel(f)
            if _git("status", "--porcelain", "--", str(rel)):
                bad.append(f"{rel} has uncommitted changes")
            text = Path(f).read_text() if Path(f).exists() else ""
            for needle in needles:
                if needle not in text:
                    bad.append(f"{rel} does not contain the amendment text {needle!r}")
    if cfg.check_locks and not cfg.smoke:
        for lock in LOCK_FILES:
            if _pid_alive(lock):
                bad.append(f"a GPU job holds {_rel(lock)}")
    return bad


def verify_hash(cfg: PipelineConfig) -> str:
    recorded = Path(cfg.hash_file).read_text().split()[0]
    computed = sha256_of_dir(Path(cfg.sealed_dir))
    if computed != recorded:
        raise Refused(f"sealed directory hash {computed} != recorded {recorded}")
    return computed


# ---------------------------------------------------------------------------------------
# Windows, labels, metadata -- same functions as the validation split
# ---------------------------------------------------------------------------------------


def build_eval_root(raw_dir: Path, eval_root: Path, corpus_root: Path, expected_events: int | None) -> dict:
    """``raw_sim`` -> ``eval_root/processed`` + ``splits/all.txt``, readable by ``WindowGraphDataset(eval_root, "all")``.

    Graph and normalisation come from the training corpus (the checkpoints were fit against them).
    ``eval_root`` must not be inside the raw directory's parent (the sealed set is not written to).
    """
    from fault_localization.processing.labels import assign_labels
    from fault_localization.processing.windowing import process_all_events

    raw_dir, eval_root, corpus_root = Path(raw_dir), Path(eval_root), Path(corpus_root)
    if eval_root.resolve() == raw_dir.resolve().parent or raw_dir.resolve().parent in eval_root.resolve().parents:
        raise Refused("the evaluation root must be outside the directory that holds raw_sim")
    n_raw = len(list(raw_dir.glob("*.npz")))
    if expected_events is not None and n_raw != expected_events:
        raise Refused(f"{n_raw} raw events, expected {expected_events}")
    processed = eval_root / "processed"
    (eval_root / "splits").mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_FILES:
        shutil.copy2(corpus_root / "processed" / name, processed / name)
    results = process_all_events(raw_dir, processed / "windows")
    if len(results) != n_raw:
        raise Refused(f"windowing produced {len(results)} events from {n_raw} raw files (process_all_events logs and skips failures)")
    if any(n != WINDOWS_PER_EVENT for _e, n in results):
        raise Refused(f"an event does not have {WINDOWS_PER_EVENT} windows")
    labels, meta = assign_labels(raw_dir, processed / "labels.csv", processed / "metadata.csv")
    ids = [r["window_id"] for r in labels]
    (eval_root / "splits" / "all.txt").write_text("\n".join(ids) + "\n")
    n_zero = sum(1 for r in labels if int(r["label"]) == 0)
    if n_zero * 2 != len(labels):
        raise Refused(f"no-fault share is {n_zero}/{len(labels)}, expected exactly one half")
    res = sorted({float(r["resistance"]) for r in meta})
    return {"n_events": n_raw, "n_windows": len(ids), "n_no_fault": n_zero, "resistances": res,
            "fault_types": sorted({r["fault_type"] for r in meta})}


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------


def log_row(cfg: PipelineConfig, when: dict, head: str, computed: str, recorded: str) -> str:
    ok = "yes" if computed == recorded else "NO"
    return (f"| `data/ood_test/` opened (Stage G, once) | {when['ist']} / {when['utc']} | {cfg.go_ahead}; "
            f"`scripts/run_stage_g_ood.py` at commit `{head}` | sha256 `{computed}` = `{recorded}`: {ok} |\n")


def run(cfg: PipelineConfig) -> dict:
    from fault_localization.data.dataset import WindowGraphDataset
    from fault_localization.stage_g import scoring as S
    import torch

    bad = preflight(cfg)
    if bad:
        raise Refused("; ".join(bad))
    out = Path(cfg.out_dir)
    when = _now()
    head = _git("rev-parse", "--short", "HEAD")

    if cfg.smoke:
        computed = recorded = "smoke-no-hash"
    else:
        recorded = Path(cfg.hash_file).read_text().split()[0]
        computed = verify_hash(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "STARTED.json").exists():
        (out / "STARTED.json").write_text(json.dumps(
            {"when": when, "git_head": head, "go_ahead": cfg.go_ahead, "sealed_dir": str(cfg.sealed_dir),
             "hash_computed": computed, "hash_recorded": recorded, "smoke": cfg.smoke}, indent=2) + "\n")
    try:
        eval_root = out / "eval_root"
        if (eval_root / "splits" / "all.txt").exists() and cfg.resume:
            stats = json.loads((out / "eval_root_stats.json").read_text())
        else:
            stats = build_eval_root(Path(cfg.sealed_dir) / "raw_sim", eval_root, cfg.corpus_root, cfg.expected_events)
            (out / "eval_root_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

        ds = WindowGraphDataset(str(eval_root), "all")
        device = torch.device(cfg.device)
        subset = None
        if cfg.max_windows is not None:
            subset = {r[0] for r in ds.records[:cfg.max_windows]}
        outputs, entries = [], load_manifest(cfg.manifest)
        for e in entries:
            name = f"{e['name']}_seed{int(e['seed']):05d}"
            if (out / f"{name}_scoring.json").exists():
                outputs.append(out / f"{name}_scoring.json")
                continue
            model, _cfg = S.load_model(REPO / e["run_dir"], ds.edge_index, device)
            res = S.score_dataset(model, ds, S.GRIDS[cfg.grid], device=device, window_subset=subset,
                                  progress_every=500)
            meta = {"model": e["name"], "seed": e["seed"], "run_dir": e["run_dir"],
                    "checkpoint_sha256": e["checkpoint_sha256"], "split": "ood_test" if not cfg.smoke else "ood_pilot_smoke",
                    "grid": cfg.grid, "resistances": stats["resistances"]}
            outputs.append(S.write_summary(out, name, res, meta, resistances=stats["resistances"]))
            S.release(model)

        (out / "log_row.md").write_text(log_row(cfg, when, head, computed, recorded))
        files = sorted(p for p in out.rglob("*") if p.is_file() and "eval_root" not in p.relative_to(out).parts)
        completed = {"when": _now(), "git_head": head, "n_models": len(entries), "eval_root": stats,
                     "outputs": {p.relative_to(out).as_posix(): _sha256_file(p) for p in files
                                 if p.name != "COMPLETED.json"}}
        (out / "COMPLETED.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
        for p in files + [out / "COMPLETED.json"]:
            os.chmod(p, 0o444)
        return completed
    except Exception:
        (out / "FAILED.json").write_text(json.dumps({"when": _now(), "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
