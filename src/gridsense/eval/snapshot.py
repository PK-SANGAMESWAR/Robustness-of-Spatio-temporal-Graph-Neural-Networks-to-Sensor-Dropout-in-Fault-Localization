"""Snapshot writer (IMPLEMENTATION.md § 7; rag-eval.md § 8.1). Snapshots are append-only: a file is never overwritten."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gridsense.config import ROOT, TRACK

SNAP_DIR, ITEMS_DIR = TRACK / "eval" / "snapshots", TRACK / "eval" / "items"
KEYS = ("git_commit", "git_dirty", "label", "timestamp_utc", "config_effective", "index_fingerprint", "case_store_fingerprint",
        "prompt_hashes", "rubric_hashes", "golden_versions", "judge_model", "generation_model", "deepeval_version", "metrics",
        "n_items", "n_failures", "failure_causes", "tokens", "cost_usd", "langsmith_experiment_url", "items_path")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def write_snapshot(label: str, metrics: dict, items: list[dict], **fields) -> Path:
    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    snap, items_path = SNAP_DIR / f"{stamp}_{label}.json", ITEMS_DIR / f"{stamp}_{label}.jsonl"
    if snap.exists() or items_path.exists():
        raise FileExistsError(f"{snap.name} exists; snapshots are append-only")
    SNAP_DIR.mkdir(parents=True, exist_ok=True), ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    items_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    body = {k: None for k in KEYS}
    body.update(git_commit=_git("rev-parse", "--short", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--", "src/gridsense", "rag-for-dropout/configs", "rag-for-dropout/corpus")),
                label=label, timestamp_utc=ts.strftime("%Y-%m-%dT%H:%M:%SZ"), metrics=flatten(metrics), n_items=len(items),
                n_failures=sum(1 for r in items if r.get("error")), failure_causes=sorted({r["error"] for r in items if r.get("error")}),
                tokens={}, cost_usd=0.0, items_path=str(items_path.relative_to(ROOT)))
    unknown = set(fields) - set(KEYS) - {"preliminary", "preliminary_reason", "notes"}
    if unknown:
        raise KeyError(f"unknown snapshot fields {sorted(unknown)}")
    body.update(fields)
    snap.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return snap
