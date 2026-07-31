"""Content-addressed inventory for an experiment benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def create_manifest(data_root: str | Path, benchmark_version: str = "benchmark_v1.0") -> dict[str, Any]:
    """Return a deterministic fingerprint of every benchmark input file."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = _files(root)
    payload = {"schema_version": 1, "benchmark_version": benchmark_version,
               "data_root": str(root), "files": files}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    payload["counts"] = {
        "window_files": sum(k.startswith("processed/windows/") and k.endswith(".npz") for k in files),
        "raw_files": sum(k.startswith("raw_sim/") and k.endswith(".npz") for k in files),
        "files": len(files),
    }
    return payload


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def verify_manifest(data_root: str | Path, expected: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Fail closed when current inputs do not match the frozen fingerprint."""
    if isinstance(expected, (str, Path)):
        expected = json.loads(Path(expected).read_text())
    current = create_manifest(data_root, expected.get("benchmark_version", "benchmark_v1.0"))
    if current["fingerprint"] != expected.get("fingerprint"):
        raise RuntimeError("Benchmark fingerprint mismatch; create a new benchmark version instead of overwriting evidence.")
    return current
