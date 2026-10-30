"""Typed access to rag-for-dropout/configs/eval.yaml. One config file for the track (IMPLEMENTATION.md § 1)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "rag-for-dropout"
CONFIG_PATH = TRACK / "configs" / "eval.yaml"


class OpenItemError(RuntimeError):
    """A value that the author has not chosen yet (configs/eval.yaml: null with '# OPEN')."""


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def require(cfg: dict[str, Any], dotted: str) -> Any:
    """Return cfg[a][b]...; raise OpenItemError when the value is null."""
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    if node is None:
        raise OpenItemError(f"configs/eval.yaml '{dotted}' is not set (see OPEN_ITEMS.md)")
    return node


def torch_threads() -> int:
    """Torch threads for CPU jobs: `rag-for-dropout/eval/threads.txt` if it exists, else 4 (author rule: 4; 2 if the S4.4 epoch time rises > 10 %)."""
    f = TRACK / "eval" / "threads.txt"
    return int(f.read_text().strip()) if f.exists() else 4
