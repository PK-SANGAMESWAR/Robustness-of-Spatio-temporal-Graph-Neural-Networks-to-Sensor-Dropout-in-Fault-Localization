"""Persistence for the workflow (spec/impl_08.md § 12.3: persist graph state when human review may happen later).

`sqlite_saver(path)` is LangGraph's own SQLite checkpoint saver (package `langgraph-checkpoint-sqlite`, added to the `rag` group with the author's approval, 2026-11-18).
`JsonExportStore` keeps the idempotent export of approved reports (one record per event ID).
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def sqlite_saver(path: Path) -> SqliteSaver:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))      # LangGraph writes checkpoints from several threads; the saver serialises them with its own lock
    saver.setup()                                                                  # raises sqlite3.DatabaseError on a file that is not a database: never silently ignored
    return saver


class JsonExportStore(dict):
    """Idempotent export (one record per event ID) kept in a JSON file."""
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            super().update(json.loads(self.path.read_text(encoding="utf-8")))

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(self), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
