"""Persistence for the workflow without a new dependency (claude-rag.md § 6: a dependency needs the author's approval).

`FileSaver` is LangGraph's `InMemorySaver` that also writes its tables to one local file after every change, so a paused review survives a restart (spec/impl_08.md § 12.3).
The file is written and read by this program only (pickle of plain dicts: do not load a file from another source). `JsonExportStore` keeps the idempotent export of approved reports.
"""
from __future__ import annotations

import json
import os
import pickle
import threading
from collections import defaultdict
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver


class FileSaver(InMemorySaver):
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self._lock = threading.RLock()                                # LangGraph writes checkpoints from several threads
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = pickle.loads(self.path.read_bytes())
        for t, nss in data["storage"].items():
            for ns, d in nss.items():
                self.storage[t][ns] = dict(d)
        for k, v in data["writes"].items():
            self.writes[k] = dict(v)
        self.blobs.update(data["blobs"])

    def _dump(self) -> None:
        with self._lock:
            self._dump_locked()

    def _dump_locked(self) -> None:
        data = {"storage": {t: {ns: dict(d) for ns, d in nss.items()} for t, nss in self.storage.items()},
                "writes": {k: dict(v) for k, v in self.writes.items()}, "blobs": dict(self.blobs)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps(data))
        os.replace(tmp, self.path)                                   # atomic: a crash never leaves half a file

    def put(self, *a, **k):
        with self._lock:                                              # the write and the dump are one step
            r = super().put(*a, **k)
            self._dump()
        return r

    def put_writes(self, *a, **k):
        with self._lock:                                              # the write and the dump are one step
            r = super().put_writes(*a, **k)
            self._dump()
        return r

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._dump()


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
