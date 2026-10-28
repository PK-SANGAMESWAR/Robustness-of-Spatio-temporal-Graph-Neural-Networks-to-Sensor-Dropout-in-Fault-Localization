"""The only module that reads environment variables and rag-for-dropout/.env (claude-rag.md § 4 rule 10).

Import this module before deepeval, chromadb or langsmith: it switches their telemetry off and keeps LangSmith
tracing off until configs/eval.yaml langsmith.enabled is true (O8). Never log or print a key value.
"""
from __future__ import annotations

import os

from gridsense.config import TRACK

_TELEMETRY_OFF = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",
    "ANONYMIZED_TELEMETRY": "False",  # chromadb
}


def _load_env() -> None:
    env_file = TRACK / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv  # rag group

        load_dotenv(env_file, override=False)
    for name, value in _TELEMETRY_OFF.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("LANGSMITH_TRACING", "false")


_load_env()


def secret(name: str) -> str:
    """Return a key from the environment; raise without printing anything about its value."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set (rag-for-dropout/.env)")
    return value


def has_secret(name: str) -> bool:
    return bool(os.environ.get(name))
