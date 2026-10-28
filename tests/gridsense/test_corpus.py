"""Corpus v1 integrity (O11): frozen copies match the manifest; their numbers match the CSV for the checkpoint in use."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "rag-for-dropout" / "corpus" / "v1"
MANIFEST = json.loads((V1 / "MANIFEST.json").read_text(encoding="utf-8"))
CURVES = ROOT / "results" / "dropout_study_aware" / "degradation_curves.csv"


def _require(doc: dict) -> Path:
    path = V1 / doc["file"]
    if not path.is_file():  # 1170_revised.tex is Git-ignored (unpublished manuscript; rag-for-dropout/.gitignore)
        pytest.skip(f"pending: {doc['file']} is kept local, not in Git; copy it to corpus/v1/ (sha256 in MANIFEST.json)")
    return path


@pytest.mark.parametrize("doc", MANIFEST["documents"], ids=lambda d: d["document_id"])
def test_corpus_file_matches_manifest_sha256(doc: dict) -> None:
    data = _require(doc).read_bytes()
    assert hashlib.sha256(data).hexdigest() == doc["sha256"], f"{doc['file']} changed after freezing"
    assert len(data) == doc["bytes"]


def _dropout_aware_rgsage_max_f1() -> dict[int, str]:
    rows = [r for r in csv.DictReader(CURVES.open()) if r["model"] == "rgsage_max"]
    return {int(r["k"]): f"{float(r['f1_mean']):.3f}" for r in rows}


@pytest.mark.parametrize("doc", MANIFEST["documents"], ids=lambda d: d["document_id"])
def test_corpus_numbers_match_checkpoint_curve(doc: dict) -> None:
    """Every dropout-aware RGSAGE-Max macro-F1 (k = 0, 1, 2, 5, 10; 3 decimals) appears in the document."""
    text = _require(doc).read_text(encoding="utf-8")
    f1 = _dropout_aware_rgsage_max_f1()
    assert sorted(f1) == [0, 1, 2, 5, 10]
    missing = {k: v for k, v in f1.items() if v not in text}
    assert not missing, f"{doc['file']} lacks curve values {missing} from {CURVES.relative_to(ROOT)}"
