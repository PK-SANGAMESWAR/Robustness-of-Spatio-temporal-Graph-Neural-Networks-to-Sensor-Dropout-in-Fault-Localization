"""S0 checks for the GridSense track docs. Rules: rag-for-dropout/claude-rag.md; layout: rag-for-dropout/IMPLEMENTATION.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "rag-for-dropout"
FROZEN_SOURCES = {"implementation-plan.md", "rag-eval.md"}
MAX_LINES = 200  # every new md file must be below 200 lines (author, 2026-10-28)
HEADER = re.compile(r"<!-- Source: (?P<src>[\w.-]+) lines (?P<a>\d+)-(?P<b>\d+), verbatim\. Do not edit\. Index: 00_INDEX\.md -->")


def _new_md_files() -> list[Path]:
    files = [p for p in TRACK.rglob("*.md") if p.name not in FROZEN_SOURCES or p.parent != TRACK]
    files += sorted((ROOT / ".claude" / "skills").glob("rag-*/SKILL.md"))
    return sorted(files)


@pytest.mark.parametrize("path", _new_md_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_md_below_200_lines(path: Path) -> None:
    n = len(path.read_text(encoding="utf-8").splitlines())
    assert n < MAX_LINES, f"{path.relative_to(ROOT)} has {n} lines (limit: < {MAX_LINES})"


@pytest.mark.parametrize("source", sorted(FROZEN_SOURCES))
def test_spec_parts_rebuild_source_verbatim(source: str) -> None:
    parts = []
    for part in sorted((TRACK / "spec").glob("*.md")):
        lines = part.read_text(encoding="utf-8").splitlines(keepends=True)
        m = HEADER.fullmatch(lines[0].rstrip("\n")) if lines else None
        if m and m["src"] == source:
            assert lines[1] == "\n", f"{part.name}: line 2 must be blank"
            parts.append((int(m["a"]), int(m["b"]), "".join(lines[2:])))
    assert parts, f"no spec parts for {source}"
    parts.sort()
    expected_start = 1
    for a, b, _ in parts:
        assert a == expected_start, f"{source}: gap or overlap at line {a} (expected {expected_start})"
        expected_start = b + 1
    original = (TRACK / source).read_text(encoding="utf-8")
    assert "".join(body for _, _, body in parts) == original, f"spec parts differ from {source}"


REQUIRED_SECTIONS = ["model", "dropout", "splits", "calibration", "gate", "case_store", "corpus", "chunking",
                     "embedding", "vector_store", "retrieval", "reranker", "generation", "judge", "golden_sets",
                     "repeats", "cost", "langsmith", "snapshots"]


def test_eval_config_has_required_sections() -> None:
    cfg = yaml.safe_load((TRACK / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    assert not missing, f"configs/eval.yaml misses sections: {missing}"
    assert cfg["dropout"]["k_levels"] == [0, 1, 2, 5, 10], "dropout grid must be the paper protocol (P2)"
    assert cfg["splits"]["unit"] == "window" and cfg["splits"]["group_by"] == "run"
    for name, ckpt in cfg["model"]["checkpoints"].items():
        assert (ROOT / ckpt).is_file(), f"checkpoint for seed {name} not found: {ckpt}"


def test_registry_roles_valid() -> None:
    reg = yaml.safe_load((TRACK / "configs" / "registry.yaml").read_text(encoding="utf-8"))
    names = [m["name"] for m in reg["metrics"]]
    assert len(names) == len(set(names)), "duplicate metric names in registry.yaml"
    for m in reg["metrics"]:
        assert m["role"] in {"gate", "guardrail", "informational"}, m["name"]
        assert m["direction"] in {"higher_is_better", "lower_is_better"}, m["name"]
        assert m["tolerance_source"] in {"measured", "provisional"}, m["name"]


@pytest.mark.parametrize("path", sorted((TRACK / "rubrics").glob("*.md")), ids=lambda p: p.name)
def test_rubric_has_status_and_examples(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["status"] in {"draft", "approved"}
    assert front["rubric"] == path.stem
    for heading in ("## Criteria", "## Score scale", "## Good example", "## Bad example"):
        assert heading in text, f"{path.name} misses '{heading}'"
