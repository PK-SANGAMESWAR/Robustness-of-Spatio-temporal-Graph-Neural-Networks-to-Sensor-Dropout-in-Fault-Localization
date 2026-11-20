"""Publishing guard for the `gridsense-eval` branch (author asked to push, 2026-11-20; the GitHub repository is PUBLIC).

Why: the branch must never carry text of the UNPUBLISHED revision manuscript (`paper1_rev1170`, kept local on purpose; `corpus/v1/1170_revised.tex` is Git-ignored), nor keys, nor the LangSmith organisation id.
What it does (no network, no API call):
- `sanitize(path, data)`: for JSON, JSONL and CSV files, any record that names a sensitive document loses its text-like fields (`[withheld: unpublished source]`); LangSmith organisation ids in URLs are replaced. The hash of a span is kept.
- `EXCLUDE_PREFIXES`: regenerable files that hold the chunk text of every corpus document (`eval/doc_index/`) are not published.
- `leak_scan(blobs)`: finds (a) any 10-word run (shingle) that also occurs in the local manuscript, (b) key-like strings. It prints file names and counts only, never text.
- `rewrite(...)`: replays the unpushed commits through `sanitize`, keeping messages, authors and dates; the old tip stays as a LOCAL backup tag that must never be pushed.
- `commit(...)`: the commit procedure for every later round (replaces `git add`): sanitizes, scans, and refuses to commit a leak.
The working tree is never changed: local files keep their full text.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from gridsense.config import ROOT, TRACK

SENSITIVE = {"paper1_rev1170"}
WITHHELD = "[withheld: unpublished source]"
TEXT_FIELDS = {"text", "span", "passage_text", "excerpt", "section", "section_path", "chunk_id", "claim", "content"}
EXCLUDE_PREFIXES = ("rag-for-dropout/eval/doc_index/", "rag-for-dropout/eval/agent/", "rag-for-dropout/eval/llm_cache/",
                    "rag-for-dropout/reports/findings/paper1_open_switch.md")        # the finding quotes revision-only manuscript text: it stays local
ORG_URL = re.compile(r"smith\.langchain\.com/o/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
KEY_LIKE = re.compile(r"(sk-(?!test-0+\b)[A-Za-z0-9_-]{20,}|lsv2_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,})")      # `sk-test-000...` is the synthetic key of the safety tests
MANUSCRIPT = TRACK / "corpus" / "v1" / "1170_revised.tex"
SHINGLE = 10
SCAN_SUFFIXES = (".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".py", ".tex", ".sh")


def _scrub(x):
    if isinstance(x, dict):
        if any(isinstance(c, dict) and c.get("document_id") in SENSITIVE for c in (x.get("golden_contexts") or [])):
            x = {**x, "reference_answer": WITHHELD}                                  # the answer paraphrases the cited manuscript sentence
        if x.get("document_id") in SENSITIVE:
            return {k: (WITHHELD if k in TEXT_FIELDS else _scrub(v)) for k, v in x.items()}
        return {k: _scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


def sanitize(path: str, data: bytes) -> bytes:
    if path.startswith(EXCLUDE_PREFIXES):
        raise ValueError(f"{path} is excluded from publication")
    text = data.decode("utf-8", errors="strict") if path.endswith((".json", ".jsonl", ".csv", ".md", ".yaml", ".yml", ".txt")) else None
    if text is None:
        return data
    out = text
    if path.endswith(".jsonl"):
        lines = []
        for line in text.splitlines():
            lines.append(json.dumps(_scrub(json.loads(line)), ensure_ascii=False) if line.strip() else line)
        out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    elif path.endswith(".json"):
        obj = json.loads(text)
        scrubbed = _scrub(obj)
        if scrubbed != obj:                                                          # keep the original bytes when nothing changes
            out = json.dumps(scrubbed, indent=1, ensure_ascii=False) + ("\n" if text.endswith("\n") else "")
    elif path.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))
        if rows:
            head = rows[0]
            doc_i = head.index("document_id") if "document_id" in head else None
            changed = False
            for r in rows[1:]:
                if doc_i is not None and len(r) > doc_i and r[doc_i] in SENSITIVE:
                    for j, h in enumerate(head):
                        if h in TEXT_FIELDS and j < len(r) and r[j]:
                            r[j], changed = WITHHELD, True
            if changed:
                buf = io.StringIO()
                csv.writer(buf, lineterminator="\n").writerows(rows)
                out = buf.getvalue()
    out = ORG_URL.sub("smith.langchain.com/o/<org-id-withheld>", out)
    return out.encode("utf-8")


# ---- leak scan ---------------------------------------------------------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


PUBLIC_REFS = ("paper/overleaf.tex", "results/RESULTS.md", "README.md")      # text that is already public on origin/main (the submitted paper): not a leak


def manuscript_shingles(subtract_public: bool = True) -> set[int]:
    """10-word runs of the local revision manuscript. With `subtract_public`, runs that also occur in the public submitted paper or `RESULTS.md` (on origin/main) are removed: only REVISION-ONLY text counts."""
    if not MANUSCRIPT.exists():
        raise FileNotFoundError(f"{MANUSCRIPT} not found: the leak scan needs the local manuscript (it is never committed)")
    t = _tokens(MANUSCRIPT.read_text(encoding="utf-8", errors="ignore"))
    sh = {hash(" ".join(t[i:i + SHINGLE])) for i in range(len(t) - SHINGLE + 1)}
    if subtract_public:
        for ref in PUBLIC_REFS:
            try:
                pub = _tokens(_git("show", f"origin/main:{ref}").decode("utf-8", errors="ignore"))
            except RuntimeError:
                continue
            sh -= {hash(" ".join(pub[i:i + SHINGLE])) for i in range(len(pub) - SHINGLE + 1)}
    return sh


def leak_scan_text(path: str, data: bytes, shingles: set[int]) -> dict:
    """{path, shingle_hits, key_hits}. A hit counts 10-word runs that also occur in the manuscript; boilerplate shared by every paper is not removed here: the author reads the list."""
    if not path.endswith(SCAN_SUFFIXES):
        return {"path": path, "shingle_hits": 0, "key_hits": 0}
    text = data.decode("utf-8", errors="ignore")
    t = _tokens(text)
    hits = sum(1 for i in range(len(t) - SHINGLE + 1) if hash(" ".join(t[i:i + SHINGLE])) in shingles)
    return {"path": path, "shingle_hits": hits, "key_hits": len(KEY_LIKE.findall(text))}


def _git(*args: str, env: dict | None = None, input: bytes | None = None) -> bytes:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, env={**os.environ, **(env or {})}, input=input)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode()[:300]}")
    return r.stdout


def scan_range(rev_range: str, shingles: set[int], sanitized: bool = False) -> list[dict]:
    """Every blob that a range of commits introduces. With `sanitized`, each blob is scanned in the form `sanitize` would publish it (excluded paths are skipped)."""
    objs = _git("rev-list", "--objects", rev_range).decode().splitlines()
    out = []
    for line in objs:
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        sha, path = parts
        if _git("cat-file", "-t", sha).strip() != b"blob":
            continue
        data = _git("cat-file", "blob", sha)
        if sanitized:
            if path.startswith(EXCLUDE_PREFIXES):
                continue
            data = sanitize(path, data)
        r = leak_scan_text(path, data, shingles)
        if r["shingle_hits"] or r["key_hits"]:
            out.append(r)
    return out


# ---- history rewrite and commit ---------------------------------------------------------------------------------------------------------
def _candidates(tree: str) -> list[tuple[str, str]]:
    """(path, blob sha) of every blob in a tree that sanitize may change or must exclude."""
    out = []
    for line in _git("ls-tree", "-r", tree).decode().splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob" and (path.startswith(EXCLUDE_PREFIXES) or path.endswith((".json", ".jsonl", ".csv", ".md", ".yaml", ".yml", ".txt"))) and path.startswith(("rag-for-dropout/", "src/gridsense", "tests/gridsense")):
            out.append((path, sha))
    return out


def sanitized_tree(tree: str, cache: dict) -> str:
    idx = tempfile.mktemp()
    env = {"GIT_INDEX_FILE": idx}
    try:
        _git("read-tree", tree, env=env)
        for path, sha in _candidates(tree):
            if path.startswith(EXCLUDE_PREFIXES):
                _git("update-index", "--force-remove", "--", path, env=env)
                continue
            if sha not in cache:
                new = sanitize(path, _git("cat-file", "blob", sha))
                cache[sha] = sha if new == _git("cat-file", "blob", sha) else _git("hash-object", "-w", "--stdin", input=new).decode().strip()
            if cache[sha] != sha:
                _git("update-index", "--cacheinfo", f"100644,{cache[sha]},{path}", env=env)
        return _git("write-tree", env=env).decode().strip()
    finally:
        if os.path.exists(idx):
            os.remove(idx)


def rewrite(base: str, tip: str, backup_tag: str) -> str:
    """Replay base..tip through `sanitized_tree`. Returns the new tip. The old tip is kept as a local tag (never push it)."""
    commits = _git("rev-list", "--reverse", f"{base}..{tip}").decode().split()
    _git("tag", backup_tag, tip)
    parent, cache = base, {}
    for c in commits:
        fmt = _git("log", "-1", "--format=%an%n%ae%n%aI%n%cn%n%ce%n%cI", c).decode().splitlines()
        env = {"GIT_AUTHOR_NAME": fmt[0], "GIT_AUTHOR_EMAIL": fmt[1], "GIT_AUTHOR_DATE": fmt[2], "GIT_COMMITTER_NAME": fmt[3], "GIT_COMMITTER_EMAIL": fmt[4], "GIT_COMMITTER_DATE": fmt[5]}
        msg = _git("log", "-1", "--format=%B", c)
        parent = _git("commit-tree", sanitized_tree(f"{c}^{{tree}}", cache), "-p", parent, env=env, input=msg).decode().strip()
    return parent


def commit(branch: str, message_file: str, paths: list[str]) -> str:
    """The round commit with the guard: sanitized blobs only, scan, refuse a leak. `paths` are working-tree paths (files or folders) relative to the repository root."""
    shingles = manuscript_shingles()
    idx = tempfile.mktemp()
    env = {"GIT_INDEX_FILE": idx}
    try:
        _git("read-tree", branch, env=env)
        files = set()
        for p in paths:
            files |= {f for f in _git("ls-files", "-co", "--exclude-standard", "--", p).decode().split("\n") if f}
        tracked = {f for f in _git("ls-tree", "-r", "--name-only", branch).decode().split("\n") if f and any(f == p or f.startswith(p.rstrip("/") + "/") for p in paths)}
        for f in tracked - files:
            _git("update-index", "--force-remove", "--", f, env=env)
        findings = []
        for f in sorted(files):
            if f.startswith(EXCLUDE_PREFIXES) or not (ROOT / f).is_file():
                continue
            data = sanitize(f, (ROOT / f).read_bytes())
            r = leak_scan_text(f, data, shingles)
            if r["shingle_hits"] or r["key_hits"]:
                findings.append(r)
            _git("update-index", "--add", "--cacheinfo", f"100644,{_git('hash-object', '-w', '--stdin', input=data).decode().strip()},{f}", env=env)
        if findings:
            raise RuntimeError("leak scan found manuscript text or keys (names and counts only): " + json.dumps(findings)[:1500])
        tree = _git("write-tree", env=env).decode().strip()
        new = _git("commit-tree", tree, "-p", branch, "-F", message_file).decode().strip()
        _git("update-ref", f"refs/heads/{branch}", new)
        return new
    finally:
        if os.path.exists(idx):
            os.remove(idx)
