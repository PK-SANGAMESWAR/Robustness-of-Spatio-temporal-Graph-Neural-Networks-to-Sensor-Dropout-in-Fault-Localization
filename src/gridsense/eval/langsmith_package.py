"""S8, offline part: the LangSmith dataset payload, its redaction check and a dry-run preview. NOTHING is sent. `upload` refuses until the author approves (O8; skill `rag-langsmith-sync`).

Rules (claude-rag.md § 4 rule 10; skill): only `status: reviewed` items; no drafts (events, safety items, near-miss queries stay local); no secrets, absolute paths, environment values or user names; repo-relative references only.
Default payload (a conservative choice taken 2026-11-20 on the author's delegation, reversible): the reviewed retrieval queries as dataset examples with the query, the reference answer, the answerable flag and the golden context
REFERENCES (document id, section path, span hash; no span text). Items whose golden contexts come from the unpublished manuscript (`paper1_rev1170`), from `results_md` or from the draft SOP are EXCLUDED: an upload publishes text outside this machine.
Traces, experiments and online evaluators are not part of this package (they would carry passage text and need the author's decision on what may leave the machine).
`python -m gridsense.eval.langsmith_package preview` | `upload --confirm-hash <sha256 of the preview payload>`
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

from gridsense.config import ROOT, TRACK, load_config

SENSITIVE_DOCS = {"paper1_rev1170", "results_md", "sop_illustrative_gridsense_review"}
OUT = TRACK / "reports" / "s7"
FORBIDDEN = re.compile(r"(/home/|/Users/|[A-Za-z]:\\\\|sk-[A-Za-z0-9_-]{8,}|[\w.+-]+@[\w-]+\.[\w.-]+|\bubuntu\b|api[_-]?key|LANGSMITH|OPENAI)", re.I)
DATASET = "gridsense-retrieval"


def build_payload(rows: list[dict]) -> dict:
    ver = {r["golden_version"] for r in rows if "golden_version" in r}
    keep, excluded = [], {"draft": 0, "sensitive_source": []}
    for r in rows:
        if r["status"] != "reviewed":
            excluded["draft"] += 1
            continue
        docs = {c["document_id"] for c in r["golden_contexts"]}
        if docs & SENSITIVE_DOCS:
            excluded["sensitive_source"].append({"query_id": r["query_id"], "documents": sorted(docs & SENSITIVE_DOCS)})
            continue
        keep.append({"inputs": {"query": r["query"]}, "outputs": {"reference_answer": r.get("reference_answer", ""), "answerable": r["answerable"],
                                                                  "golden_context_refs": [{"document_id": c["document_id"], "section_path": c["section_path"], "span_hash": c["span_hash"], "grade": c["grade"]} for c in r["golden_contexts"]]},
                     "metadata": {"query_id": r["query_id"], "query_type": r["query_type"], "split": r["split"], "golden_version": r.get("golden_version")}})
    return {"dataset": DATASET, "golden_versions": sorted(map(str, ver)), "examples": keep, "excluded": excluded}


def redaction_hits(payload: dict) -> list[str]:
    text = json.dumps(payload["examples"], ensure_ascii=False)
    return sorted({m.group(0) for m in FORBIDDEN.finditer(text)})


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload["examples"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_rows() -> list[dict]:
    return [json.loads(x) for x in (TRACK / "data" / "eval" / "golden_retrieval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def preview() -> dict:
    p = build_payload(load_rows())
    hits = redaction_hits(p)
    h = payload_hash(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "langsmith_payload_preview.json").write_text(json.dumps({"payload_sha256": h, **p, "redaction_hits": hits}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = load_config()["langsmith"]
    md = [f"# LangSmith dry run (S8, offline; nothing was sent)", "",
          f"Written for the author. Dataset `{DATASET}`: **{len(p['examples'])} examples** would be uploaded (reviewed retrieval queries only). Payload sha256 `{h}`.", "",
          "| Group | Count | Why |", "|---|---|---|",
          f"| In the payload | {len(p['examples'])} | status reviewed; no golden source from the manuscript, `RESULTS.md` or the draft SOP; no span text, only references |",
          f"| Excluded: draft | {p['excluded']['draft']} | drafts never go up (near-miss queries) |",
          f"| Excluded: sensitive source | {len(p['excluded']['sensitive_source'])} | the golden context comes from `paper1_rev1170` (unpublished manuscript), `results_md` or the draft SOP |",
          "| Not part of this package | – | golden events (60, draft), safety items (25, draft), traces, experiments, online evaluators |", "",
          f"Redaction check on the payload (absolute paths, keys, addresses, user names, env names): **{'no hits' if not hits else 'HITS: ' + ', '.join(hits)}**.", "",
          f"Config: `langsmith.enabled` = {cfg.get('enabled')}, `project` = {cfg.get('project')}, `upload_approved` = {cfg.get('upload_approved')}. Tracing stays off (`LANGSMITH_TRACING=false`).", "",
          "## What the author decides", "1. Whether the payload may leave the machine, and the project name.",
          "2. Whether the excluded items may go up (they quote the unpublished manuscript or internal results).",
          "3. Whether golden events and safety items may go up after the author has reviewed them (they hold event facts and passage text).", "",
          f"To approve: set `langsmith.upload_approved: true`, `enabled: true` and `project` in `configs/eval.yaml`, then run `python -m gridsense.eval.langsmith_package upload --confirm-hash {h}`.",
          "Inspect `reports/s7/langsmith_payload_preview.json` first: it is exactly what would be sent.", "",
          "Excluded sensitive-source items: " + ", ".join(f"{e['query_id']} ({'+'.join(e['documents'])})" for e in p["excluded"]["sensitive_source"])]
    (OUT / "LANGSMITH_DRYRUN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"examples": len(p["examples"]), "excluded": {"draft": p["excluded"]["draft"], "sensitive_source": len(p["excluded"]["sensitive_source"])}, "redaction_hits": hits, "payload_sha256": h}


def upload(confirm_hash: str, client=None) -> dict:
    cfg = load_config()["langsmith"]
    if not (cfg.get("enabled") and cfg.get("upload_approved") and cfg.get("project")):
        raise PermissionError("LangSmith upload refused: configs/eval.yaml needs langsmith.enabled, upload_approved and project (the author's approval, O8)")
    p = build_payload(load_rows())
    if redaction_hits(p):
        raise ValueError(f"redaction hits: {redaction_hits(p)}")
    if confirm_hash != payload_hash(p):
        raise PermissionError("the confirmed hash does not match the current payload: preview it again and confirm the new hash")
    if client is None:
        from langsmith import Client

        from gridsense import settings
        client = Client(api_key=settings.secret("LANGSMITH_API_KEY"), api_url=cfg["endpoint"])
    ds = client.create_dataset(dataset_name=DATASET, description=f"GridSense reviewed retrieval queries, golden version {p['golden_versions']}; simulated data; replay, not a live grid")
    client.create_examples(dataset_id=ds.id, inputs=[e["inputs"] for e in p["examples"]], outputs=[e["outputs"] for e in p["examples"]], metadata=[e["metadata"] for e in p["examples"]])
    return {"dataset": DATASET, "examples": len(p["examples"]), "dataset_id": str(ds.id)}


def main() -> None:
    a = sys.argv[1:]
    if a[:1] == ["preview"]:
        print(json.dumps(preview(), indent=1))
    elif a[:2] == ["upload", "--confirm-hash"] and len(a) == 3:
        try:
            print(json.dumps(upload(a[2]), indent=1))
        except (PermissionError, ValueError) as e:
            raise SystemExit(f"REFUSED: {e}")
    else:
        raise SystemExit("usage: preview | upload --confirm-hash <sha256>")


if __name__ == "__main__":
    main()
