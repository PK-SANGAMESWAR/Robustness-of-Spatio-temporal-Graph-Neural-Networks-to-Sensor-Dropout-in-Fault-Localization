"""S8: traces of the LangGraph workflow in LangSmith, from a REPLAY of the 30 S4 events (skill `rag-langsmith-sync`, action `tracing`). No API call to OpenAI, no cost: the generator is a replay client that returns the stored raw
generator report of the S4 live run (model `gpt-6-luna`); retrieval, tau, the gate fields, the checks and the assembly run for real, locally.
Every trace stops at the human-approval pause (state: pending). NO human decision is made or simulated here.
What leaves the machine, after `scrub`: node inputs and outputs of the workflow (event facts without truth, the report, citation checks, retrieved passages of the generated research notes and the feeder documents).
Passages, excerpts and ids of the SENSITIVE sources (unpublished manuscript, `RESULTS.md`, draft SOP) are withheld, and any string that names them is replaced. The data are simulated; this is a replay stream, not a live grid (every trace says so).
`python -m gridsense.eval.langsmith_replay`
"""
from __future__ import annotations

import json
import re

from gridsense import settings
from gridsense.config import ROOT, load_config
from gridsense.eval.langsmith_package import SENSITIVE_DOCS
from gridsense.eval.snapshot import ITEMS_DIR, write_snapshot

WITHHELD = "[withheld: sensitive source]"
TOKENS = re.compile("|".join(sorted(SENSITIVE_DOCS)) + r"|1170_revised|/home/\w+", re.I)
PROJECT_TAGS = ["replay", "s4-events", "draft-events", "simulated-data"]


def scrub(x):
    """Recursively withhold sensitive sources. A dict that names a sensitive document loses its text-like fields; any string that contains a sensitive token is replaced."""
    if isinstance(x, dict):
        if x.get("document_id") in SENSITIVE_DOCS:
            return {k: (v if k in ("score", "grade") else WITHHELD) for k, v in x.items()}
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scrub(v) for v in x]
    if isinstance(x, str) and TOKENS.search(x):
        return WITHHELD
    return x


class ReplayGenerator:
    """Returns the stored raw report of the S4 live run for the event under replay. Never calls an API; cost 0."""
    def __init__(self, raw_by_anon: dict[str, dict]):
        self.raw = raw_by_anon

    def complete(self, messages, **kw):
        from gridsense.s4 import llm
        user = messages[-1]["content"]
        m = re.search(r'"event_id":\s*"(event-\d+)"', user)
        return llm.Reply(json.dumps(self.raw[m.group(1)]), 0, 0)


def run(limit: int | None = None, project: str | None = None) -> dict:
    cfg = load_config()["langsmith"]
    if not (cfg.get("enabled") and cfg.get("upload_approved") and cfg.get("project")):
        raise PermissionError("LangSmith is not approved in configs/eval.yaml (O8)")
    import os

    os.environ.pop("LANGSMITH_TRACING", None)            # LangSmith copies this env value into the run metadata; tracing is switched on by `tracing_context` below
    from langsmith import Client, tracing_context
    from langgraph.checkpoint.memory import InMemorySaver

    from gridsense.agent import workflow as W
    from gridsense.s4 import run as R
    ev = R.load_events()
    order = sorted(ev)
    src = sorted(ITEMS_DIR.glob("*_s4_main_live_rescored.jsonl"))[-1]
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    raw = {r["anon_id"]: r.get("report_raw") or r["report"] for r in rows if "report" in r}
    by_anon = {r["anon_id"]: r["event_id"] for r in rows if "report" in r}
    retr, tau, rinfo = R.retriever_and_tau()
    client = Client(api_key=settings.secret("LANGSMITH_API_KEY"), api_url=cfg["endpoint"], hide_inputs=scrub, hide_outputs=scrub)
    deps = {"event_lookup": lambda e: ev[e], "retrieve": lambda item: R.retrieve_all(retr, item, "single"), "tau": tau, "generator": ReplayGenerator(raw), "index_version": rinfo["index_fingerprint"],
            "export_store": {}, "anon_index": lambda e: order.index(e), "checkpointer": InMemorySaver()}
    app = W.build(deps)
    ids = list(by_anon.values())[:limit] if limit else list(by_anon.values())
    done = []
    for i, eid in enumerate(ids):
        with tracing_context(enabled=True, client=client, project_name=project or cfg["project"], tags=PROJECT_TAGS,
                             metadata={"replay": True, "note": "replay of the stored S4 live generator output; no human decision; simulated data; draft events", "policy_version": W.POLICY_VERSION,
                                       "gate_sha256": W.GATE_SHA[:16], "index_version": rinfo["index_fingerprint"][:16], "generator_model_of_replay": cfg and load_config()["generation"]["model"]}):
            cfg_run = {"configurable": {"thread_id": f"replay-{i}"}, "run_name": f"gridsense-workflow-{ev[eid]['stratum']}-k{ev[eid]['k']}", "tags": PROJECT_TAGS}
            app.invoke({"event_id": eid, "trace_id": f"replay-{i}", "errors": [], "audit": []}, cfg_run)
        st = app.get_state({"configurable": {"thread_id": f"replay-{i}"}}).values
        done.append({"event": order.index(eid), "stratum": ev[eid]["stratum"], "k": ev[eid]["k"], "approval_status": st.get("approval_status"), "paused_for_human": bool(app.get_state({"configurable": {"thread_id": f"replay-{i}"}}).next),
                     "manual_review_reason": st.get("manual_review_reason")})
    client.flush()
    snap = write_snapshot("s8_langsmith_replay_traces", {"replay": {"events": len(done), "paused_for_human": sum(d["paused_for_human"] for d in done), "manual_review": sum(d["approval_status"] == "manual_review" for d in done)}}, [],
                          config_effective={"project": project or cfg["project"], "source_items": src.name, "scrub": "sensitive sources withheld"}, index_fingerprint=rinfo["index_fingerprint"],
                          notes="Replay traces of the workflow in LangSmith. No API call, cost 0. No human decision. Sensitive sources withheld (scrub).")
    return {"snapshot": str(snap.relative_to(ROOT)), "events": len(done), "summary": done}


def main() -> None:
    print(json.dumps(run(), indent=1, default=str))


if __name__ == "__main__":
    main()
