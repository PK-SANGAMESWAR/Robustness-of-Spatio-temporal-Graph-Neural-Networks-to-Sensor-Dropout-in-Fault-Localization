"""Run one event through the workflow, in separate steps (a human decision happens later).

  python -m gridsense.agent.cli start <event_id> [--live]
  python -m gridsense.agent.cli show <event_id>
  python -m gridsense.agent.cli decide <event_id> --decision approve|reject|clarify --reviewer NAME [--note TEXT]

`start` stops at the human-approval pause and prints the draft report. `decide` is the explicit human action: only a person runs it. The state is kept in
`eval/agent/` (Git-ignored). Default generator: the mock (no API call). `--live` needs the S4 approval file and spends about 0.0006 USD per report.
No authentication: the reviewer name is a plain field (a prototype; see EXPERIMENT_LOG_2 E31).
"""
from __future__ import annotations

import argparse
import json

from gridsense.agent import workflow as W
from gridsense.agent.store import FileSaver, JsonExportStore
from gridsense.config import TRACK
from gridsense.s4 import generate as GEN
from gridsense.s4.llm import ApprovalMissing

DIR = TRACK / "eval" / "agent"


def _app(live: bool):
    from gridsense.s4 import run as R
    from gridsense.s4.judge import MockJudgeResponder
    from gridsense.s4.llm import make_clients
    ev = R.load_events()
    order = sorted(ev)
    gen, _ = make_clients(live, "agent_live" if live else "agent_dry", mock_gen=R.mock_generator, mock_judge=MockJudgeResponder())
    retr, tau, rinfo = R.retriever_and_tau()
    store = JsonExportStore(DIR / "exports.json")
    deps = {"event_lookup": lambda e: ev[e], "retrieve": lambda item: R.retrieve_all(retr, item, "single"), "tau": tau, "generator": gen, "index_version": rinfo["index_fingerprint"],
            "export_store": store, "anon_index": lambda e: order.index(e), "checkpointer": FileSaver(DIR / "checkpoints.pkl")}
    return W.build(deps), store


def _show(app, cfg) -> dict:
    st = app.get_state(cfg).values
    rep = st.get("draft_report") or {}
    print(json.dumps({"event_id": st.get("event_id"), "approval_status": st.get("approval_status"), "paused_for_human": W.paused(app, cfg), "report_mode": st.get("report_mode"),
                      "manual_review_reason": st.get("manual_review_reason"), "gate": st.get("verification_result"), "report_is": "draft (not approved)" if st.get("approval_status") == "pending" else st.get("approval_status")}, indent=1, default=str))
    if rep:
        print("\n" + GEN.report_text(rep))
    print("\naudit:", json.dumps(st.get("audit"), default=str))
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "show", "decide"])
    ap.add_argument("event_id")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--decision", choices=["approve", "reject", "clarify"])
    ap.add_argument("--reviewer")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    try:
        app, _ = _app(a.live)
    except ApprovalMissing as e:
        raise SystemExit(f"REFUSED: {e}")
    cfg = {"configurable": {"thread_id": f"event-{a.event_id}"}}
    if a.cmd == "start":
        if app.get_state(cfg).values:
            raise SystemExit(f"event {a.event_id} already has a saved state; use `show` or `decide`")
        W.start(app, a.event_id)
    elif a.cmd == "decide":
        if not W.paused(app, cfg):
            raise SystemExit("this event is not waiting for a human decision")
        if not (a.decision and a.reviewer):
            raise SystemExit("decide needs --decision and --reviewer")
        W.decide(app, cfg, {"decision": a.decision, "reviewer": a.reviewer, "note": a.note})
    _show(app, cfg)


if __name__ == "__main__":
    main()
