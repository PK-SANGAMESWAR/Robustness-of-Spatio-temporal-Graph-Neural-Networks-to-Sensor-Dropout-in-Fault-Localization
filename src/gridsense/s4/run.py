"""S4 runner. DEFAULT = DRY RUN with mock models: no network call, no cost. A LIVE run needs `--live` AND the author's `s4_approval.json` AND approved rubrics (llm.require_live_approval).

    PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.s4.run pilot            # dry run, 5 events
    PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.s4.run pilot --live     # live (refused until approved)
    ... main | noise [--repeats 3] | evidence-check

Events: the 30 of `reports/events/s4_selection.json`. Pilot: 5 events, one per k, seeded. Noise: 10 events (2 per k, seeded) x repeats, cache salt per repeat.
Retrieval: the frozen configuration v2, top 3, one query per event (`generate.retrieval_query`); tau from the dev split (snapshot of the frozen v2 run). Failures are recorded with a cause.
Events whose status is `draft` are refused for a live run unless the approval file says `draft_events_allowed: true`; the snapshot then states `events_status: draft`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from collections import Counter
from dataclasses import replace

import numpy as np

from gridsense import settings  # noqa: F401
from gridsense.config import ROOT, TRACK, load_config
from gridsense.eval import golden as G
from gridsense.eval.snapshot import write_snapshot
from gridsense.s4 import checks as CK
from gridsense.s4 import generate as GEN
from gridsense.s4 import report as RP
from gridsense.s4.decision import assemble, final_evidence_status, tau_flag
from gridsense.s4.judge import MockJudgeResponder, run_judge
from gridsense.s4.llm import ApprovalMissing, MockClient, load_approval, make_clients

EVENTS = TRACK / "data" / "eval" / "golden_events.jsonl"
SELECTION = TRACK / "reports" / "events" / "s4_selection.json"
OUT = TRACK / "reports" / "s4"
PILOT_SEED, NOISE_SEED = 360777, 360888
# Shown to the judge as part of the facts. The first live pilot (2026-11-11) scored geval_adherence 1 for all 5 reports because the judge read these two system fields as inventions (they were not in its context). Harness artefact, fixed.
SYSTEM_FIELDS = ("SYSTEM-SET FIELDS (not chosen by the report writer)\nprediction_status: candidate_not_confirmed (the schema fixes it)\napproval_status: pending_human_review (the system sets it; the writer cannot approve)\nhuman_review_required: a fixed sentence the system adds to every report; when the evidence is insufficient the system also empties recommended_operator_review")


def load_events() -> dict[str, dict]:
    return {r["event_id"]: r for r in map(json.loads, EVENTS.read_text(encoding="utf-8").splitlines())}


def select(kind: str) -> list[str]:
    """Event IDs for a run kind. `main` = the 30 approved events. `pilot` = 1 per k (5), `noise` = 2 per k (10), both seeded among the 30; no hand selection."""
    sel = json.loads(SELECTION.read_text())
    ids = sel["selected"]
    if kind == "main":
        return ids
    ev = load_events()
    per = 1 if kind == "pilot" else 2
    rng = np.random.default_rng(PILOT_SEED if kind == "pilot" else NOISE_SEED)
    out = []
    for k in (0, 1, 2, 5, 10):
        pool = sorted(e for e in ids if ev[e]["k"] == k)
        out += [pool[j] for j in sorted(rng.choice(len(pool), size=per, replace=False).tolist())]
    return out


def sop_decision() -> str:
    """The author's choice for the illustrative SOP (O42): 'keep' or 'exclude'. Read from the approval file; a dry run without the file keeps it."""
    try:
        return load_approval()["sop_decision"]
    except ApprovalMissing:
        return "keep"


def apply_sop_decision(retr, decision: str) -> None:
    """`exclude`: the SOP (approval_status `illustrative`) is not retrieved. Only the query-time filter changes; the index and its fingerprint stay as frozen. The dense top-1 used for tau is then taken over the remaining documents."""
    if decision == "exclude":
        retr.cfg = replace(retr.cfg, filters={**(retr.cfg.filters or {}), "approval_status": ["curated_for_research"]})


def retriever_and_tau():
    from gridsense.eval.layer1 import embedder_from_config, get_index, retrieval_config
    from gridsense.eval.sweep import latest_snapshot, load_winners
    from gridsense.ingest.corpus import load_corpus
    from gridsense.retrieval.retriever import Retriever
    win, docs = load_winners(), load_corpus()
    st = win["state"]
    assert st["retrieval"]["table_rows"] is True and st["retrieval"]["top_k"] == 3, "the frozen configuration v2 is expected"
    rc, emb, minilm = replace(retrieval_config({}), **st["retrieval"]), embedder_from_config(st["embedder"]), embedder_from_config("minilm")
    ix = get_index(rc, emb, docs, chunk_embedder=minilm, rebuild=False)
    snap = latest_snapshot("frozen_final_v2_test")
    tau = snap["metrics"].get("sufficiency.tau_dev.tau")
    retr = Retriever(ix)
    sop = sop_decision()
    apply_sop_decision(retr, sop)
    return retr, tau, {"sop_decision": sop, "index_fingerprint": ix.fingerprint, "tau": tau, "tau_source": "dev split, snapshot " + snap["label"], "retrieval": st["retrieval"]}


# ---- mock generator (dry run): a compliant report built from the prompt itself -----------------------------------------------------
def mock_generator(messages: list[dict], purpose: str) -> str:
    import re
    user = messages[-1]["content"]
    if purpose == "evidence_check":
        return json.dumps({"evidence_status": "sufficient", "reason": "mock"})
    ev = json.loads(re.search(r"EVENT\n(\{.*?\n\})\n\nCASE NEIGHBOURS", user, flags=re.S).group(1))
    pas = re.findall(r"\[([\w\-]+) › ([^\]]*)\]\n(.+?)(?=\n\n\[|\n\nSCHEMA|\Z)", user.split("PASSAGES", 1)[1], flags=re.S)
    cnt = Counter(int(m.group(1)) for m in re.finditer(r"^c\d+ \| (\d+) \|", user, flags=re.M))
    top = cnt.most_common(1)[0] if cnt else (None, 0)
    ex = []
    for d, s, t in pas[:2]:
        sent = re.split(r"(?<=[.!?])\s+", t.strip().split("\n")[-1])[0][:160]
        ex.append({"claim": "The passage explains this gate outcome.", "document_id": d, "section": s, "excerpt": sent})
    cov = (f"{ev['k']} of 25 μPMUs are dark: buses {', '.join(map(str, ev['dark_uPMU_buses']))}." if ev["k"] else "No μPMU is dark (k = 0).")
    return json.dumps({"event_id": ev["event_id"], "predicted_location": ev["predicted_bus"], "prediction_status": "candidate_not_confirmed", "gate_outcome": ev["gate"]["outcome"],
                       "gate_reason_codes": ev["gate"]["reason_codes"], "candidate_set": ev["candidate_set"] or [], "sensor_coverage_summary": cov,
                       "case_evidence_summary": f"{top[1]} of 10 similar cases have fault bus {top[0]}. Neighbours are precedents, not proof.", "document_evidence": ex,
                       "limitations": ["The data are simulated."], "recommended_operator_review": ["Review the cited sections before any action."],
                       "evidence_status": "sufficient", "evidence_status_reason": "mock"})


def retrieve_all(retr, item: dict, mode: str) -> dict:
    """Passages of all queries, de-duplicated by chunk in order; the tau flag uses the LOWEST top-1 similarity over the queries (fail closed)."""
    seen, passages, tops = set(), [], []
    for q, k in GEN.retrieval_queries(item, mode):
        out = retr.retrieve(q, top_k=k)
        tops.append(out["dense_top1"])
        for p in out["passages"]:
            if p["chunk_id"] not in seen:
                seen.add(p["chunk_id"])
                passages.append(p)
    return {"passages": passages, "dense_top1": min(tops)}


def process(item: dict, index: int, retr, tau, gen, judge, salt: str, do_judge: bool = True, queries: str = "single") -> dict:
    view = GEN.event_view(item, index)
    q = GEN.retrieval_query(item)
    rec = {"event_id": item["event_id"], "anon_id": view["event_id"], "k": item["k"], "stratum": item["stratum"], "prediction_correct": item["prediction_correct"], "query": q, "salt": salt}
    try:
        r = retrieve_all(retr, item, queries)
        passages = r["passages"]
        rec |= {"passages": [{"chunk_id": p["chunk_id"], "document_id": p["document_id"], "section_path": p["section_path"], "score": p["score"]} for p in passages], "dense_top1": r["dense_top1"]}
        msgs = GEN.build_messages(item, index, passages)
        reply = gen.complete(msgs, purpose="generator", json_mode=True, max_tokens=4000, salt=salt)
        rec["generation"] = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "usd": reply.usd, "cached": reply.cached, "temperature_sent": reply.temperature_sent}
        rep, err = GEN.parse_report(reply.text)
        if err:
            rec["error"] = err
            rec["raw_reply_head"] = reply.text[:200]
            return rec
        flag = tau_flag(r["dense_top1"], tau)
        rec["evidence"] = final_evidence_status(flag, rep.get("evidence_status"))
        rec["report_raw"] = rep                                  # exactly what the generator wrote
        rep, info = assemble(rep, rec["evidence"]["final"])     # E + A (author 2026-11-18)
        rec["evidence"] |= info
        rec["report"] = rep
        rec["checks"] = CK.run_checks(rep, view, item["event_object"]["knn"]["neighbours"], passages, rec["evidence"]["final"])
        if do_judge:
            ctx = ("EVENT FACTS\n" + json.dumps(view, indent=1) + "\n\nCASE NEIGHBOURS\n" + GEN.neighbours_table(item) + "\n\n" + SYSTEM_FIELDS)
            question = f"Write a decision-support report for a fault-location prediction. {q}"
            rec["judge"] = run_judge(judge, GEN.report_text(rep), question, ctx, [GEN.format_passages([p]) for p in passages], salt=salt)
    except ApprovalMissing:
        raise
    except Exception as e:                                       # a failure with a cause, never a score
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec


def run(kind: str, live: bool, repeats: int = 1, queries: str = "single") -> dict:
    cfg = load_config()
    ev = load_events()
    ids = select(kind)
    status = Counter(ev[i]["status"] for i in ids)
    events_status = "reviewed" if set(status) == {"reviewed"} else "draft"
    if live:
        a = load_approval()
        if events_status == "draft" and not a.get("draft_events_allowed"):
            raise ApprovalMissing("the S4 events are still status draft; the approval file must say draft_events_allowed: true (the results then carry events_status: draft)")
    label = f"s4_{kind}" + ("_live" if live else "_dry")
    gen, judge = make_clients(live, label, mock_gen=mock_generator, mock_judge=MockJudgeResponder())
    retr, tau, rinfo = retriever_and_tau()
    order = sorted(ev)                                           # a stable index per event: the anonymous id does not depend on the selection
    items, t0 = [], time.time()
    for rep in range(repeats):
        for eid in ids:
            items.append(process(ev[eid], order.index(eid), retr, tau, gen, judge, salt="main" if kind != "noise" else f"noise{rep}", queries=queries) | {"repeat": rep})
    metrics = {"per_stratum": RP.per_stratum(items), "weighted_over_the_covered_strata": RP.weighted_over_covered(items), "informational_unweighted": RP.informational_unweighted(items)}
    if kind == "noise":
        metrics["noise"] = noise_summary(items)
    usd = sum((i.get("generation", {}).get("usd", 0.0) + (i.get("judge") or {}).get("usage", {}).get("usd", 0.0)) for i in items)
    phash = {n: hashlib.sha256((TRACK / "prompts" / f"{n}.md").read_bytes()).hexdigest()[:16] for n in ("generator_v0", "evidence_check_v0")}
    rhash = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((TRACK / "rubrics").glob("*.md"))}
    snap = write_snapshot(label, metrics, items, config_effective={"kind": kind, "live": live, "events": ids, "events_status": events_status, "retrieval": rinfo,
                                                                  "generation_temperature_asked": cfg["generation"].get("temperature"), "judge_temperature_asked": cfg["judge"].get("temperature"), "reasoning_effort": {"generator": cfg["generation"].get("reasoning_effort"), "judge": cfg["judge"].get("reasoning_effort")},
                                                                  "temperature_notes": {"generator": getattr(gen, "default_temperature_note", None), "judge": getattr(judge, "default_temperature_note", None)}, "repeats": repeats, "queries": queries},
                          prompt_hashes=phash, rubric_hashes=rhash, golden_versions={"events": cfg["golden_sets"]["events"]["version"]}, judge_model=cfg["judge"]["model"] + ("" if live else " (mock)"),
                          generation_model=cfg["generation"]["model"] + ("" if live else " (mock)"), index_fingerprint=rinfo["index_fingerprint"],
                          notes=f"S4 {kind} {'LIVE' if live else 'DRY RUN (mock models, no API call)'}; events_status={events_status}; {time.time() - t0:.0f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    md = RP.markdown(items, f"S4 {kind} ({'live' if live else 'dry run with mock models: the numbers below test the plumbing and mean nothing'}); events_status: {events_status}")
    (OUT / f"S4_{kind}_{'live' if live else 'dry'}.md").write_text(md, encoding="utf-8")
    fails = [i for i in items if i.get("error")]
    return {"snapshot": str(snap.relative_to(ROOT)), "items": len(items), "failures": len(fails), "causes": sorted({i["error"] for i in fails})[:5], "usd": usd, "events_status": events_status}


def rescore(items_path: str, kind: str) -> dict:
    """Offline, no API call. Re-assembles stored reports (E + A, 2026-11-18) and recomputes the two checks that changed: `field_consistency` (checker fix of 2026-11-11)
    and `evidence_consistency`. All other checks read text the assembly cannot add to. Judge scores are NOT recomputed: they were given to the raw report."""
    ev = load_events()
    order = sorted(ev)
    items = [json.loads(line) for line in Path(items_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for it in items:
        if it.get("error") or "report" not in it:
            continue
        view = GEN.event_view(ev[it["event_id"]], order.index(it["event_id"]))
        raw = it.get("report_raw") or it["report"]
        rep, info = assemble(raw, it["evidence"]["final"])
        it["report_raw"], it["report"], it["evidence"] = raw, rep, it["evidence"] | info
        it["checks"]["field_consistency"] = CK.field_consistency(rep, view)
        it["checks"]["evidence_consistency"] = CK.evidence_consistency(rep, it["evidence"]["final"])
    metrics = {"per_stratum": RP.per_stratum(items), "weighted_over_the_covered_strata": RP.weighted_over_covered(items), "informational_unweighted": RP.informational_unweighted(items)}
    if kind == "noise":
        metrics["noise"] = noise_summary(items)
    label = f"s4_{kind}_live_rescored"
    src_path = Path(items_path).parent.parent / "snapshots" / (Path(items_path).stem + ".json")        # the source snapshot: its metadata carries over (comparability, S7)
    src = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    carry = {k: src[k] for k in ("golden_versions", "judge_model", "generation_model", "index_fingerprint", "prompt_hashes", "rubric_hashes") if src.get(k) is not None}
    snap = write_snapshot(label, metrics, items, config_effective=(src.get("config_effective") or {}) | {"source_items": str(items_path), "rescored_checks": ["field_consistency", "evidence_consistency"]}, **carry,
                          notes=f"OFFLINE RESCORE of {items_path} (no API call). E + A applied to the stored reports; judge scores are those of the RAW reports, not re-judged.")
    OUT.mkdir(parents=True, exist_ok=True)
    md = RP.markdown(items, f"S4 {kind} (live, rescored offline with E + A; judge scores are of the raw reports)")
    (OUT / f"S4_{kind}_live_rescored.md").write_text(md, encoding="utf-8")
    return {"snapshot": str(snap.relative_to(ROOT)), "items": len(items)}


def noise_summary(items: list[dict]) -> dict:
    """Standard deviation of each metric across the repeats of the same event, then its mean over events (tolerance for the S7 comparison)."""
    by: dict[str, list[dict]] = {}
    for i in items:
        by.setdefault(i["event_id"], []).append(RP.metric_values(i))
    out = {}
    for name in sorted({k for v in by.values() for r in v for k in r}):
        sds = [np.std([r[name] for r in v if r.get(name) is not None], ddof=1) for v in by.values() if sum(r.get(name) is not None for r in v) >= 2]
        if sds:
            out[name] = {"mean_sd_over_events": float(np.mean(sds)), "events": len(sds)}
    return out


def evidence_check(live: bool) -> dict:
    """The evidence-status check on every golden retrieval query (S4_PLAN.md § 4, item 3): final decision = tau flag OR generator status. Reviewed queries by split; drafts apart."""
    cfg = load_config()
    rows = G.load(ROOT / cfg["golden_sets"]["retrieval"]["path"], statuses=("reviewed", "draft"))
    gen, _ = make_clients(live, "s4_evidence_check", mock_gen=mock_generator, mock_judge=MockJudgeResponder())
    retr, tau, rinfo = retriever_and_tau()
    esys, euser, _ = GEN.split_prompt("evidence_check_v0")
    items = []
    for r in rows:
        rec = {"query_id": r["query_id"], "split": r["split"], "status": r["status"], "answerable": r["answerable"]}
        try:
            out = retr.retrieve(r["query"])
            msgs = [{"role": "system", "content": esys.replace("{{", "{").replace("}}", "}")}, {"role": "user", "content": euser.format(question=r["query"], passages=GEN.format_passages(out["passages"]))}]
            reply = gen.complete(msgs, purpose="evidence_check", json_mode=True, max_tokens=1500)
            try:
                gs = json.loads(reply.text).get("evidence_status")
            except Exception:
                gs = None
            rec |= {"dense_top1": out["dense_top1"], "evidence": final_evidence_status(tau_flag(out["dense_top1"], tau), gs), "usd": reply.usd}
        except ApprovalMissing:
            raise
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        items.append(rec)
    res = {}
    for st in ("reviewed", "draft"):
        for split in ("dev", "test"):
            sel = [i for i in items if i["status"] == st and i["split"] == split and "error" not in i]
            for name, flt in (("unanswerable", lambda i: not i["answerable"]), ("answerable", lambda i: i["answerable"])):
                s = [i for i in sel if flt(i)]
                if s:
                    res[f"{st}.{split}.{name}"] = {"n": len(s), "final_insufficient": sum(i["evidence"]["final"] == "insufficient" for i in s) / len(s),
                                                  "tau_only": sum(i["evidence"]["tau_flag"] for i in s) / len(s)}
    snap = write_snapshot("s4_evidence_check" + ("_live" if live else "_dry"), {"evidence_check": res}, items, config_effective={"live": live, "retrieval": rinfo},
                          notes="evidence-status check on every golden retrieval query; " + ("LIVE" if live else "DRY RUN with mock models (the mock generator always says sufficient): only the tau-only column means something"))
    return {"snapshot": str(snap.relative_to(ROOT)), "result": res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["pilot", "main", "noise", "evidence-check", "rescore"])
    ap.add_argument("--items", help="rescore only: the items .jsonl to re-assemble")
    ap.add_argument("--of", choices=["main", "noise"], default="main", help="rescore only: the kind of the items")
    ap.add_argument("--live", action="store_true", help="real API calls; needs rag-for-dropout/s4_approval.json and approved rubrics")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--queries", choices=["single", "multi"], default="single", help="single = the plan (one query, top 3); multi = gate, dropout and neighbour queries, top 2 each")
    a = ap.parse_args()
    try:
        out = rescore(a.items, a.of) if a.kind == "rescore" else evidence_check(a.live) if a.kind == "evidence-check" else run(a.kind, a.live, a.repeats if a.kind == "noise" else 1, a.queries)
    except ApprovalMissing as e:
        raise SystemExit(f"REFUSED: {e}")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
