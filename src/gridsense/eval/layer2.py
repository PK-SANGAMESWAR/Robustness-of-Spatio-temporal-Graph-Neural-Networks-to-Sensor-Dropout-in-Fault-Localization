"""Layer 2, the generator with GOLDEN context (rag-eval.md § 4 Layer 2; EVAL.md § 4). Candidates see the golden passage spans instead of retrieved passages, so the comparison isolates the generator.

Same events (the 30 S4 events, draft), same prompt (`generator_v0`), same assembly (E + A + O46), same deterministic checks as S4. Judge metrics are NOT part of this run (they would cost about 0.013 to 0.027 USD per report); the selection rule needs only the deterministic floors.
Selection rule (EVAL.md § 4, author): **the cheapest candidate that meets every floor, not the highest mean.** Floors = the nine deterministic checks of the report as shipped pass in 100 % of the reports and no report fails to parse.
The raw generator behaviour is reported next to it (not a floor): coverage sentence incomplete, review statement while insufficient, evidence status "sufficient" given golden context, status flips between repeats.
Golden passages are `draft` (agent-built): every result is labelled draft. Spend caps per candidate are fixed here, before any live call.
`python -m gridsense.eval.layer2 [--live]`   (dry run with the mock generator by default)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from gridsense.config import ROOT, load_config
from gridsense.eval.layer6 import timing
from gridsense.eval.snapshot import write_snapshot
from gridsense.s4 import checks as CK
from gridsense.s4 import generate as GEN
from gridsense.s4.decision import assemble, final_evidence_status
from gridsense.s4.llm import ApprovalMissing, CostLimitError, make_clients, make_generator

# candidate -> (events, repeats, spend cap in USD). Fixed before the run. Worst case total 0.95 USD (ledger before: 1.99 of 5 USD).
PLAN = {"gpt-6-luna": (30, 3, 0.15), "gpt-5.4-nano": (30, 3, 0.25), "gpt-5-mini": (30, 3, 0.35), "gpt-6-sol": (10, 1, 0.20)}
CHECKS = ("schema", "field_consistency", "neighbour_counts", "citation_validity", "no_confirmation", "no_control_command", "no_approval_claim", "no_injection_echo", "evidence_consistency")
MAX_CONSECUTIVE_ERRORS = 3


def golden_passages(event: dict) -> list[dict]:
    return [{"chunk_id": f"golden:{i}", "document_id": g["document_id"], "section_path": g["section_path"], "text": g["span"], "score": 1.0} for i, g in enumerate(event["golden_passages"])]


def one(event: dict, index: int, gen, salt: str) -> dict:
    view = GEN.event_view(event, index)
    passages = golden_passages(event)
    rec = {"event_id": event["event_id"], "stratum": event["stratum"], "k": event["k"]}
    reply = gen.complete(GEN.build_messages(event, index, passages), purpose="layer2_generator", json_mode=True, max_tokens=4000, salt=salt)
    rec["generation"] = {"usd": reply.usd, "cached": reply.cached, "latency_s": reply.latency_s, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
    raw, err = GEN.parse_report(reply.text)
    if err:
        return rec | {"error": err}
    final = final_evidence_status(False, raw.get("evidence_status"))                      # golden context: no retrieval score, so no tau flag
    rep, info = assemble(raw, final["final"], view)
    rec |= {"report_raw": raw, "report": rep, "evidence": final | info, "checks": CK.run_checks(rep, view, event["event_object"]["knn"]["neighbours"], passages, final["final"])}
    return rec


def summarize(items: list[dict]) -> dict:
    ok = [i for i in items if "checks" in i]
    n = len(items)
    gate = {c: float(np.mean([bool(i["checks"][c]["pass"]) for i in ok])) if ok else None for c in CHECKS}
    billed = [i["generation"] for i in items if i.get("generation") and not i["generation"].get("cached")]
    by_event: dict[str, list[str]] = defaultdict(list)
    for i in ok:
        by_event[i["event_id"]].append(i["evidence"]["final"])
    multi = [v for v in by_event.values() if len(v) > 1]
    meets = bool(n) and len(ok) == n and all(v == 1.0 for v in gate.values())
    lat = [g["latency_s"] for g in billed if g.get("latency_s")]
    return {"reports": n, "failures": n - len(ok), "meets_floors": meets, "gate_checks": gate,
            "generator_says_sufficient_given_golden_context": float(np.mean([i["evidence"]["generator_status"] == "sufficient" for i in ok])) if ok else None,
            "raw_coverage_incomplete": float(np.mean([i["evidence"]["generator_coverage_incomplete"] for i in ok])) if ok else None,
            "raw_review_while_insufficient": float(np.mean([i["evidence"]["generator_review_while_insufficient"] for i in ok])) if ok else None,
            "events_with_status_flip": {"events": len(multi), "flipping": sum(len(set(v)) > 1 for v in multi)},
            "cost_per_report_usd": float(np.mean([g["usd"] for g in billed])) if billed else None, "billed_calls": len(billed),
            "input_tokens_per_report": float(np.mean([g["input_tokens"] for g in billed])) if billed else None,
            "output_tokens_per_report": float(np.mean([g["output_tokens"] for g in billed])) if billed else None, "generation_latency": timing(lat) if lat else None}


def choose(summaries: dict[str, dict]) -> dict:
    """The cheapest candidate that meets every floor. No candidate meets them: say so; do not pick the best of the failures."""
    eligible = {m: s for m, s in summaries.items() if s["meets_floors"] and s["cost_per_report_usd"] is not None}
    if not eligible:
        return {"chosen": None, "reason": "no candidate meets every floor"}
    m = min(eligible, key=lambda x: eligible[x]["cost_per_report_usd"])
    return {"chosen": m, "reason": "cheapest candidate that meets every floor (EVAL.md § 4)", "cost_per_report_usd": eligible[m]["cost_per_report_usd"], "eligible": sorted(eligible)}


def run(live: bool, plan: dict | None = None) -> dict:
    from gridsense.s4 import run as R
    plan = plan or PLAN
    cfg = load_config()
    ev = R.load_events()
    order = sorted(ev)
    selected = R.select("main")
    if live:
        a = R.load_approval()
        if not a.get("draft_events_allowed"):
            raise ApprovalMissing("golden passages and events are draft: the approval file must allow draft events")
    items_all: list[dict] = []
    per: dict[str, dict] = {}
    notes: dict[str, str] = {}
    for model, (n_events, repeats, cap) in plan.items():
        gen = make_generator(model, f"s5_layer2_{model}", cap) if live else make_clients(False, "s5_layer2_dry", mock_gen=R.mock_generator, mock_judge=lambda m, p: "{}")[0]
        ids = selected[::max(1, len(selected) // n_events)][:n_events]
        items, errors = [], 0
        for rep in range(repeats):
            for eid in ids:
                try:
                    rec = one(ev[eid], order.index(eid), gen, salt=f"l2-{rep}") | {"model": model, "repeat": rep}
                    errors = 0
                except ApprovalMissing:
                    raise
                except CostLimitError as e:
                    notes[model] = f"stopped at the spend cap: {e}"
                    break
                except Exception as e:                                         # a failure with a cause, never a pass
                    rec = {"model": model, "event_id": eid, "repeat": rep, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    errors += 1
                items.append(rec)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    notes[model] = f"aborted after {MAX_CONSECUTIVE_ERRORS} consecutive API errors: {rec['error']}"
                    break
            if model in notes:
                break
        per[model] = summarize(items) | {"plan": {"events": len(ids), "repeats": repeats, "cap_usd": cap}, "note": notes.get(model)}
        items_all += items
    decision = choose(per)
    snap = write_snapshot("s5_layer2" + ("_live" if live else "_dry"), {"layer2": per, "decision": decision}, items_all,
                          config_effective={"live": live, "plan": {m: list(v) for m, v in plan.items()}, "context": "golden passage spans (draft)", "reasoning_effort": cfg["generation"].get("reasoning_effort"), "events": "S4 selection (30), draft"},
                          golden_versions={"events": cfg["golden_sets"]["events"]["version"]}, generation_model="candidates: " + ", ".join(plan) + ("" if live else " (mock)"),
                          notes=f"Layer 2 {'LIVE' if live else 'DRY RUN with the mock generator: plumbing only'}; golden context; deterministic checks only; draft events")
    return {"snapshot": str(snap.relative_to(ROOT)), "decision": decision, "candidates": per}


def rescore(items_path: str) -> dict:
    """Offline, no API call (E39). Recomputes `no_approval_claim`, `no_confirmation` (clause-level negation, "rather than") and `neighbour_counts` (a true store-correctness count is allowed) from the stored reports, then the summaries and the choice.
    Also a like-for-like view: every candidate on the SAME events and the first repeat only (the events of the smallest plan), because the plans differ in size (90 reports against 10).
    The other checks need the passages, which a Layer 2 item does not store; they are unchanged."""
    from pathlib import Path
    from gridsense.s4 import run as R
    ev = R.load_events()
    order = sorted(ev)
    items = [json.loads(x) for x in Path(items_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    for it in items:
        if "checks" in it:
            e = ev[it["event_id"]]
            view = GEN.event_view(e, order.index(it["event_id"]))
            it["checks"]["no_approval_claim"] = CK.no_approval_claim(it["report"])
            it["checks"]["no_confirmation"] = CK.no_confirmation(it["report"])
            it["checks"]["neighbour_counts"] = CK.neighbour_counts(it["report"], e["event_object"]["knn"]["neighbours"], view)
    models = list(dict.fromkeys(i["model"] for i in items))
    per = {m: summarize([i for i in items if i["model"] == m]) for m in models}
    small = min(models, key=lambda m: len([i for i in items if i["model"] == m]))
    ev_small = {i["event_id"] for i in items if i["model"] == small}
    same = {m: summarize([i for i in items if i["model"] == m and i["event_id"] in ev_small and i.get("repeat", 0) == 0]) for m in models}
    snap = write_snapshot("s5_layer2_live_rescored", {"layer2": per, "decision": choose(per), "like_for_like_same_events_first_repeat": same, "like_for_like_decision": choose(same)}, items,
                          config_effective={"source_items": str(items_path), "rescored": ["no_approval_claim", "no_confirmation", "neighbour_counts"], "like_for_like_events": len(ev_small), "like_for_like_reference_candidate": small},
                          notes="OFFLINE RESCORE of the Layer 2 live run (no API call): no_approval_claim, no_confirmation and neighbour_counts fixed (E39). Golden passages and events are draft.")
    return {"snapshot": str(snap.relative_to(ROOT)), "decision": choose(per), "like_for_like_decision": choose(same), "candidates": per, "like_for_like": same}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--rescore", help="items .jsonl of a live run: recompute offline (no API call)")
    a = ap.parse_args()
    try:
        out = rescore(a.rescore) if a.rescore else run(a.live)
    except ApprovalMissing as e:
        raise SystemExit(f"REFUSED: {e}")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
