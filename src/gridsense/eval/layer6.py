"""Layer 6, operations (rag-eval.md § 4 Layer 6; EVAL.md § 9). Free and local: no API call.

Measured here (CPU, 2 threads, `nice -n 19`):
- document retrieval latency, report validation latency, workflow orchestration latency (mock generator, in-memory saver), per event, p50 and p95;
- determinism of document retrieval (same chunk IDs and scores on a rerun);
- peak RAM of this process;
- cost, tokens and failure rate of the GENERATOR from the live S4 items (judge calls are evaluation, not system cost; reported apart).
NOT measured here, and said so in the snapshot: generation latency before 2026-11-18 (the live clients did not record it; new live runs do: `generation.latency_s`), inference latency
(needs the model, the case store and dataset windows; RAM is shared with the GPU queue), case-retrieval latency, VRAM (NVML is broken on this machine).
Metric names: `layer6_partial.*`. The registry names `layer6.latency_p95_s`, `layer6.cost_per_event_usd`, `layer6.tokens_per_event` are filled only when the data they define exist.
`python -m gridsense.eval.layer6 [--events 60] [--repeats 3]`
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from gridsense.config import ROOT, TRACK
from gridsense.eval.snapshot import write_snapshot


def pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def timing(vals: list[float]) -> dict:
    return {"n": len(vals), "p50_s": pct(vals, 50), "p95_s": pct(vals, 95), "max_s": max(vals) if vals else None}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0          # Linux: KB


def generator_cost(items: list[dict]) -> dict:
    """From live S4 items: per generator call that was billed (cached replies cost 0 and are left out), the cost, tokens and failure rate. Latency only where recorded."""
    billed = [i["generation"] for i in items if i.get("generation") and not i["generation"].get("cached")]
    lat = [g["latency_s"] for g in billed if g.get("latency_s")]
    usd, tin, tout = [g["usd"] for g in billed], [g["input_tokens"] for g in billed], [g["output_tokens"] for g in billed]
    fails = sum(1 for i in items if i.get("error"))
    return {"reports": len(items), "billed_calls": len(billed), "failure_rate": fails / len(items) if items else None, "cost_per_report_usd": float(np.mean(usd)) if usd else None,
            "input_tokens_per_report": float(np.mean(tin)) if tin else None, "output_tokens_per_report": float(np.mean(tout)) if tout else None,
            "generation_latency": timing(lat) if lat else "NOT MEASURED: these live runs did not record per-call latency (new live runs do)"}


def judge_cost(items: list[dict]) -> dict:
    u = [i["judge"]["usage"]["usd"] for i in items if i.get("judge") and i["judge"].get("usage")]
    return {"reports_judged": len(u), "judge_cost_per_report_usd": float(np.mean(u)) if u else None, "note": "evaluation cost, not a cost of the deployed system"}


def profile(n_events: int, repeats: int) -> dict:
    from gridsense.agent import workflow as W
    from gridsense.s4 import checks as CK
    from gridsense.s4 import generate as GEN
    from gridsense.s4 import llm
    from gridsense.s4 import run as R
    ev, order = R.load_events(), None
    order = sorted(ev)
    ids = order[:n_events]
    rss0 = rss_mb()
    retr, tau, rinfo = R.retriever_and_tau()
    rss_loaded = rss_mb()
    retrieval, validation, workflow, same = [], [], [], 0
    gen = llm.MockClient(R.mock_generator)
    deps = {"event_lookup": lambda e: ev[e], "retrieve": lambda item: R.retrieve_all(retr, item, "single"), "tau": tau, "generator": gen, "index_version": rinfo["index_fingerprint"],
            "export_store": {}, "anon_index": lambda e: order.index(e)}
    app = W.build(deps)
    for rep in range(repeats):
        for j, eid in enumerate(ids):
            item = ev[eid]
            t0 = time.perf_counter()
            r1 = R.retrieve_all(retr, item, "single")
            retrieval.append(time.perf_counter() - t0)
            if rep == 0 and j < 20:                                                       # determinism: a rerun gives the same chunk IDs and scores
                r2 = R.retrieve_all(retr, item, "single")
                same += int([(p["chunk_id"], p["score"]) for p in r1["passages"]] == [(p["chunk_id"], p["score"]) for p in r2["passages"]] and r1["dense_top1"] == r2["dense_top1"])
            view = GEN.event_view(item, order.index(eid))
            rep_json, _ = GEN.parse_report(R.mock_generator(GEN.build_messages(item, order.index(eid), r1["passages"]), "generator"))
            t0 = time.perf_counter()
            CK.run_checks(rep_json, view, item["event_object"]["knn"]["neighbours"], r1["passages"], "sufficient")
            validation.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            W.start(app, eid, thread_id=f"prof-{rep}-{eid}")
            workflow.append(time.perf_counter() - t0)
    rss_end = rss_mb()
    return {"document_retrieval": timing(retrieval), "validation": timing(validation), "workflow_end_to_end_mock_generator": timing(workflow),
            "determinism_document_retrieval": {"events_checked": min(20, len(ids)), "identical": same},
            "ram_mb": {"process_peak_before_load": rss0, "after_load_retriever": rss_loaded, "process_peak_end": rss_end}, "events": len(ids), "repeats": repeats, "index_fingerprint": rinfo["index_fingerprint"]}


def run(n_events: int = 60, repeats: int = 3) -> dict:
    prof = profile(n_events, repeats)
    live = []
    for label in ("s4_main_live", "s4_noise_live"):
        for p in sorted((TRACK / "eval" / "items").glob(f"*_{label}.jsonl")):
            live += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    gen_c, jud_c = generator_cost(live), judge_cost(live)
    metrics = {"layer6_partial": {"profile": prof, "generator": gen_c, "judge_evaluation": jud_c,
                                  "not_measured": ["inference latency", "case-retrieval latency", "VRAM (NVML broken)", "generation latency of the S4 live runs", "per-stage latency of a live generation"]}}
    if gen_c["cost_per_report_usd"] is not None:
        metrics["layer6"] = {"cost_per_event_usd": gen_c["cost_per_report_usd"], "tokens_per_event": gen_c["input_tokens_per_report"] + gen_c["output_tokens_per_report"]}   # generator only; registry names
    snap = write_snapshot("s7_layer6_partial", metrics, [], config_effective={"threads": 2, "nice": 19, "events": n_events, "repeats": repeats, "live_items": len(live)},
                          index_fingerprint=prof["index_fingerprint"], notes="Layer 6 partial profile: local stages with a mock generator (no API call) + generator cost from the S4 live items. Latency of generation, inference and case retrieval NOT measured.")
    return {"snapshot": str(snap.relative_to(ROOT)), "profile": prof, "generator": gen_c, "judge": jud_c}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    print(json.dumps(run(a.events, a.repeats), indent=1, default=str))


if __name__ == "__main__":
    main()
