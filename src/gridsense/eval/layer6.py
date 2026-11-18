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


def profile_inference_and_cases(n_events: int = 10, repeats: int = 5, threads: int = 2) -> dict:
    """Local CPU stages that need the checkpoint and the full case store: inference (one window, batch 1), case retrieval (exact FAISS over 1.155 M vectors), gate decision, and the load times.
    Windows: the report window w0025 of the first golden events (test split, already in use as events; only latency is measured, no accuracy). RAM is shared with the GPU queue: 2 threads, nice 19."""
    import torch
    from fault_localization.data.dataset import make_dataloader  # noqa: F401  (import check: the paper's code is on the path)
    from fault_localization.dropout.masking import apply_sensor_dropout_batch
    from fault_localization.models.voting import soft_vote

    from gridsense.cases.store import CaseStore
    from gridsense.data.splits import DATA_ROOT
    from gridsense.model.latent_repr import represent
    from gridsense.model.predictor import LatentCapture, checkpoint_sha256, lean_dataset, load_model
    from gridsense.s4 import run as R
    from gridsense.verification.gate import Gate

    torch.set_num_threads(threads)
    ev = R.load_events()
    wanted = {ev[e]["window_id"]: e for e in sorted(ev)[:n_events]}
    runs = {w.rsplit("_w", 1)[0] for w in wanted}
    t0 = time.perf_counter()
    ds = lean_dataset("test", DATA_ROOT, keep_events=runs)
    t_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, _ = load_model(42, ds.edge_index, "cpu")
    cap = LatentCapture(model)
    t_model = time.perf_counter() - t0
    store_dir = TRACK / "eval" / "case_store" / "seed42_full"
    fp = json.loads((store_dir / "store_manifest.json").read_text())["fingerprint"]
    rss_before = rss_mb()
    t0 = time.perf_counter()
    store = CaseStore.load(store_dir, fp)
    t_store = time.perf_counter() - t0
    gate = Gate.load(TRACK / "configs" / "gate_v1.json")
    sha = checkpoint_sha256(42)
    xs = {}
    for i in range(len(ds)):
        s = ds[i]
        if s["window_id"] in wanted:
            xs[s["window_id"]] = torch.as_tensor(np.asarray(s["x"]), dtype=torch.float32).unsqueeze(0)       # one window: [1, 25, 20, 6]
    inference, case_search, gate_t, e2e = {0: [], 10: []}, {0: [], 10: []}, {0: [], 10: []}, {0: [], 10: []}
    edge_index = getattr(model, "edge_index", None)
    with torch.no_grad():
        for rep in range(repeats):
            for w, x in xs.items():
                for k in (0, 10):
                    t_all = time.perf_counter()
                    xd = x if k == 0 else apply_sensor_dropout_batch(x, k=k, seed=7, policy="zero_messages", edge_index=edge_index)[0]
                    t0 = time.perf_counter()
                    scores, _ = soft_vote(model(xd))
                    h = cap.h.numpy()
                    inference[k].append(time.perf_counter() - t0)
                    q = represent(h, "b_mean_max")
                    t0 = time.perf_counter()
                    sim, idx = store.search(q, 10, query_checkpoint_sha256=sha)
                    case_search[k].append(time.perf_counter() - t0)
                    top = int(scores.numpy().argmax(1)[0])
                    agree = np.array([float((store.labels[idx[0]] == top).mean())])
                    t0 = time.perf_counter()
                    gate.decide(k, scores.numpy(), agree)
                    gate_t[k].append(time.perf_counter() - t0)
                    e2e[k].append(time.perf_counter() - t_all)
    return {"windows": len(xs), "repeats": repeats, "threads": threads, "load_seconds": {"dataset_subset": t_data, "model": t_model, "case_store_1.155M_vectors": t_store},
            "ram_mb": {"before_store": rss_before, "peak_after": rss_mb()}, "per_window_batch1": {f"k{k}": {"inference_with_soft_vote": timing(inference[k]), "case_search_exact": timing(case_search[k]),
                                                                                                          "gate_decision": timing(gate_t[k]), "inference_to_gate": timing(e2e[k])} for k in (0, 10)}}


def run(n_events: int = 60, repeats: int = 3) -> dict:
    prof = profile(n_events, repeats)
    live = []
    for label in ("s4_main_live", "s4_noise_live"):
        for p in sorted((TRACK / "eval" / "items").glob(f"*_{label}.jsonl")):
            live += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    safety = []
    for p in sorted((TRACK / "eval" / "items").glob("*_s6_safety_live.jsonl")):
        safety += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    gen_c, jud_c = generator_cost(live), judge_cost(live)
    gen_safety = generator_cost(safety) if safety else None
    metrics = {"layer6_partial": {"profile": prof, "generator": gen_c, "generator_safety_live_runs": gen_safety, "judge_evaluation": jud_c,
                                  "not_measured": ["VRAM (NVML broken)", "generation latency of the S4 live runs (before 2026-11-18 not recorded; the safety live runs have it)", "end-to-end latency with a live generator (stages are measured apart: sums of p95 are not a p95)"],
                                  "see_also": "snapshot s7_layer6_inference_cases for inference, case search and gate"}}
    if gen_c["cost_per_report_usd"] is not None:
        metrics["layer6"] = {"cost_per_event_usd": gen_c["cost_per_report_usd"], "tokens_per_event": gen_c["input_tokens_per_report"] + gen_c["output_tokens_per_report"]}   # generator only; registry names
    snap = write_snapshot("s7_layer6_partial", metrics, [], config_effective={"threads": 2, "nice": 19, "events": n_events, "repeats": repeats, "live_items": len(live)},
                          index_fingerprint=prof["index_fingerprint"], notes="Layer 6 partial profile: local stages with a mock generator (no API call) + generator cost from the S4 live items + generation latency from the safety live runs. VRAM and end-to-end live latency NOT measured.")
    return {"snapshot": str(snap.relative_to(ROOT)), "profile": prof, "generator": gen_c, "judge": jud_c}


def run_heavy(n_events: int = 10, repeats: int = 5) -> dict:
    prof = profile_inference_and_cases(n_events, repeats)
    snap = write_snapshot("s7_layer6_inference_cases", {"layer6_partial": {"inference_and_cases": prof}}, [], config_effective={"threads": 2, "nice": 19, "events": n_events, "repeats": repeats},
                          notes="Layer 6: inference, exact case search over 1.155 M vectors and gate decision, batch 1, CPU 2 threads, seed-42 checkpoint. Latency only; no accuracy is measured. VRAM not measured (NVML broken).")
    return {"snapshot": str(snap.relative_to(ROOT)), "profile": prof}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--heavy", action="store_true", help="inference + case search + gate on the CPU (needs the checkpoint and the full case store, about 1.3 GB RAM)")
    a = ap.parse_args()
    print(json.dumps(run_heavy(min(a.events, 10), a.repeats) if a.heavy else run(a.events, a.repeats), indent=1, default=str))


if __name__ == "__main__":
    main()
