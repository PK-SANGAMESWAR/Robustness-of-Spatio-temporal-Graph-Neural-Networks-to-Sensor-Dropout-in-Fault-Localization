"""Layer 1 component sweeps (EVAL.md § 3; skill rag-component-sweep). One variable at a time; the winner of an axis is fixed for the next.

    PYTHONPATH=src:. OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -m gridsense.eval.sweep run --axis chunker
    ... --axis chunk_size | overlap | embedder [--allow-paid] | retrieval_mode | reranker | candidates | top_k
    ... run-all [--allow-paid]          (the whole chain, in order)
    ... frozen --configs final local baseline   (the ONE run on the golden test split per frozen configuration; appends to GOLDEN_TEST_USE.md)

Selection rule v2 (author, 2026-11-02: "look for best + cheap", not just cheap). Rule v1 (cheapest candidate inside the 95 % interval of the best) was used for a first
run of three axes and was replaced after those dev results had been seen; its results are kept in reports/sweeps/rule_v1/.
  * dev split only (never the test split); golden set v1 (reviewed); no comparison with any earlier baseline;
  * quality Q = mean over the answerable dev queries of (MRR@10 + Recall@5 + nDCG@5) / 3; the best candidate has the highest Q (top_k axis: Recall@k alone);
  * a candidate is "as good as the best" when Q_best - Q <= 1 / (n_queries * n_metrics) (one query on one metric: the resolution of the dev set;
    0.013 for n = 25 and 3 metrics) and, among those, it has the fewest golden spans split by chunk boundaries;
  * the winner is the cheapest of the candidates that are as good as the best. Cost: mean context tokens at k = 5 for the chunk axes (chunker, chunk_size,
    overlap: what the generator must read); otherwise the declared order in ORDER (free before paid, one stage before two, smaller before larger);
  * the paired bootstrap interval (2,000 resamples, seed 42) of best - candidate is reported for information; it does not choose;
  * the OpenAI embedder runs through the cost ledger with a cap of CAP_USD, the author's approved estimate (O31) with a margin.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gridsense.config import ROOT, TRACK, load_config
from gridsense.cost import Ledger
from gridsense.eval import golden as G
from gridsense.eval import retrieval_metrics as RM
from gridsense.eval.layer1 import embedder_from_config, evaluate, retrieval_config
from gridsense.eval.snapshot import write_snapshot
from gridsense.ingest.corpus import load_corpus
from gridsense.retrieval.embeddings import Embedder

METER = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY = "mrr@10"
CAP_USD = 0.002                      # O31: estimate 0.0013 USD (2026-10-31); the ledger stops the run above this cap
SWEEPS = TRACK / "reports" / "sweeps"
WINNERS = SWEEPS / "WINNERS.json"
AXES = ["chunker", "chunk_size", "overlap", "embedder", "retrieval_mode", "reranker", "candidates", "top_k"]
CHUNKERS = ["fixed_token", "recursive", "structure_aware", "sentence", "semantic", "parent_child"]
SIZES = [100, 150, 200]              # tokens (MiniLM word pieces); the largest fits the 256-token window with the section header
OVERLAPS = [0.0, 0.10, 0.15]
EMBEDDERS = ["minilm", "bge_small", "e5_small", "openai_3_small"]
MODES = ["dense", "bm25", "hybrid_k20", "hybrid_k60", "hybrid_k100"]
# cheapest or simplest first: used only to break a tie inside the confidence interval
ORDER = {
    "chunker": ["fixed_token", "recursive", "sentence", "structure_aware", "parent_child", "semantic"],
    "chunk_size": ["size_100", "size_150", "size_200"],
    "overlap": ["overlap_0.00", "overlap_0.10", "overlap_0.15"],
    "embedder": EMBEDDERS,
    "retrieval_mode": ["bm25", "dense", "hybrid_k60", "hybrid_k20", "hybrid_k100"],
    "reranker": ["rerank_off", "rerank_on"],
    "candidates": ["cand_20", "cand_50"],
    "top_k": ["k3", "k5", "k8"],
}


def base_state() -> dict:
    rc = retrieval_config({"chunk_meter": METER})
    return {"retrieval": asdict(rc), "embedder": "minilm"}


def candidates(axis: str, state: dict) -> list[tuple[str, dict, str]]:
    """(name, RetrievalConfig fields, embedder name) for each candidate of the axis; the others stay at the current winner."""
    rc, emb = state["retrieval"], state["embedder"]
    out: list[tuple[str, dict, str]] = []
    if axis == "chunker":
        for c in CHUNKERS:
            out.append((c, {**rc, "chunker": c, "parent_size": 4 * rc["chunk_size"] if c == "parent_child" else None, "overlap": 0.0}, emb))
    elif axis == "chunk_size":
        for s in SIZES:
            out.append((f"size_{s}", {**rc, "chunk_size": s, "parent_size": 4 * s if rc["chunker"] == "parent_child" else None}, emb))
    elif axis == "overlap":
        if rc["chunker"] == "parent_child":
            return []                                                 # no overlap in this chunker: the axis does not apply
        out = [(f"overlap_{o:.2f}", {**rc, "overlap": o}, emb) for o in OVERLAPS]
    elif axis == "embedder":
        out = [(e, rc, e) for e in EMBEDDERS]
    elif axis == "retrieval_mode":
        for m in MODES:
            mode, k = ("hybrid", int(m.split("_k")[1])) if m.startswith("hybrid") else (m, rc["rrf_k"])
            out.append((m, {**rc, "mode": mode, "rrf_k": k}, emb))
    elif axis == "reranker":
        cfg = load_config()["reranker"]
        out = [("rerank_off", {**rc, "reranker": None}, emb), ("rerank_on", {**rc, "reranker": cfg["model"], "reranker_revision": cfg["revision"], "rerank_top": cfg["rerank_top"]}, emb)]
    elif axis == "candidates":
        out = [(f"cand_{c}", {**rc, "dense_candidates": c, "lexical_candidates": c}, emb) for c in (20, 50)]
    elif axis == "top_k":
        out = [(f"k{k}", {**rc, "top_k": k}, emb) for k in (3, 5, 8)]
    else:
        raise ValueError(axis)
    return out


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95 % bootstrap interval of mean(a - b) over the same queries."""
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def select(axis: str, per_q: dict[str, np.ndarray], cost: dict[str, float] | None = None, splits: dict[str, int] | None = None, n_metrics: int = 3) -> dict:
    """Rule v2: best quality, then the cheapest candidate that is as good as the best (see the module docstring)."""
    names, order = list(per_q), ORDER[axis]
    n = len(next(iter(per_q.values())))
    resolution = 1.0 / (n * n_metrics)
    q = {k: float(v.mean()) for k, v in per_q.items()}
    best = max(names, key=lambda k: (q[k], -order.index(k)))
    band = [k for k in names if q[best] - q[k] <= resolution + 1e-12]
    if splits is not None:                      # a chunking that cuts golden spans is mechanically worse: keep only the candidates with the fewest cut spans
        fewest = min(splits[k] for k in band)
        band = [k for k in band if splits[k] == fewest]
    cost = cost or {k: float(order.index(k)) for k in names}
    winner = min(band, key=lambda k: (cost[k], order.index(k)))
    ci = {k: paired_ci(per_q[best], per_q[k]) for k in names if k != best}
    if winner == best:
        reason = f"{best} has the best quality" + (f" and is also the cheapest of the {len(band)} candidates within {resolution:.3f} of it" if len(band) > 1 else "; no other candidate is within " + f"{resolution:.3f} of it")
    else:
        reason = f"{best} has the best quality (Q = {q[best]:.3f}); {winner} is the cheapest of the {len(band)} candidates within {resolution:.3f} of it (Q = {q[winner]:.3f})"
    return {"best": best, "band": band, "winner": winner, "quality": q, "cost": cost, "resolution": resolution, "splits": splits,
            "diff_best_minus_candidate_ci": ci, "reason": reason, "rule": "v2"}


def estimate_openai(rc, docs, rows: list[dict], emb: Embedder, ledger: Ledger) -> dict:
    """Tokens of everything that the OpenAI embedder would send (passages and dev queries), minus nothing: the disk cache can only lower the real cost."""
    from gridsense.ingest.chunking import chunk_document
    from gridsense.retrieval.embeddings import get_meter
    meter = get_meter(rc.chunk_meter)
    chunks = [c for d in docs for c in chunk_document(d, rc.chunker, rc.chunk_size, meter.count_tokens, rc.overlap, parent_size=rc.parent_size)]
    pt = sum(emb.count_tokens(c.embed_text) for c in chunks)
    qt = sum(emb.count_tokens(r["query"]) for r in rows)
    return {"passage_tokens": pt, "query_tokens": qt, "tokens": pt + qt, "usd": ledger.price(emb.spec.model, pt + qt, 0)}


def load_winners() -> dict:
    return json.loads(WINNERS.read_text()) if WINNERS.exists() else {"state": base_state(), "axes": {}}


def _fmt(m: dict, nd: int = 2) -> str:
    return f"{m['mean']:.{nd}f} ({m['ci_low']:.2f}–{m['ci_high']:.2f})" if m.get("n") else "–"


def worst_queries(out: dict, rows: list[dict], n: int = 5) -> list[dict]:
    by = {r["query_id"]: r for r in rows}
    items = [x for x in out["items"] if x["answerable"] and "error" not in x]
    items.sort(key=lambda x: (x[PRIMARY], x["recall@5"], x["query_id"]))
    res = []
    for x in items[:n]:
        gold = [c["document_id"] for c in by[x["query_id"]]["golden_contexts"]]
        top = x["retrieved"][0]
        first = int(round(1 / x[PRIMARY])) if x[PRIMARY] > 0 else None        # rank of the first passage that contains a golden span
        res.append({"query_id": x["query_id"], "query": by[x["query_id"]]["query"], "gold": gold, "rank_of_first_hit": first, "top1": f"{top['document_id']} › {' › '.join(top['section_path'][-1:])}"})
    return res


def write_report(axis: str, state_in: dict, results: dict, sel: dict, rows: list[dict], worst: list[dict], extra: list[str], cost: dict | None) -> Path:
    SWEEPS.mkdir(parents=True, exist_ok=True)
    n_ans = sum(r["answerable"] for r in rows)
    fixed = {k: v for k, v in state_in["retrieval"].items() if k in ("chunker", "chunk_size", "overlap", "parent_size", "mode", "rrf_k", "reranker", "dense_candidates", "store", "top_k")}
    L = [f"# Sweep — {axis}", "", f"Written for: the author. Generated by `gridsense.eval.sweep` on {datetime.now(timezone.utc):%Y-%m-%d}. Golden set v1 (reviewed), **dev split only**: "
         f"{len(rows)} queries, {n_ans} answerable, {len(rows) - n_ans} unanswerable. CPU, 2 threads. 95 % intervals: bootstrap over queries. No comparison with any earlier baseline.", "",
         f"Fixed during this step (the winners so far): embedder `{state_in['embedder']}`, " + ", ".join(f"{k} = {v}" for k, v in fixed.items()) + ".", "",
         "## Quality (answerable dev queries)", "", "| candidate | chunks | split spans | R@1 | R@3 | R@5 | R@8 | MRR@10 | nDCG@5 | id-recall@5 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        m = r["metrics"]; d = m["dev"]; re_ = m["reachability"]
        L.append(f"| {n} | {m['index']['n_chunks']} | {re_['contexts'] - re_['reachable']} of {re_['contexts']} | " + " | ".join(_fmt(d[k]) for k in ("recall@1", "recall@3", "recall@5", "recall@8", "mrr@10", "ndcg@5")) + f" | {_fmt(d['identifier_recall@5']) if 'identifier_recall@5' in d else '–'} |")
    L += ["", "## No-answer behaviour and operations", "", "τ is the no-answer threshold on the top-1 dense similarity, fitted on dev only (so dev numbers are in-sample; n = "
          f"{len(rows) - n_ans} unanswerable queries: read them as a rough guide).", "",
          "| candidate | τ (dev) | unanswerable flagged | answerable flagged | context tokens @5 | truncated chunks | build s | latency p50 / p95 ms | RSS MB |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        m = r["metrics"]; sf = m["dev"]["sufficiency"]; tau = m["sufficiency"]["tau_dev"].get("tau")
        na = f"{sf['unanswerable_flagged_insufficient']:.2f} | {sf['answerable_flagged_insufficient']:.2f}" if tau is not None else "– | –"
        L.append(f"| {n} | {'–' if tau is None else format(tau, '.3f')} | {na} | {m['dev']['ctx_tokens_mean']['5']:.0f} | {m['index']['truncated']} | "
                 f"{m['index']['build_seconds']:.1f} | {m['latency_ms']['p50']:.1f} / {m['latency_ms']['p95']:.1f} | {m['index']['rss_mb_after']:.0f} |")
    ck = axis in ("chunker", "chunk_size", "overlap")
    L += ["", "## Selection (rule v2: best quality, then cheap)", "",
          f"Quality Q = mean of MRR@10, Recall@5 and nDCG@5. As good as the best = within {sel['resolution']:.3f} (one query on one of three metrics)" + (" and, among those, the fewest split golden spans" if ck else "") +
          f". Cost = {'mean context tokens at k = 5' if ck else 'the declared order (free before paid, simpler before more complex)'}. The interval is for information only.", "",
          "| candidate | quality Q | best − candidate (95 % interval, information) | as good as the best | cost | split spans |", "|---|---|---|---|---|---|"]
    for n in results:
        d = sel["diff_best_minus_candidate_ci"].get(n)
        re_ = results[n]["metrics"]["reachability"]
        L.append(f"| {n} | {sel['quality'][n]:.3f} | " + (f"{sel['quality'][sel['best']] - sel['quality'][n]:+.3f} ({d[0]:+.3f} to {d[1]:+.3f})" if d else "(the best)") +
                 f" | {'yes' if n in sel['band'] else 'no'} | {sel['cost'][n]:.0f} | {re_['contexts'] - re_['reachable']} |")
    L += ["", f"**Winner: `{sel['winner']}`.** Reason: {sel['reason']}.", ""]
    if cost:
        L += [f"**Paid calls on this axis:** estimated {cost['estimate_usd']:.6f} USD before the call; actual {cost['actual_usd']:.6f} USD (cap {CAP_USD} USD; ledger total {cost['ledger_total_usd']:.6f} USD of 5 USD).", ""]
    L += extra
    L += ["## The 5 worst queries of the winner (listed by the script; read by hand in the sweep summary)", "", "| query | question | gold document | rank of the first passage with a golden span | top-1 passage |", "|---|---|---|---|---|"]
    for w in worst:
        L.append(f"| {w['query_id']} | {w['query']} | {', '.join(sorted(set(w['gold'])))} | {w['rank_of_first_hit'] or '>10'} | {w['top1']} |")
    p = SWEEPS / f"{axis}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def run_top_k(win: dict, rows: list[dict], docs, n_boot: int) -> dict:
    """top_k is a read-out of one run (the retriever always fetches 10): recall@k and the context size for k = 3, 5, 8 and 10."""
    state = win["state"]
    rc = replace(retrieval_config({}), **state["retrieval"])
    emb = embedder_from_config(state["embedder"])
    out = evaluate(rows, docs, rc, emb, n_boot, chunk_embedder=embedder_from_config("minilm"))
    ok = {x["query_id"]: x for x in out["items"] if x["answerable"] and "error" not in x}
    qids = sorted(ok)
    per_q = {f"k{k}": np.array([ok[q][f"recall@{k}"] for q in qids]) for k in (3, 5, 8)}
    d = out["metrics"]["dev"]
    sel = select("top_k", per_q, cost={k: float(d["ctx_tokens_mean"][k[1:]]) for k in per_q}, n_metrics=1)
    ref = np.array([ok[q]["recall@10"] for q in qids])
    L = ["# Sweep — top_k", "", f"Written for: the author. Generated by `gridsense.eval.sweep`. Golden set v1, **dev split only**, {len(qids)} answerable queries. top_k is a read-out of one run of the current winner "
         f"(embedder `{state['embedder']}`, chunker `{state['retrieval']['chunker']}`, size {state['retrieval']['chunk_size']}, mode `{state['retrieval']['mode']}`); the retriever always fetches 10 passages.", "",
         "| k | recall@k | precision@k | context coverage@k | nDCG@k | mean context tokens | recall@10 − recall@k (95 % interval) |", "|---|---|---|---|---|---|---|"]
    for k in (3, 5, 8, 10):
        lo, hi = paired_ci(ref, np.array([ok[q][f"recall@{k}"] for q in qids]))
        L.append(f"| {k} | {_fmt(d[f'recall@{k}'])} | {_fmt(d[f'precision@{k}'])} | {_fmt(d[f'context_coverage@{k}'])} | {_fmt(d[f'ndcg@{k}'])} | {d['ctx_tokens_mean'][str(k)]:.0f} | {np.mean(ref) - np.mean([ok[q][f'recall@{k}'] for q in qids]):+.3f} ({lo:+.3f} to {hi:+.3f}) |")
    L += ["", f"**Winner: `{sel['winner']}`** (rule v2: the best recall@k among k = 3, 5, 8; then the cheapest k within one query ({sel['resolution']:.2f}) of it; cost = mean context tokens). Reason: {sel['reason']}.", ""]
    p = SWEEPS / "top_k.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_snapshot("sweep_top_k_readout", out["metrics"], out["items"], config_effective={"retrieval": state["retrieval"], "embedder": state["embedder"], "axis": "top_k"},
                   index_fingerprint=out["ix"].fingerprint, golden_versions={"retrieval": 1, "split": "dev", "items": len(rows)}, notes="top_k read-out; dev split only")
    k = int(sel["winner"][1:])
    win["state"]["retrieval"]["top_k"] = k
    win["axes"]["top_k"] = {"winner": sel["winner"], "best": sel["best"], "as_good_as_best": sel["band"], "reason": sel["reason"], "rule": "v2",
                            "report": str(p.relative_to(ROOT)), "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    WINNERS.write_text(json.dumps(win, indent=1, ensure_ascii=False))
    print(f"[top_k] winner: {sel['winner']} ({sel['reason']})")
    return win["axes"]["top_k"]


def run_axis(axis: str, allow_paid: bool = False) -> dict:
    cfg = load_config()
    rows = [r for r in G.load(ROOT / cfg["golden_sets"]["retrieval"]["path"], statuses=("reviewed",)) if r["split"] == "dev"]
    docs = load_corpus()
    assert not G.validate(rows, docs)
    win = load_winners()
    idx = AXES.index(axis)
    if idx and AXES[idx - 1] not in win["axes"]:
        raise SystemExit(f"the axis before '{axis}' is '{AXES[idx - 1]}': it has no recorded winner in {WINNERS.relative_to(ROOT)}")
    if axis == "top_k":
        return run_top_k(win, rows, docs, cfg["repeats"]["bootstrap_resamples"])
    state = json.loads(json.dumps(win["state"]))
    cands = candidates(axis, state)
    if not cands:
        win["axes"][axis] = {"skipped": "not applicable for the chunker in use"}
        WINNERS.write_text(json.dumps(win, indent=1))
        return win["axes"][axis]
    n_boot = cfg["repeats"]["bootstrap_resamples"]
    ledger = Ledger(limit_usd=CAP_USD, label=f"sweep_{axis}")
    embedders: dict[str, Embedder] = {}
    def emb_of(name: str) -> Embedder:
        if name not in embedders:
            embedders[name] = embedder_from_config(name, ledger=ledger if name.startswith("openai") else None)
        return embedders[name]
    paid = None
    results: dict[str, dict] = {}
    for name, rcd, en in cands:
        if en.startswith("openai"):
            if not allow_paid:
                print(f"[skip] {name}: paid; run with --allow-paid (O31 approved the call; estimate in EXPERIMENT_LOG)")
                continue
            spent0, est = ledger.spent(), estimate_openai(replace(retrieval_config({}), **rcd), docs, rows, emb_of(en), ledger)
            print(f"[{axis}] {name}: estimate {est['usd']:.6f} USD ({est['tokens']:,} tokens: {est['passage_tokens']:,} passages + {est['query_tokens']:,} queries); cap {CAP_USD} USD", flush=True)
            if est["usd"] > CAP_USD:
                raise SystemExit(f"estimate {est['usd']:.6f} USD is above the approved cap {CAP_USD} USD: stop and ask the author")
        rc = replace(retrieval_config({}), **rcd)
        out = evaluate(rows, docs, rc, emb_of(en), n_boot, chunk_embedder=emb_of("minilm"), rebuild=not en.startswith("openai"))   # rebuild: measure the build time (the OpenAI cache would hide it)
        if en.startswith("openai"):
            paid = {"estimate_usd": est["usd"], "actual_usd": ledger.spent() - spent0, "ledger_total_usd": ledger.spent(), "tokens_estimated": est["tokens"]}
        per_q = {x["query_id"]: (x["mrr@10"] + x["recall@5"] + x["ndcg@5"]) / 3 for x in out["items"] if x["answerable"] and "error" not in x}   # quality Q
        results[name] = {"metrics": out["metrics"], "items": out["items"], "ix": out["ix"], "per_q": per_q, "rc": rcd, "emb": en, "out": out}
        write_snapshot(f"sweep_{axis}_{name}", out["metrics"], out["items"], config_effective={"retrieval": rcd, "embedder": emb_of(en).spec.key(), "axis": axis},
                       index_fingerprint=out["ix"].fingerprint, golden_versions={"retrieval": cfg["golden_sets"]["retrieval"]["version"], "split": "dev", "items": len(rows)},
                       notes=f"sweep axis {axis}, candidate {name}; dev split only; 11 corpus notes and the SOP are still draft (O34)", cost_usd=float(paid["actual_usd"]) if (paid and en.startswith("openai")) else 0.0)
        m = out["metrics"]["dev"]
        print(f"[{axis}] {name:16} MRR@10 {m['mrr@10']['mean']:.3f}  R@1 {m['recall@1']['mean']:.2f}  R@5 {m['recall@5']['mean']:.2f}  chunks {out['metrics']['index']['n_chunks']}  split spans {out['reach']['contexts'] - out['reach']['reachable']}", flush=True)
    qids = sorted(set.intersection(*[set(r["per_q"]) for r in results.values()]))
    per_q = {n: np.array([r["per_q"][q] for q in qids]) for n, r in results.items()}
    for r in results.values():
        r["per_q"] = np.array([r["per_q"][q] for q in qids])
    ck = axis in ("chunker", "chunk_size", "overlap")
    cost = {n: float(r["metrics"]["dev"]["ctx_tokens_mean"]["5"]) for n, r in results.items()} if ck else None
    splits = {n: r["metrics"]["reachability"]["contexts"] - r["metrics"]["reachability"]["reachable"] for n, r in results.items()}
    sel = select(axis, per_q, cost=cost, splits=splits)
    win_r = results[sel["winner"]]
    extra = []
    split_lines = [f"- `{n}`: " + "; ".join(f"{s['query_id']} ({s['document_id']}): “{s['span']}…”" for s in r["out"]["reach"]["split_spans"]) for n, r in results.items() if r["out"]["reach"]["split_spans"]]
    if split_lines:
        extra += ["## Golden spans that no chunk contains (they cap recall for that candidate)", ""] + split_lines + [""]
    report = write_report(axis, state, results, sel, rows, worst_queries(win_r["out"], rows), extra, paid)
    new_state = {"retrieval": win_r["rc"], "embedder": win_r["emb"]}
    win["axes"][axis] = {"winner": sel["winner"], "best": sel["best"], "as_good_as_best": sel["band"], "reason": sel["reason"], "rule": "v2",
                         "quality": sel["quality"], "report": str(report.relative_to(ROOT)), "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    win["state"] = new_state
    WINNERS.write_text(json.dumps(win, indent=1, ensure_ascii=False))
    print(f"[{axis}] winner: {sel['winner']} ({sel['reason']})  -> {report.relative_to(ROOT)}")
    return win["axes"][axis]


USE_LEDGER = TRACK / "reports" / "layer1" / "GOLDEN_TEST_USE.md"


def run_frozen() -> Path:
    """The ONE run per frozen configuration on the golden TEST split (author rule: test once per frozen configuration). tau comes from dev only.
    Configurations: final (the sweep winners), final_chroma (same, Chroma store: precision parity on test), final_dense (same, dense only: is the hybrid gain real on test?)."""
    from gridsense.eval.layer1 import reachability
    cfg = load_config()
    rows = G.load(ROOT / cfg["golden_sets"]["retrieval"]["path"], statuses=("reviewed",))
    docs = load_corpus()
    assert not G.validate(rows, docs)
    state = load_winners()["state"]
    missing = [a for a in AXES + ["vector_store"] if a not in load_winners()["axes"]]
    if missing:
        raise SystemExit(f"cannot freeze: no recorded winner for {missing}")
    variants = {"final": {}, "final_chroma": {"store": "chroma"}, "final_dense": {"mode": "dense"}}
    done = USE_LEDGER.read_text() if USE_LEDGER.exists() else ""
    emb, minilm = embedder_from_config(state["embedder"]), embedder_from_config("minilm")
    test_ans = [r for r in rows if r["split"] == "test" and r["answerable"]]
    res = {}
    for name, ch in variants.items():
        rc = replace(retrieval_config({}), **{**state["retrieval"], **ch})
        from gridsense.retrieval.retriever import index_fingerprint
        fp = index_fingerprint(rc, emb, docs, minilm)[:16]
        key = f"| {name} | {fp} |"
        if key in done:
            raise SystemExit(f"{name} (index {fp}) was already run once on the golden test split (see {USE_LEDGER.relative_to(ROOT)}): a second run is not allowed")
        out = evaluate(rows, docs, rc, emb, cfg["repeats"]["bootstrap_resamples"], chunk_embedder=minilm, rebuild=True)
        out["reach_test"] = reachability(out["ix"], test_ans)
        snap = write_snapshot(f"frozen_{name}_test", out["metrics"], out["items"], config_effective={"retrieval": asdict(rc), "embedder": emb.spec.key(), "frozen": name},
                              index_fingerprint=out["ix"].fingerprint, golden_versions={"retrieval": cfg["golden_sets"]["retrieval"]["version"], "split": "dev (tau) + test (numbers)", "items": len(rows)},
                              notes="FROZEN configuration, ONE run on the golden test split; tau from dev only; 11 corpus notes and the SOP are still draft (O34)")
        res[name] = {"out": out, "snapshot": str(snap.relative_to(ROOT)), "fp": fp, "rc": asdict(rc)}
        print(f"[frozen] {name}: test MRR@10 {out['metrics']['test']['mrr@10']['mean']:.3f}  R@1 {out['metrics']['test']['recall@1']['mean']:.2f}  R@5 {out['metrics']['test']['recall@5']['mean']:.2f}", flush=True)
    # paired comparison on test (information): final vs final_dense
    def per_q(name, key):
        return {x["query_id"]: x[key] for x in res[name]["out"]["items"] if x["split"] == "test" and x["answerable"] and "error" not in x}
    qids = sorted(per_q("final", "mrr@10"))
    d_mrr = paired_ci(np.array([per_q("final", "mrr@10")[q] for q in qids]), np.array([per_q("final_dense", "mrr@10")[q] for q in qids]))
    d_r1 = paired_ci(np.array([per_q("final", "recall@1")[q] for q in qids]), np.array([per_q("final_dense", "recall@1")[q] for q in qids]))
    L = ["# Frozen configurations: the one run on the golden test split", "", f"Written for: the author. Generated by `gridsense.eval.sweep frozen` on {datetime.now(timezone.utc):%Y-%m-%d}. Golden set v1 (reviewed). The configuration was frozen "
         "from the dev sweeps (`WINNERS.json`); each configuration ran **once** on the test split ({} answerable and {} unanswerable queries); the no-answer threshold tau comes from dev only. "
         "The test split played no part in any choice. 11 corpus notes and the SOP are still draft (O34); the numbers are for the golden set as reviewed.".format(len(test_ans), sum(1 for r in rows if r["split"] == "test" and not r["answerable"])), "",
         "| configuration | R@1 | R@3 | R@5 | MRR@10 | nDCG@5 | P@3 | P@5 | split test spans | unanswerable flagged / answerable flagged | tau (dev) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        t, rt = r["out"]["metrics"]["test"], r["out"]["reach_test"]
        L.append(f"| {name} | " + " | ".join(_fmt(t[k]) for k in ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@5", "precision@3", "precision@5")) +
                 f" | {rt['contexts'] - rt['reachable']} of {rt['contexts']} | {t['sufficiency'].get('unanswerable_flagged_insufficient', float('nan')):.2f} / {t['sufficiency'].get('answerable_flagged_insufficient', float('nan')):.2f} | {r['out']['metrics']['sufficiency']['tau_dev'].get('tau', float('nan')):.3f} |")
    L += ["", "95 % intervals: bootstrap over the test queries. Precision@k cannot exceed (golden contexts of the query) / k, so P@5 is capped at 0.2 to 0.4.", "",
          f"**Hybrid against dense, on test (information, paired over {len(qids)} queries):** MRR@10 final − dense: {np.mean([per_q('final','mrr@10')[q] - per_q('final_dense','mrr@10')[q] for q in qids]):+.3f} (95 % interval {d_mrr[0]:+.3f} to {d_mrr[1]:+.3f}); "
          f"Recall@1: {np.mean([per_q('final','recall@1')[q] - per_q('final_dense','recall@1')[q] for q in qids]):+.3f} ({d_r1[0]:+.3f} to {d_r1[1]:+.3f}).", "",
          "**Store parity on test:** " + ("the Chroma run retrieved the same top-10 lists as the Qdrant run for every test query." if all([x["chunk_id"] for x in a["retrieved"]] == [x["chunk_id"] for x in b["retrieved"]]
           for a, b in zip(res["final"]["out"]["items"], res["final_chroma"]["out"]["items"])) else "the lists of the Chroma and Qdrant runs DIFFER for at least one test query: see the snapshots."), "",
          "Snapshots: " + ", ".join(f"`{r['snapshot']}`" for r in res.values()) + "."]
    p = SWEEPS / "FROZEN_TEST.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    new = not USE_LEDGER.exists()
    with USE_LEDGER.open("a") as f:
        if new:
            f.write("# Golden test split: use ledger (append-only)\n\nEach row is the ONE run of a frozen configuration on the golden retrieval test split. Separate from the STGNN test-split ledger (`reports/layer0/TEST_USE.md`).\n\n| config | index fingerprint | date | snapshot |\n|---|---|---|---|\n")
        for name, r in res.items():
            f.write(f"| {name} | {r['fp']} | {datetime.now(timezone.utc):%Y-%m-%d} | `{r['snapshot']}` |\n")
    (SWEEPS / "FROZEN_TEST.json").write_text(json.dumps({n: {"metrics": r["out"]["metrics"]["test"], "config": r["rc"], "snapshot": r["snapshot"]} for n, r in res.items()}, indent=1, default=float))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--axis", choices=AXES, required=True); r.add_argument("--allow-paid", action="store_true")
    a = sub.add_parser("run-all"); a.add_argument("--allow-paid", action="store_true")
    sub.add_parser("frozen")
    ns = ap.parse_args()
    if ns.cmd == "frozen":
        print(run_frozen())
    elif ns.cmd == "run":
        run_axis(ns.axis, ns.allow_paid)
    else:
        for ax in AXES:
            run_axis(ax, ns.allow_paid)


if __name__ == "__main__":
    main()
