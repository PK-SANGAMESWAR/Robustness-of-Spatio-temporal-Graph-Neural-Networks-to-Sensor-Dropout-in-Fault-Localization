"""Vector-store benchmark (sweep axis 7; EVAL.md § 3). With exact search every store returns the same ranking for the same vectors, so quality does not choose:
parity with FAISS-flat must be 100 %, then the operational numbers decide. Each store runs in a FRESH process (clean peak memory).

    PYTHONPATH=src:. OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -m gridsense.eval.store_bench run        (prepare, run the three stores, write the report)

Stores: faiss_flat (reference), chroma (high-recall HNSW), chroma_default (Chroma's defaults), qdrant_local.
Measured per store: build time, query latency p50/p95 (dev queries, top 10, 5 repeats), top-10 parity and overlap with FAISS-flat, peak RSS of the fresh process
(imports included), disk size of the persisted index, re-open time, and a metadata filter test. NOT measured: approximate-search recall (Qdrant local mode always searches
exactly; Chroma's HNSW is measured through the parity numbers) and native hybrid search (documented, not tested here).
Rule v2.1 (author, 2026-11-02: a real vector store, not FAISS; precision must be good): FAISS-flat is the exact reference only and is never eligible. Eligible = native metadata
filter works, dense top-10 identical to exact search, and the retrieved lists and all metrics identical to the reference end to end (the real hybrid retriever, dev golden queries). Quality is
equal for the eligible stores, so the cheapest wins: lowest p95 latency; within 10 % is a tie, broken by the declared order (qdrant_local, chroma, chroma_default).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from gridsense.config import ROOT, TRACK

WORK = TRACK / "eval" / "store_bench"
ORDER = ["qdrant_local", "chroma", "chroma_default", "faiss_flat"]      # tie order of the eligible stores; faiss_flat is the exact reference and is never eligible
REFERENCE = "faiss_flat"
FILTER = {"approval_status": ["curated_for_research"]}


def prepare() -> dict:
    """Final index of the sweep winners + dev query vectors + the exact FAISS top-10 per query."""
    import gridsense.settings  # noqa: F401
    from gridsense.eval import golden as G
    from gridsense.eval.layer1 import embedder_from_config, get_index
    from gridsense.eval.sweep import load_winners
    from gridsense.config import load_config
    from gridsense.ingest.corpus import load_corpus
    from gridsense.retrieval.retriever import RetrievalConfig
    from dataclasses import replace
    from gridsense.eval.layer1 import retrieval_config

    state = load_winners()["state"]
    rc = replace(retrieval_config({}), **state["retrieval"])
    emb = embedder_from_config(state["embedder"])
    ix = get_index(rc, emb, load_corpus(), embedder_from_config("minilm"))
    rows = [r for r in G.load(ROOT / load_config()["golden_sets"]["retrieval"]["path"], statuses=("reviewed",)) if r["split"] == "dev"]
    q = emb.encode([r["query"] for r in rows], "query")
    ids = [c.chunk_id for c in ix.chunks]
    D, I = _faiss_exact(ix.vectors, q, 10)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "vectors.npy", ix.vectors), np.save(WORK / "queries.npy", q)
    (WORK / "meta.json").write_text(json.dumps({"ids": ids, "payloads": ix.payloads, "exact_top10": [[ids[j] for j in row] for row in I],
                                                "n_vectors": len(ids), "dim": int(ix.vectors.shape[1]), "n_queries": len(q), "fingerprint": ix.fingerprint}))
    return {"n_vectors": len(ids), "dim": int(ix.vectors.shape[1]), "n_queries": len(q), "fingerprint": ix.fingerprint[:16]}


def _faiss_exact(v: np.ndarray, q: np.ndarray, k: int):
    import faiss
    ix = faiss.IndexFlatIP(v.shape[1]); ix.add(np.ascontiguousarray(v, np.float32))
    return ix.search(np.ascontiguousarray(q, np.float32), k)


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size


def bench_one(store: str) -> dict:
    v, q = np.load(WORK / "vectors.npy"), np.load(WORK / "queries.npy")
    meta = json.loads((WORK / "meta.json").read_text())
    ids, pay, exact = meta["ids"], meta["payloads"], meta["exact_top10"]
    tmp = Path(tempfile.mkdtemp(prefix=f"sb_{store}_"))
    res: dict = {"store": store}
    allowed = {i for i, p in zip(ids, pay) if p["approval_status"] in FILTER["approval_status"]}
    if store == "faiss_flat":
        import faiss
        t = time.perf_counter(); ix = faiss.IndexFlatIP(v.shape[1]); ix.add(v); res["build_s"] = time.perf_counter() - t
        search = lambda qq, k, f=False: [[(ids[j], float(s)) for s, j in zip(sr, jr) if j >= 0 and (not f or ids[j] in allowed)][:k] for sr, jr in zip(*ix.search(qq, len(ids) if f else k))]
        faiss.write_index(ix, str(tmp / "index.faiss")); (tmp / "meta.json").write_text(json.dumps({"ids": ids, "payloads": pay}))
        t = time.perf_counter(); ix2 = faiss.read_index(str(tmp / "index.faiss")); json.loads((tmp / "meta.json").read_text()); ix2.search(q[:1], 10); res["reopen_s"] = time.perf_counter() - t
    elif store in ("chroma", "chroma_default"):
        import gridsense.settings  # noqa: F401
        import chromadb
        from gridsense.retrieval.stores import CHROMA_HNSW
        t = time.perf_counter(); cl = chromadb.PersistentClient(path=str(tmp / "chroma")); col = cl.create_collection("chunks", configuration={"hnsw": {"space": "ip", **CHROMA_HNSW[store]}})
        for s in range(0, len(ids), 500):
            col.add(ids=ids[s:s + 500], embeddings=v[s:s + 500].tolist(), metadatas=[{k: str(x) for k, x in p.items()} for p in pay[s:s + 500]])
        res["build_s"] = time.perf_counter() - t
        def search(qq, k, f=False):
            r = col.query(query_embeddings=qq.tolist(), n_results=k, where={"approval_status": {"$in": FILTER["approval_status"]}} if f else None)
            return [[(i, 1.0 - d) for i, d in zip(a, b)] for a, b in zip(r["ids"], r["distances"])]
        del cl
        t = time.perf_counter(); cl2 = chromadb.PersistentClient(path=str(tmp / "chroma")); c2 = cl2.get_collection("chunks"); c2.query(query_embeddings=q[:1].tolist(), n_results=10); res["reopen_s"] = time.perf_counter() - t
    elif store == "qdrant_local":
        import uuid
        from qdrant_client import QdrantClient, models
        t = time.perf_counter(); cl = QdrantClient(path=str(tmp / "qdrant")); cl.create_collection("c", vectors_config=models.VectorParams(size=v.shape[1], distance=models.Distance.DOT))
        for s in range(0, len(ids), 500):
            cl.upsert("c", points=[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)), vector=x.tolist(), payload={**p, "_id": i}) for i, x, p in zip(ids[s:s + 500], v[s:s + 500], pay[s:s + 500])])
        res["build_s"] = time.perf_counter() - t
        def search(qq, k, f=False):
            flt = models.Filter(must=[models.FieldCondition(key="approval_status", match=models.MatchAny(any=FILTER["approval_status"]))]) if f else None
            return [[(p.payload["_id"], p.score) for p in cl.query_points("c", query=row.tolist(), limit=k, query_filter=flt).points] for row in qq]
        cl.close()
        t = time.perf_counter(); cl3 = QdrantClient(path=str(tmp / "qdrant")); cl3.query_points("c", query=q[0].tolist(), limit=10); res["reopen_s"] = time.perf_counter() - t; cl3.close()
        cl = QdrantClient(path=str(tmp / "qdrant"))
    else:
        raise ValueError(store)
    lat, tops = [], None
    for rep in range(5):
        for i in range(len(q)):
            t = time.perf_counter(); r = search(q[i:i + 1], 10); lat.append((time.perf_counter() - t) * 1000)
            if rep == 0:
                tops = (tops or []) + [[x for x, _ in r[0]]]
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    res["parity_top10_identical"] = float(np.mean([a == b for a, b in zip(tops, exact)]))
    res["overlap_at_10"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(tops, exact)]))
    fr = search(q[:5], 10, True)
    res["filter_ok"] = all(all(i in allowed for i, _ in row) and len(row) > 0 for row in fr)
    res["disk_mb"] = _size(tmp) / 1e6
    # VmHWM = peak resident set of THIS process image. ru_maxrss would be wrong here: a child inherits the parent's high-water mark through fork/exec.
    res["peak_rss_mb"] = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 1024
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def end_to_end() -> dict:
    """The real hybrid retriever, once per store, on the dev golden queries: are the retrieved top-10 lists and the precision / recall / MRR the same as with the exact reference?"""
    from dataclasses import replace
    import gridsense.settings  # noqa: F401
    from gridsense.config import load_config
    from gridsense.eval import golden as G
    from gridsense.eval.layer1 import embedder_from_config, evaluate, retrieval_config
    from gridsense.eval.sweep import load_winners
    from gridsense.ingest.corpus import load_corpus
    cfg = load_config()
    state, docs = load_winners()["state"], load_corpus()
    rows = [r for r in G.load(ROOT / cfg["golden_sets"]["retrieval"]["path"], statuses=("reviewed",)) if r["split"] == "dev"]
    emb, minilm = embedder_from_config(state["embedder"]), embedder_from_config("minilm")
    runs = {}
    for st in ["faiss_flat", "chroma_default", "chroma", "qdrant_local"]:
        rc = replace(retrieval_config({}), **{**state["retrieval"], "store": st})
        out = evaluate(rows, docs, rc, emb, cfg["repeats"]["bootstrap_resamples"], chunk_embedder=minilm, rebuild=True)
        d = out["metrics"]["dev"]
        runs[st] = {"lists": [[x["chunk_id"] for x in it["retrieved"]] for it in out["items"]],
                    "metrics": {k: d[k]["mean"] for k in ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@5", "precision@1" if "precision@1" in d else "precision@3", "precision@3", "precision@5", "context_coverage@5")}}
    ref = runs["faiss_flat"]
    res = {}
    for st, r in runs.items():
        res[st] = {"metrics": r["metrics"], "top10_lists_identical_to_reference": float(np.mean([a == b for a, b in zip(r["lists"], ref["lists"])])),
                   "max_abs_metric_diff": max(abs(r["metrics"][k] - ref["metrics"][k]) for k in ref["metrics"])}
        print(f"[vector_store] end-to-end {st}: MRR@10 {r['metrics']['mrr@10']:.3f} R@1 {r['metrics']['recall@1']:.2f} P@5 {r['metrics']['precision@5']:.3f}; lists identical {res[st]['top10_lists_identical_to_reference']:.0%}", flush=True)
    return res


def run() -> dict:
    info = prepare()
    results = []
    for s in ["faiss_flat", "chroma_default", "chroma", "qdrant_local"]:
        p = subprocess.run([sys.executable, "-m", "gridsense.eval.store_bench", "one", "--store", s], capture_output=True, text=True,
                           env={**__import__("os").environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode or not line:
            raise SystemExit(f"{s} failed:\n{p.stderr[-800:]}")
        results.append(json.loads(line[-1]))
        print(f"[vector_store] {s}: p95 {results[-1]['latency_ms']['p95']:.2f} ms, parity {results[-1]['parity_top10_identical']:.2f}, build {results[-1]['build_s']:.2f} s, rss {results[-1]['peak_rss_mb']:.0f} MB", flush=True)
    e2e = end_to_end()
    # rule v2.1 (author, 2026-11-02: a real vector store, not FAISS; precision must be good): eligible = not the library reference, native metadata filter works,
    # dense top-10 identical to exact search, and the end-to-end retrieved lists and metrics identical to the reference
    eligible = [r for r in results if r["store"] != REFERENCE and r["parity_top10_identical"] == 1.0 and r["filter_ok"]
                and e2e[r["store"]]["top10_lists_identical_to_reference"] == 1.0 and e2e[r["store"]]["max_abs_metric_diff"] < 1e-12]
    best_p95 = min(r["latency_ms"]["p95"] for r in eligible)
    tied = [r["store"] for r in eligible if r["latency_ms"]["p95"] <= 1.10 * best_p95]
    winner = min(tied, key=ORDER.index)
    return {"info": info, "results": results, "end_to_end": e2e, "eligible": [r["store"] for r in eligible], "tied": tied, "winner": winner}


def write(out: dict) -> Path:
    from datetime import datetime, timezone
    from gridsense.eval.sweep import SWEEPS, WINNERS, load_winners
    r, e = out["results"], out["end_to_end"]
    L = ["# Sweep — vector_store", "", f"Written for: the author. Generated by `gridsense.eval.store_bench` on {datetime.now(timezone.utc):%Y-%m-%d}. Final index of the sweep winners: {out['info']['n_vectors']} vectors of {out['info']['dim']} dimensions, "
         f"{out['info']['n_queries']} dev queries, CPU, 2 threads. **Author instruction (2026-11-02): use a real vector store, not FAISS, and keep the precision good.** FAISS-flat stays only as the exact "
         "reference for the parity checks (it is a library: no native metadata filter, no payload storage). Chroma and Qdrant are tested; \"chroma\" has high-recall HNSW settings "
         "(ef_search 200, ef_construction 200, M 32), \"chroma_default\" has Chroma's defaults (100 / 100 / 16).", "",
         "## 1. Precision: the real hybrid retriever with each store (dev, answerable queries)", "",
         "Precision@k cannot exceed (golden contexts of the query) / k: with 1 or 2 golden contexts per query, P@5 is at most 0.2 to 0.4. Read P@1 (= Recall@1), Recall@k and MRR first.", "",
         "| store | top-10 lists identical to FAISS-flat (all dev queries) | P@1 / R@1 | P@3 | P@5 | R@5 | MRR@10 | nDCG@5 | max metric difference |", "|---|---|---|---|---|---|---|---|---|"]
    for st in ["faiss_flat", "chroma_default", "chroma", "qdrant_local"]:
        m = e[st]["metrics"]
        L.append(f"| {st}{' (reference)' if st == REFERENCE else ''} | {e[st]['top10_lists_identical_to_reference']:.0%} | {m['recall@1']:.2f} | {m['precision@3']:.3f} | {m['precision@5']:.3f} | {m['recall@5']:.2f} | {m['mrr@10']:.3f} | {m['ndcg@5']:.3f} | {e[st]['max_abs_metric_diff']:.1e} |")
    L += ["", "## 2. Operations", "", "| store | dense top-10 identical to exact | build s | latency p50 / p95 ms | peak RSS MB (fresh process) | disk MB | re-open s | native metadata filter |", "|---|---|---|---|---|---|---|---|"]
    for x in r:
        L.append(f"| {x['store']} | {x['parity_top10_identical']:.0%} | {x['build_s']:.2f} | {x['latency_ms']['p50']:.2f} / {x['latency_ms']['p95']:.2f} | {x['peak_rss_mb']:.0f} | {x['disk_mb']:.2f} | {x['reopen_s']:.2f} | {'works' if x['filter_ok'] else 'FAILED'} |")
    L += ["", "Limits of this test (stated, not hidden): the corpus has only a few hundred vectors, so Chroma's HNSW (ef_search 100 or 200, above the corpus size) is close to exact search and cannot show an "
          "approximation loss. Re-run this benchmark when the corpus grows. Qdrant local mode always searches exactly; its approximate HNSW mode needs a Qdrant server, which was not run. Native hybrid search "
          "was not tested (the BM25 half of the hybrid retriever lives in this pipeline, outside the store).", "",
          f"**Rule (best + cheap, author 2026-11-02):** eligible = a real vector store (not the FAISS reference), native metadata filter works, dense top-10 identical to exact search, and the retrieved lists and all "
          f"metrics identical to the reference end to end → {', '.join(out['eligible'])}. Quality is equal for all eligible stores, so the cheapest wins: lowest p95 latency; within 10 % of it is a tie and the order "
          f"(qdrant_local, chroma, chroma_default) breaks it → {', '.join(out['tied'])}. **Winner: `{out['winner']}`.** The pipeline supports every store above with one setting (`vector_store.backend`); "
          "all eligible stores give the same ranking here, so switching to another one does not change a retrieval result.", ""]
    p = SWEEPS / "vector_store.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (SWEEPS / "vector_store.json").write_text(json.dumps(out, indent=1))
    win = load_winners()
    win["state"]["retrieval"]["store"] = out["winner"]
    win["axes"]["vector_store"] = {"winner": out["winner"], "eligible": out["eligible"], "tied": out["tied"], "rule": "v2.1 (author: real vector store, not FAISS; precision identical to exact)",
                                   "report": str(p.relative_to(ROOT)), "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    WINNERS.write_text(json.dumps(win, indent=1, ensure_ascii=False))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    o = sub.add_parser("one"); o.add_argument("--store", required=True, choices=["faiss_flat", "chroma", "chroma_default", "qdrant_local"])
    ns = ap.parse_args()
    if ns.cmd == "one":
        print(json.dumps(bench_one(ns.store)))
    else:
        out = run()
        print(write(out), "winner:", out["winner"])


if __name__ == "__main__":
    main()
