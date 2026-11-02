"""Layer 1 runner: one retrieval configuration on the retrieval golden set, deterministic metrics only (free; CPU).

    PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.eval.layer1 --label smoke_baseline
    ... --set chunker=recursive --set chunk_size=128 --embedder bge_small      (one variable at a time; rag-component-sweep)

Settings come from configs/eval.yaml (`chunking`, `embedding`, `vector_store`, `retrieval`, `reranker`); --set overrides one RetrievalConfig field.
`evaluate()` is the pure part that the sweeps (gridsense.eval.sweep) call. Draft golden items need --preliminary (claude-rag.md § 4 rule 5).
tau (no-answer threshold) is fitted on `dev` only and applied to `test`. A query that raises is recorded as a failure with its cause, never as a
score. The OpenAI embedder is refused unless --allow-paid is given; it runs through the cost ledger (gridsense.cost).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, fields

import numpy as np

from gridsense.cases.store import StaleIndexError
from gridsense.config import ROOT, TRACK, load_config
from gridsense.eval import golden as G
from gridsense.eval import retrieval_metrics as RM
from gridsense.eval.snapshot import write_snapshot
from gridsense.ingest.corpus import load_corpus
from gridsense.retrieval.embeddings import Embedder, EmbedderSpec, get_meter
from gridsense.retrieval.retriever import DocIndex, RetrievalConfig, Retriever, index_fingerprint

INDEX_DIR = TRACK / "eval" / "doc_index"
CTX_METER = "sentence-transformers/all-MiniLM-L6-v2"   # one neutral meter for the size of the returned context
CTX_KS = (3, 5, 8, 10)


def embedder_from_config(name: str | None = None, ledger=None) -> Embedder:
    e = load_config()["embedding"]
    c = e if not name or name == "baseline" else e["candidates"][name]
    spec = EmbedderSpec(model=c["model"], backend=c.get("backend", "local"), revision=c.get("revision"),
                        query_prefix=c.get("query_prefix", ""), passage_prefix=c.get("passage_prefix", ""),
                        max_seq_length=c.get("max_seq_length") if c.get("backend") == "openai" else None,
                        dimensions=c.get("dimensions"))
    return Embedder(spec, ledger=ledger)


def retrieval_config(overrides: dict) -> RetrievalConfig:
    cfg = load_config()
    base = RetrievalConfig(chunker=cfg["chunking"]["strategy"], chunk_size=cfg["chunking"]["chunk_size_tokens"],
                           overlap=cfg["chunking"]["overlap_fraction"], parent_size=cfg["chunking"].get("parent_size_tokens"),
                           store=cfg["vector_store"]["backend"], mode=cfg["retrieval"]["mode"], top_k=cfg["retrieval"]["top_k"],
                           dense_candidates=cfg["retrieval"]["dense_candidates"], lexical_candidates=cfg["retrieval"]["lexical_candidates"],
                           rrf_k=cfg["retrieval"]["rrf_k"], reranker=cfg["reranker"]["model"] if cfg["reranker"]["enabled"] else None,
                           reranker_revision=cfg["reranker"].get("revision"), rerank_top=cfg["reranker"]["rerank_top"],
                           filters=cfg["retrieval"]["filters"], chunk_meter=cfg["chunking"].get("chunk_meter"),
                           table_rows=bool(cfg["chunking"].get("table_rows", False)))
    types = {f.name: f.type for f in fields(RetrievalConfig)}
    for k, v in overrides.items():
        if k not in types:
            raise KeyError(f"unknown RetrievalConfig field {k}")
        setattr(base, k, None if v == "none" else (json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false") else v))
    return base


def get_index(rc: RetrievalConfig, emb: Embedder, docs, chunk_embedder: Embedder | None = None, rebuild: bool = False) -> DocIndex:
    folder = INDEX_DIR / index_fingerprint(rc, emb, docs, chunk_embedder)[:16]
    if (folder / "index_manifest.json").exists() and not rebuild:
        try:
            return DocIndex.load(folder, rc, emb, docs, chunk_embedder)
        except StaleIndexError:
            raise
    ix = DocIndex.build(rc, emb, docs, chunk_embedder)
    ix.save(folder)
    return ix


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return float("nan")


def reachability(ix: DocIndex, rows: list[dict]) -> dict:
    """A golden span is reachable when some chunk's returned text contains it. A span that every chunk splits can never be a hit,
    whatever the retriever does: it caps recall. Per context, so that one split span among several is visible."""
    by_doc: dict[str, list[str]] = defaultdict(list)
    for c in ix.chunks:
        by_doc[c.document_id].append(G.norm(c.text))
    miss, total = [], 0
    for r in rows:
        for c in r["golden_contexts"]:
            total += 1
            if not any(G.norm(c["span"]) in t for t in by_doc[c["document_id"]]):
                miss.append({"query_id": r["query_id"], "document_id": c["document_id"], "span": c["span"][:90]})
    return {"contexts": total, "reachable": total - len(miss), "fraction": (total - len(miss)) / total if total else float("nan"), "split_spans": miss}


def evaluate(rows: list[dict], docs, rc: RetrievalConfig, emb: Embedder, n_boot: int, chunk_embedder: Embedder | None = None, rebuild: bool = False) -> dict:
    """Run one configuration on `rows` (golden items). tau is fitted on the dev rows among them and applied to the test rows (if any)."""
    t0 = time.perf_counter()
    ix = get_index(rc, emb, docs, chunk_embedder, rebuild)
    t_index = time.perf_counter() - t0
    ret, meter, items = Retriever(ix), get_meter(CTX_METER), []
    for r in rows:
        rec = {k: r[k] for k in ("query_id", "query_type", "split", "answerable")}
        try:
            res = ret.retrieve(r["query"], top_k=max(RM.KS))
            rec.update(dense_top1=res["dense_top1"], latency_ms=res["latency_ms"],
                       retrieved=[{k: p[k] for k in ("rank", "document_id", "section_path", "chunk_id", "score")} for p in res["passages"]])
            for k in CTX_KS:
                rec[f"ctx_tokens@{k}"] = sum(meter.count_tokens(p["text"]) for p in res["passages"][:k])
            if r["answerable"]:
                rec.update(RM.query_metrics(res["passages"], r))
        except Exception as e:  # recorded as a failure, never as a score
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=3)
        items.append(rec)
    ok = [x for x in items if "error" not in x]
    dev = [x for x in ok if x["split"] == "dev"]
    suff = RM.choose_sufficiency_threshold([x["dense_top1"] for x in dev], [x["answerable"] for x in dev])
    keys = sorted({k for x in ok for k in x if "@" in k and not k.startswith("ctx_tokens")})
    metrics: dict = {"sufficiency": {"tau_dev": suff}}
    for split in ("all", "dev", "test"):
        sel = [x for x in ok if split == "all" or x["split"] == split]
        if not sel:
            continue
        ans = [x for x in sel if x["answerable"]]
        metrics[split] = {k: RM.aggregate(ans, k, n_boot) for k in keys}
        metrics[split]["sufficiency"] = RM.sufficiency_metrics([x["dense_top1"] for x in sel], [x["answerable"] for x in sel], suff["tau"])
        metrics[split]["ctx_tokens_mean"] = {str(k): float(np.mean([x[f"ctx_tokens@{k}"] for x in sel])) for k in CTX_KS}
    by_type = defaultdict(list)
    for x in ok:
        if x["answerable"]:
            by_type[x["query_type"]].append(x)
    metrics["by_type"] = {t: {k: RM.aggregate(v, k, n_boot)["mean"] for k in ("recall@1", "recall@5", "mrr@10")} for t, v in by_type.items()}
    lat = [x["latency_ms"] for x in ok]
    metrics["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    reach = reachability(ix, [r for r in rows if r["answerable"]])
    metrics["reachability"] = {k: v for k, v in reach.items() if k != "split_spans"}
    metrics["index"] = {"n_chunks": len(ix.chunks), "truncated": ix.truncated, "max_embed_tokens": ix.max_embed_tokens, "window": emb.window,
                        "build_seconds": ix.build_seconds, "index_or_load_seconds": t_index, "rss_mb_after": rss_mb(),
                        "mean_chunk_tokens": float(np.mean([get_meter(CTX_METER).count_tokens(c.text) for c in ix.chunks]))}
    return {"items": items, "metrics": metrics, "ix": ix, "reach": reach}


def run(label: str, overrides: dict, embedder_name: str | None, preliminary: bool, allow_paid: bool = False, split: str | None = None) -> dict:
    cfg = load_config()
    rows = G.load(ROOT / cfg["golden_sets"]["retrieval"]["path"], statuses=("draft", "reviewed") if preliminary else ("reviewed",))
    drafts = sum(r["status"] == "draft" for r in rows)       # drafts are only loaded with --preliminary; a result with drafts is never reported
    docs = load_corpus()
    problems = G.validate(rows, docs)
    if problems:
        raise SystemExit("golden set invalid:\n" + "\n".join(problems))
    emb, rc = embedder_from_config(embedder_name), retrieval_config(overrides)
    if emb.spec.backend == "openai":
        raise SystemExit("author 2026-11-02: no API call and no paid embedder on this corpus again")
    if split:
        rows = [r for r in rows if r["split"] == split]
    out = evaluate(rows, docs, rc, emb, cfg["repeats"]["bootstrap_resamples"])
    draft_docs = sorted(d.document_id for d in docs if d.status == "draft")
    snap = write_snapshot(label, out["metrics"], out["items"], config_effective={"retrieval": asdict(rc), "embedder": emb.spec.key()},
                          index_fingerprint=out["ix"].fingerprint, golden_versions={"retrieval": cfg["golden_sets"]["retrieval"]["version"], "drafts": drafts, "items": len(rows)},
                          preliminary=bool(drafts), preliminary_reason="golden items are drafts" if drafts else None,
                          notes=f"golden set reviewed; {len(draft_docs)} corpus documents still draft: {', '.join(draft_docs)}" if draft_docs else None)
    return {"snapshot": str(snap), "metrics": out["metrics"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--set", action="append", default=[], help="RetrievalConfig field=value")
    ap.add_argument("--embedder", default=None, help="key in configs/eval.yaml embedding.candidates (default: baseline)")
    ap.add_argument("--preliminary", action="store_true")
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    a = ap.parse_args()
    out = run(a.label, dict(s.split("=", 1) for s in a.set), a.embedder, a.preliminary, a.allow_paid, a.split)
    m = out["metrics"]
    first = "all" if "all" in m else next(iter(k for k in m if k in ("dev", "test")))
    print(json.dumps({"snapshot": out["snapshot"], first: {k: round(m[first][k]["mean"], 3) for k in ("recall@1", "recall@5", "mrr@10", "ndcg@5")},
                      "sufficiency": m["sufficiency"], "latency_ms": m["latency_ms"], "index": m["index"], "reachability": m["reachability"]}, indent=1))


if __name__ == "__main__":
    main()
