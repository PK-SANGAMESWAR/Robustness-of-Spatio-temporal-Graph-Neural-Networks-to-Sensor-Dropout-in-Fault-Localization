"""Layer 1 runner: one retrieval configuration on the retrieval golden set, deterministic metrics only (free; CPU).

    PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.eval.layer1 --label smoke_baseline --preliminary
    ... --set chunker=recursive --set chunk_size=128 --embedder bge_small      (one variable at a time; rag-component-sweep)

Settings come from configs/eval.yaml (`chunking`, `embedding`, `vector_store`, `retrieval`, `reranker`); --set overrides one
RetrievalConfig field. Draft golden items need --preliminary: the snapshot is then marked PRELIMINARY and is never a reported
result (claude-rag.md § 4 rule 5). tau (sufficiency) is fitted on dev and applied to test. A query that raises is recorded
as a failure with its cause, never as a score. The OpenAI embedder is refused here unless --allow-paid is given.
"""
from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from dataclasses import asdict, fields

import numpy as np

from gridsense.cases.store import StaleIndexError
from gridsense.config import TRACK, load_config
from gridsense.eval import golden as G
from gridsense.eval import retrieval_metrics as RM
from gridsense.eval.snapshot import write_snapshot
from gridsense.ingest.corpus import load_corpus
from gridsense.retrieval.embeddings import Embedder, EmbedderSpec
from gridsense.retrieval.retriever import DocIndex, RetrievalConfig, Retriever

INDEX_DIR = TRACK / "eval" / "doc_index"


def embedder_from_config(name: str | None = None) -> Embedder:
    e = load_config()["embedding"]
    c = e if not name or name == "baseline" else e["candidates"][name]
    spec = EmbedderSpec(model=c["model"], backend=c.get("backend", "local"), revision=c.get("revision"),
                        query_prefix=c.get("query_prefix", ""), passage_prefix=c.get("passage_prefix", ""),
                        max_seq_length=c.get("max_seq_length") if c.get("backend") == "openai" else None,
                        dimensions=c.get("dimensions"))
    return Embedder(spec)


def retrieval_config(overrides: dict) -> RetrievalConfig:
    cfg = load_config()
    base = RetrievalConfig(chunker=cfg["chunking"]["strategy"], chunk_size=cfg["chunking"]["chunk_size_tokens"],
                           overlap=cfg["chunking"]["overlap_fraction"], parent_size=cfg["chunking"].get("parent_size_tokens"),
                           store=cfg["vector_store"]["backend"], mode=cfg["retrieval"]["mode"], top_k=cfg["retrieval"]["top_k"],
                           dense_candidates=cfg["retrieval"]["dense_candidates"], lexical_candidates=cfg["retrieval"]["lexical_candidates"],
                           rrf_k=cfg["retrieval"]["rrf_k"], reranker=cfg["reranker"]["model"] if cfg["reranker"]["enabled"] else None,
                           reranker_revision=cfg["reranker"].get("revision"), rerank_top=cfg["reranker"]["rerank_top"],
                           filters=cfg["retrieval"]["filters"])
    types = {f.name: f.type for f in fields(RetrievalConfig)}
    for k, v in overrides.items():
        if k not in types:
            raise KeyError(f"unknown RetrievalConfig field {k}")
        setattr(base, k, None if v == "none" else (json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false") else v))
    return base


def get_index(rc: RetrievalConfig, emb: Embedder, docs) -> DocIndex:
    from gridsense.retrieval.retriever import index_fingerprint
    folder = INDEX_DIR / index_fingerprint(rc, emb, docs)[:16]
    if (folder / "index_manifest.json").exists():
        try:
            return DocIndex.load(folder, rc, emb, docs)
        except StaleIndexError:
            raise
    ix = DocIndex.build(rc, emb, docs)
    ix.save(folder)
    return ix


def run(label: str, overrides: dict, embedder_name: str | None, preliminary: bool, allow_paid: bool = False) -> dict:
    cfg = load_config()
    rows = G.load(TRACK.parent / cfg["golden_sets"]["retrieval"]["path"])
    drafts = sum(r["status"] == "draft" for r in rows)
    if drafts and not preliminary:
        raise SystemExit(f"{drafts} golden items are drafts: run with --preliminary (result is never reported) or review them first")
    docs = load_corpus()
    problems = G.validate(rows, docs)
    if problems:
        raise SystemExit("golden set invalid:\n" + "\n".join(problems))
    emb, rc = embedder_from_config(embedder_name), retrieval_config(overrides)
    if emb.spec.backend == "openai" and not allow_paid:
        raise SystemExit("the OpenAI embedder is paid: run rag-cost-check, get approval, then pass --allow-paid")
    ix = get_index(rc, emb, docs)
    ret, items = Retriever(ix), []
    for r in rows:
        rec = {k: r[k] for k in ("query_id", "query_type", "split", "answerable")}
        try:
            res = ret.retrieve(r["query"], top_k=max(RM.KS))
            rec.update(dense_top1=res["dense_top1"], latency_ms=res["latency_ms"],
                       retrieved=[{k: p[k] for k in ("rank", "document_id", "section_path", "chunk_id", "score")} for p in res["passages"]])
            if r["answerable"]:
                rec.update(RM.query_metrics(res["passages"], r))
        except Exception as e:  # recorded as a failure, never as a score
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=3)
        items.append(rec)
    ok = [x for x in items if "error" not in x]
    dev = [x for x in ok if x["split"] == "dev"]
    suff = RM.choose_sufficiency_threshold([x["dense_top1"] for x in dev], [x["answerable"] for x in dev])
    n_boot = cfg["repeats"]["bootstrap_resamples"]
    keys = sorted({k for x in ok for k in x if "@" in k})
    metrics: dict = {"sufficiency": {"tau_dev": suff}}
    for split in ("all", "dev", "test"):
        sel = [x for x in ok if split == "all" or x["split"] == split]
        ans = [x for x in sel if x["answerable"]]
        metrics[split] = {k: RM.aggregate(ans, k, n_boot) for k in keys}
        metrics[split]["sufficiency"] = RM.sufficiency_metrics([x["dense_top1"] for x in sel], [x["answerable"] for x in sel], suff["tau"])
    by_type = defaultdict(list)
    for x in ok:
        if x["answerable"]:
            by_type[x["query_type"]].append(x)
    metrics["by_type"] = {t: {k: RM.aggregate(v, k, n_boot)["mean"] for k in ("recall@1", "recall@5", "mrr@10")} for t, v in by_type.items()}
    lat = [x["latency_ms"] for x in ok]
    metrics["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    metrics["index"] = {"n_chunks": len(ix.chunks), "truncated": ix.truncated, "max_embed_tokens": ix.max_embed_tokens,
                        "window": emb.window, "build_seconds": ix.build_seconds}
    snap = write_snapshot(label, metrics, items, config_effective={"retrieval": asdict(rc), "embedder": emb.spec.key()},
                          index_fingerprint=ix.fingerprint, golden_versions={"retrieval": 0, "drafts": drafts, "items": len(rows)},
                          preliminary=bool(drafts), preliminary_reason="golden items are drafts (O29)" if drafts else None)
    return {"snapshot": str(snap), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--set", action="append", default=[], help="RetrievalConfig field=value")
    ap.add_argument("--embedder", default=None, help="key in configs/eval.yaml embedding.candidates (default: baseline)")
    ap.add_argument("--preliminary", action="store_true")
    ap.add_argument("--allow-paid", action="store_true")
    a = ap.parse_args()
    out = run(a.label, dict(s.split("=", 1) for s in a.set), a.embedder, a.preliminary, a.allow_paid)
    m = out["metrics"]
    print(json.dumps({"snapshot": out["snapshot"], "all": {k: round(m["all"][k]["mean"], 3) for k in ("recall@1", "recall@5", "mrr@10", "ndcg@5")},
                      "sufficiency": m["sufficiency"], "test_sufficiency": m["test"]["sufficiency"], "latency_ms": m["latency_ms"],
                      "index": m["index"]}, indent=1))


if __name__ == "__main__":
    main()
