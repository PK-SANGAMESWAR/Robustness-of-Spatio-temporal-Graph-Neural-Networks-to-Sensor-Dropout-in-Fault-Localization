"""S8: one offline LangSmith experiment with the local evaluators (skill `rag-langsmith-sync`, action `experiment`). The local snapshot is the source of truth; LangSmith is a mirror.

What it does: runs the FROZEN retrieval configuration v2 on the DEV split of the `gridsense-retrieval` dataset (the test split is not run again: it was used once, `reports/layer1/GOLDEN_TEST_USE.md`) and scores each query with the local metrics
(recall@1/3/5, MRR@10, nDCG@5 for answerable queries; the evidence flag for all). What leaves the machine: the query (already in the dataset), and per query the retrieved document ids, section names, chunk ids and scores. NO passage text. Chunks of sensitive sources (unpublished manuscript, `RESULTS.md`, draft SOP) are withheld: only their score is sent.
The golden span text stays local: matching runs in this process. Replay of dev queries, not a live grid. The frozen test numbers are in `reports/sweeps/FROZEN_TEST_v2.md`; these dev numbers are not a result.
`python -m gridsense.eval.langsmith_experiment`
"""
from __future__ import annotations

import json

from gridsense import settings
from gridsense.config import ROOT, load_config
from gridsense.eval import langsmith_package as LP
from gridsense.eval.retrieval_metrics import query_metrics
from gridsense.eval.snapshot import write_snapshot

def public_ref(p: dict) -> dict:
    """What may leave the machine for one retrieved chunk. A chunk of an unpublished or internal source: id and section heading stay local too (E38)."""
    if p["document_id"] in LP.SENSITIVE_DOCS:
        return {"document_id": "[withheld: sensitive source]", "section": "[withheld]", "chunk_id": "[withheld]", "score": round(float(p["score"]), 4)}
    return {"document_id": p["document_id"], "section": " > ".join(p["section_path"][-1:]), "chunk_id": p["chunk_id"], "score": round(float(p["score"]), 4)}


METRICS = ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@5")


def run(split: str = "dev") -> dict:
    cfg = load_config()["langsmith"]
    if not (cfg.get("enabled") and cfg.get("upload_approved") and cfg.get("project")):
        raise PermissionError("LangSmith is not approved in configs/eval.yaml (O8)")
    if split != "dev":
        raise ValueError("only the dev split may be run: the test split was used once per frozen configuration")
    from langsmith import Client, evaluate

    from gridsense.s4 import run as R
    client = Client(api_key=settings.secret("LANGSMITH_API_KEY"), api_url=cfg["endpoint"])
    ds = client.read_dataset(dataset_name=LP.DATASET)
    examples = [e for e in client.list_examples(dataset_id=ds.id) if (e.metadata or {}).get("split") == split]
    local = {r["query_id"]: r for r in LP.load_rows() if r["status"] == "reviewed"}
    retr, tau, rinfo = R.retriever_and_tau()
    stash: dict[str, list[dict]] = {}

    def target(inputs: dict) -> dict:
        out = retr.retrieve(inputs["query"], top_k=10)
        stash[inputs["query"]] = out["passages"]                                   # full text stays in this process
        return {"retrieved": [public_ref(p) for p in out["passages"]], "dense_top1": round(float(out["dense_top1"]), 4)}

    def make(metric: str):
        def ev(run, example):
            item = local[example.metadata["query_id"]]
            if not item["answerable"]:
                return {"key": metric, "score": None, "comment": "unanswerable: not scored"}
            return {"key": metric, "score": float(query_metrics(stash[example.inputs["query"]], item)[metric])}
        ev.__name__ = metric
        return ev

    def evidence_flag(run, example):
        item = local[example.metadata["query_id"]]
        flagged = run.outputs["dense_top1"] < tau
        return {"key": "evidence_flag_correct", "score": float(flagged == (not item["answerable"])), "comment": f"tau {tau:.4f} (dev-fitted)"}

    res = evaluate(target, data=examples, evaluators=[make(m) for m in METRICS] + [evidence_flag], experiment_prefix="gridsense-retrieval-v2-dev", client=client, max_concurrency=1,
                   metadata={"config": "retrieval v2 (frozen)", "split": split, "index_fingerprint": rinfo["index_fingerprint"][:16], "note": "replay of dev queries; ids and scores only, no passage text; not a result"},
                   description="GridSense retrieval, frozen configuration v2, dev split. Local evaluators. Simulated data; replay, not a live grid.")
    rows = list(res)
    agg: dict[str, list[float]] = {}
    for r in rows:
        for fb in r["evaluation_results"]["results"]:
            if fb.score is not None:
                agg.setdefault(fb.key, []).append(float(fb.score))
    mean = {k: {"mean": sum(v) / len(v), "n": len(v)} for k, v in agg.items()}
    url = getattr(res, "url", None) or str(getattr(res, "experiment_name", ""))
    snap = write_snapshot("s8_langsmith_retrieval_dev", {"dev": mean}, [], config_effective={"dataset": LP.DATASET, "split": split, "queries": len(examples), "project": cfg["project"]},
                          index_fingerprint=rinfo["index_fingerprint"], langsmith_experiment_url=url,
                          notes="LangSmith mirror of a local evaluation on the DEV split (retrieval v2). Not the frozen test result. Outputs sent: ids, sections, scores; no passage text.")
    return {"snapshot": str(snap.relative_to(ROOT)), "queries": len(examples), "metrics": mean, "experiment": url}


def main() -> None:
    print(json.dumps(run(), indent=1, default=str))


if __name__ == "__main__":
    main()
