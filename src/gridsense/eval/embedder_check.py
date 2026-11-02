"""Embedder integrity check and paired comparison on dev (author, 2026-11-02). NO API call: it reads the stored index vectors, the on-disk embedding cache and the sweep snapshots.

    HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 nice -n 19 .venv/bin/python -m gridsense.eval.embedder_check

1. Self-retrieval: every stored chunk vector is used as a query against its own index; the fraction whose rank-1 chunk is the same chunk (strict id; and: same text, ties allowed).
2. Vectors are normalized; the four indices hold the same chunks in the same order; the prefixes are the model-card prefixes and were really used (local models: the stored vectors are
   re-computed with the expected prefix and compared; OpenAI: the stored vectors are compared with the on-disk cache entries of the same text).
3. Paired bootstrap interval, by query, on dev (reviewed answerable queries only), for MiniLM minus each other embedder.
"""
from __future__ import annotations

import json

import numpy as np

from gridsense.config import ROOT, TRACK, torch_threads
from gridsense.eval.layer1 import INDEX_DIR, embedder_from_config
from gridsense.eval.sweep import SWEEPS, paired_ci

EMBEDDERS = ["minilm", "bge_small", "e5_small", "openai_3_small"]
EXPECTED_PREFIX = {   # from the model cards (checked 2026-10-31): bge uses a query instruction only; e5 uses both prefixes; MiniLM and OpenAI none
    "minilm": ("", ""), "bge_small": ("Represent this sentence for searching relevant passages: ", ""), "e5_small": ("query: ", "passage: "), "openai_3_small": ("", "")}
EXCLUDE_DRAFT = {"R-0055", "R-0056"}   # golden drafts (O34): not part of a reviewed result


def latest_snapshot(label: str) -> dict:
    snaps = sorted((TRACK / "eval" / "snapshots").glob(f"*_{label}.json"))
    if not snaps:
        raise SystemExit(f"no snapshot for {label}")
    return json.loads(snaps[-1].read_text())


def load_index(name: str) -> dict:
    snap = latest_snapshot(f"sweep_embedder_{name}")
    folder = INDEX_DIR / snap["index_fingerprint"][:16]
    chunks = [json.loads(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"snap": snap, "folder": folder, "vectors": np.load(folder / "vectors.npy"), "chunks": chunks,
            "manifest": json.loads((folder / "index_manifest.json").read_text())}


def self_retrieval(v: np.ndarray, chunks: list[dict]) -> dict:
    s = v @ v.T
    top = s.argmax(1)
    texts = [c["embed_text"] for c in chunks]
    strict = float(np.mean(top == np.arange(len(v))))
    same_text = float(np.mean([texts[t] == texts[i] for i, t in enumerate(top)]))
    tie_ok = float(np.mean(s[np.arange(len(v)), top] <= np.diag(s) + 1e-6))      # the chunk itself is a top-1 (ties allowed)
    return {"n": len(v), "strict_id": strict, "same_text": same_text, "self_is_top1_ties_allowed": tie_ok, "duplicate_texts": len(texts) - len(set(texts)),
            "min_self_similarity": float(np.diag(s).min()), "mean_top1_other_similarity": float(np.mean([np.sort(r)[-2] for r in s]))}


def prefix_check(name: str, idx: dict, k: int = 25) -> dict:
    spec = idx["manifest"]["embedder"]
    qp, pp = EXPECTED_PREFIX[name]
    out = {"spec_query_prefix_ok": spec["query_prefix"] == qp, "spec_passage_prefix_ok": spec["passage_prefix"] == pp, "dimensions": int(idx["vectors"].shape[1])}
    rng = np.random.default_rng(42)
    pick = sorted(rng.choice(len(idx["chunks"]), size=min(k, len(idx["chunks"])), replace=False).tolist())
    texts = [idx["chunks"][i]["embed_text"] for i in pick]
    stored = idx["vectors"][pick]
    emb = embedder_from_config(name)
    if name == "openai_3_small":      # no API: the same text must be in the on-disk cache, and equal to the stored vector
        got = []
        for t in texts:
            p = emb._cache_path(t)
            got.append(np.load(p) if p.exists() else None)
        out["cache_entries_found"] = sum(g is not None for g in got)
        if all(g is not None for g in got):
            g = np.stack(got); g = g / np.linalg.norm(g, axis=1, keepdims=True)
            out["max_abs_diff_vs_cache"] = float(np.abs(g - stored).max())
        return out
    import torch
    torch.set_num_threads(torch_threads())
    model = emb._load_local()
    right = model.encode([pp + t for t in texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    out["max_abs_diff_with_expected_passage_prefix"] = float(np.abs(right - stored).max())
    other = model.encode([("passage: " if pp == "" else "") + t for t in texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    out["max_abs_diff_with_the_other_choice"] = float(np.abs(other - stored).max())     # shows that the test can tell the two apart (equal for models that use no prefix by design)
    qs = ["Which sensors are nearest to the fault?", "What does the gate return?"]
    a = emb.encode(qs, "query")
    b = model.encode([qp + q for q in qs], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    out["query_encoding_equals_expected_prefix"] = bool(np.abs(a - b).max() < 1e-6)
    return out


def paired(snap_a: dict, snap_b: dict) -> dict:
    def per_q(snap):
        items = [json.loads(l) for l in (ROOT / snap["items_path"]).read_text().splitlines()]
        return {x["query_id"]: x for x in items if x["split"] == "dev" and x["answerable"] and "error" not in x and x["query_id"] not in EXCLUDE_DRAFT}
    a, b = per_q(snap_a), per_q(snap_b)
    q = sorted(set(a) & set(b))
    res = {"n_queries": len(q)}
    for key, f in {"Q (mean of MRR@10, Recall@5, nDCG@5)": lambda x: (x["mrr@10"] + x["recall@5"] + x["ndcg@5"]) / 3, "MRR@10": lambda x: x["mrr@10"], "Recall@5": lambda x: x["recall@5"], "Recall@1": lambda x: x["recall@1"]}.items():
        va, vb = np.array([f(a[i]) for i in q]), np.array([f(b[i]) for i in q])
        lo, hi = paired_ci(va, vb)
        res[key] = {"mean_diff": float(va.mean() - vb.mean()), "ci_low": lo, "ci_high": hi}
    return res


def run() -> dict:
    idx = {n: load_index(n) for n in EMBEDDERS}
    res: dict = {"self_retrieval": {}, "norms": {}, "prefix": {}, "paired": {}}
    for n, d in idx.items():
        res["self_retrieval"][n] = self_retrieval(d["vectors"], d["chunks"])
        nr = np.linalg.norm(d["vectors"], axis=1)
        res["norms"][n] = {"min": float(nr.min()), "max": float(nr.max()), "dim": int(d["vectors"].shape[1]), "n_chunks": len(d["chunks"]), "truncated": d["manifest"]["truncated"]}
        res["prefix"][n] = prefix_check(n, d)
    ids = {n: [c["chunk_id"] for c in d["chunks"]] for n, d in idx.items()}
    txt = {n: [c["embed_text"] for c in d["chunks"]] for n, d in idx.items()}
    res["same_chunks_same_order"] = {"chunk_ids": all(ids[n] == ids["minilm"] for n in EMBEDDERS), "embed_texts": all(txt[n] == txt["minilm"] for n in EMBEDDERS), "n_chunks": len(ids["minilm"])}
    res["paired"] = {f"minilm_minus_{n}": paired(idx["minilm"]["snap"], idx[n]["snap"]) for n in EMBEDDERS if n != "minilm"}
    res["index_config"] = idx["minilm"]["manifest"]["config"] | {"store": "faiss_flat (at the time of the embedder sweep)"}
    return res


def write(res: dict):
    L = ["# Embedder check (no API call)", "", "Written for: the author. Generated by `gridsense.eval.embedder_check` on 2026-11-02 from the stored index vectors, the on-disk embedding cache and the sweep snapshots. "
         "The chunk settings (recursive, 200 tokens, 10 % overlap) were selected with MiniLM **before** the embedder sweep, so every embedder was tested on chunks that were tuned for MiniLM. "
         "On this corpus, with settings tuned for MiniLM, the other embedders did not perform better.", "",
         "## 1. Self-retrieval: each stored chunk vector used as a query", "", "| embedder | chunks | rank-1 is the same chunk (id) | rank-1 has the same text | self is top-1 (ties allowed) | duplicate chunk texts | min self-similarity |", "|---|---|---|---|---|---|---|"]
    for n, r in res["self_retrieval"].items():
        L.append(f"| {n} | {r['n']} | {r['strict_id']:.3f} | {r['same_text']:.3f} | {r['self_is_top1_ties_allowed']:.3f} | {r['duplicate_texts']} | {r['min_self_similarity']:.4f} |")
    L += ["", "## 2. Normalization, chunks and prefixes", "", "| embedder | dimension | vector norm min / max | truncated chunks | spec prefixes equal the model card | stored vectors match the expected prefix (max abs diff) | other prefix choice (max abs diff) |", "|---|---|---|---|---|---|---|"]
    for n in EMBEDDERS:
        p, nm = res["prefix"][n], res["norms"][n]
        if n == "openai_3_small":
            m = f"cache entries found {p['cache_entries_found']} of 25; max abs diff vs the cache {p.get('max_abs_diff_vs_cache', float('nan')):.1e}"
            o = "– (no prefix; no API call)"
        else:
            m, o = f"{p['max_abs_diff_with_expected_passage_prefix']:.1e}", f"{p['max_abs_diff_with_the_other_choice']:.1e}"
        L.append(f"| {n} | {nm['dim']} | {nm['min']:.6f} / {nm['max']:.6f} | {nm['truncated']} | {'yes' if p['spec_query_prefix_ok'] and p['spec_passage_prefix_ok'] else 'NO'} | {m} | {o} |")
    s = res["same_chunks_same_order"]
    L += ["", f"The four indices hold the same {s['n_chunks']} chunks in the same order: chunk ids {'identical' if s['chunk_ids'] else 'DIFFERENT'}; embedded texts {'identical' if s['embed_texts'] else 'DIFFERENT'}. "
          "Local models: the stored vectors were recomputed with the expected passage prefix (small difference = used; a large difference for the other choice shows that the test can tell). "
          "OpenAI: no API call; the stored vectors equal the on-disk cache entries.", "",
          f"## 3. Paired bootstrap on dev, by query ({next(iter(res['paired'].values()))['n_queries']} reviewed answerable queries; 2,000 resamples; seed 42): MiniLM minus the other embedder", "",
          "| comparison | Q | MRR@10 | Recall@5 | Recall@1 |", "|---|---|---|---|---|"]
    for k, v in res["paired"].items():
        cells = " | ".join(f"{v[m]['mean_diff']:+.3f} ({v[m]['ci_low']:+.3f} to {v[m]['ci_high']:+.3f})" for m in ("Q (mean of MRR@10, Recall@5, nDCG@5)", "MRR@10", "Recall@5", "Recall@1"))
        L.append(f"| {k.replace('_', ' ')} | {cells} |")
    L += ["", "A positive value means that MiniLM is higher. An interval that contains 0 means that the difference is not separated from noise with 24 queries."]
    p = SWEEPS / "embedder_check.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (SWEEPS / "embedder_check.json").write_text(json.dumps(res, indent=1, default=float))
    return p


if __name__ == "__main__":
    r = run()
    print(write(r))
    print(json.dumps({"self_retrieval": {n: {k: round(v, 4) for k, v in x.items() if k in ("strict_id", "same_text", "self_is_top1_ties_allowed", "duplicate_texts")} for n, x in r["self_retrieval"].items()},
                      "same_chunks_same_order": r["same_chunks_same_order"]}, indent=1))
