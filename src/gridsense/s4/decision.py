"""The evidence decision (author, 2026-11-02; S4_PLAN.md § 4). Pure Python. The LLM never decides it alone.

insufficient = tau flag OR generator evidence_status != "sufficient" (fail-closed: a missing, malformed or unknown status counts as insufficient).
tau is the dev-fitted threshold on the top-1 dense cosine of the retrieval query (`Retriever.retrieve()["dense_top1"]`); a flag means top-1 < tau.
"""
from __future__ import annotations


def tau_flag(dense_top1: float | None, tau: float | None) -> bool:
    if tau is None or dense_top1 is None:
        return True                                  # no threshold or no score: fail closed
    return float(dense_top1) < float(tau)


def final_evidence_status(flag: bool, generator_status: object) -> dict:
    gen_ok = generator_status == "sufficient"
    final = "insufficient" if (flag or not gen_ok) else "sufficient"
    return {"tau_flag": bool(flag), "generator_status": generator_status if isinstance(generator_status, str) else None, "final": final,
            "by": [n for n, hit in (("tau", flag), ("generator", not gen_ok)) if hit]}
