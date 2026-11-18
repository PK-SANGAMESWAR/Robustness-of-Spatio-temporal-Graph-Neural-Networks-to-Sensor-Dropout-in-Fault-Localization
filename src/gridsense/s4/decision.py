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


# Author decision 2026-11-18 (E + A): the system owns the human-review statement and the fail-closed rule; the generator text never decides them.
HUMAN_REVIEW_STATEMENT = "A person must review this candidate before any action. This statement is set by the system."


def assemble(rep: dict, final_status: str) -> tuple[dict, dict]:
    """(assembled report, info). E: every report gets the fixed human-review field. A: when the final evidence status is insufficient, the generator's
    `recommended_operator_review` is emptied (no procedural statement). The raw generator list stays in `info`, so the behaviour is counted, not hidden."""
    raw = list(rep.get("recommended_operator_review") or [])
    out = dict(rep)
    out["human_review_required"] = HUMAN_REVIEW_STATEMENT
    stripped = final_status == "insufficient" and bool(raw)
    overridden = final_status == "insufficient" and rep.get("evidence_status") != "insufficient"      # a tau-only flag: the generator said sufficient
    if final_status == "insufficient":
        out["recommended_operator_review"] = []
        out["evidence_status"] = "insufficient"
        if overridden:
            out["evidence_status_reason"] = "Set by the system: the retrieval score is below the evidence threshold. " + str(rep.get("evidence_status_reason") or "")
    return out, {"generator_review_while_insufficient": stripped, "generator_review_items_removed": len(raw) if stripped else 0, "evidence_status_overridden_by_system": overridden}
