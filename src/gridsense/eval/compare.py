"""Baseline-versus-candidate comparison (rag-eval.md § 9; EVAL.md § 12; skill `rag-compare`). Pure Python, no LLM, no API call.

Outcome: FAIL if the snapshots are not comparable, a gate metric fails, or a floor is missed (baseline floors are reported as "baseline not healthy").
REVIEW if a guardrail regresses by more than its tolerance, the confidence intervals overlap on a regression, a tolerance is `provisional`, a gate has no floor yet, or a gate is not measured.
PASS only if none of the above. A PASS does not make the candidate the new baseline: only the author accepts a baseline (DONE.md).

`python -m gridsense compare <baseline.json> <candidate.json>`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from gridsense.config import ROOT, TRACK

REGISTRY = TRACK / "configs" / "registry.yaml"
OUT = TRACK / "reports" / "compare"
STRATA = ("high_conf_error", "confident_correct", "low_confidence")
# S4 snapshot metric -> registry name. Check pass rates: the minimum over the strata (a gate wants 100 % everywhere). Judge means: the weighted figure "over the covered strata".
S4_CHECKS = {"layer2.field_consistency": "check.field_consistency", "layer2.schema_validity": "check.schema", "layer2.citation_validity": "check.citation_validity"}
S4_JUDGE = {"layer2.faithfulness": "judge.faithfulness", "layer2.answer_relevancy": "judge.answer_relevancy"}


def load_registry(path: Path = REGISTRY) -> dict[str, dict]:
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {m["name"]: m for m in reg["metrics"]}


def registry_values(snap: dict) -> dict[str, float]:
    """The metrics of one snapshot under their registry names. A name that is already a registry name passes through; an S4 name is mapped. Missing = not measured (absent, never 0)."""
    m = snap.get("metrics") or {}
    out = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k.split(".")[0].startswith("layer")}
    for name, key in S4_CHECKS.items():
        vals = [m.get(f"per_stratum.all_k.{s}.metrics.{key}.mean") for s in STRATA]
        if all(v is not None for v in vals):
            out[name] = min(vals)
    for name, key in S4_JUDGE.items():
        v = m.get(f"weighted_over_the_covered_strata.metrics.{key}.value")
        if v is not None:
            out[name] = v
    return out


def comparable(base: dict, cand: dict) -> list[str]:
    """Reasons why the two snapshots cannot be compared (empty = comparable). Golden versions, split, k grid and the measured registry metrics must be equal."""
    why = []
    if (base.get("golden_versions") or {}) != (cand.get("golden_versions") or {}):
        why.append(f"golden versions differ: {base.get('golden_versions')} vs {cand.get('golden_versions')}")
    for key in ("split", "k_levels", "events"):
        a, b = (base.get("config_effective") or {}).get(key), (cand.get("config_effective") or {}).get(key)
        if a is not None and b is not None and a != b:
            why.append(f"config_effective.{key} differs")
    only = set(registry_values(base)) ^ set(registry_values(cand))
    if only:
        why.append(f"measured registry metrics differ: {sorted(only)}")
    return why


def _meets(value: float, floor: float | None, direction: str) -> bool | None:
    if floor is None:
        return None
    return value >= floor if direction == "higher_is_better" else value <= floor


def compare(base_path: str | Path, cand_path: str | Path, registry_path: Path = REGISTRY) -> dict:
    base, cand = json.loads(Path(base_path).read_text(encoding="utf-8")), json.loads(Path(cand_path).read_text(encoding="utf-8"))
    reg = load_registry(registry_path)
    bv, cv = registry_values(base), registry_values(cand)
    rows, notes, outcome = [], [], "PASS"

    def raise_to(level: str) -> None:
        nonlocal outcome
        outcome = max(outcome, level, key=("PASS", "REVIEW", "FAIL").index)

    why = comparable(base, cand)
    if why:
        raise_to("FAIL")
        notes += [f"NOT COMPARABLE: {w}" for w in why]
    for name in sorted(set(bv) | set(cv)):
        spec = reg.get(name)
        if spec is None:
            rows.append({"metric": name, "role": None, "baseline": bv.get(name), "candidate": cv.get(name), "result": "NOT IN REGISTRY", "rule": "ignored"})
            continue
        b, c, role, d = bv.get(name), cv.get(name), spec["role"], spec["direction"]
        row = {"metric": name, "role": role, "baseline": b, "candidate": c, "delta": None if b is None or c is None else c - b, "floor": spec.get("floor"), "tolerance": spec.get("tolerance"),
               "tolerance_source": spec.get("tolerance_source"), "result": "ok", "rule": ""}
        for who, v in (("baseline", b), ("candidate", c)):
            if v is None:
                continue
            m = _meets(v, spec.get("floor"), d)
            if m is False:
                row["rule"] += f"{who} misses the floor; "
                if who == "candidate":
                    row["result"] = "FAIL"
                    raise_to("FAIL")
                else:
                    notes.append(f"BASELINE NOT HEALTHY: {name} = {v} misses the floor {spec.get('floor')}")
                    row["result"] = row["result"] if row["result"] == "FAIL" else "BASELINE NOT HEALTHY"
                    raise_to("FAIL")
        if role == "gate" and c is not None and spec.get("floor") is None:
            row["result"], row["rule"] = ("REVIEW" if row["result"] == "ok" else row["result"]), row["rule"] + "gate without a floor (the author sets it); "
            raise_to("REVIEW")
        if role == "guardrail" and b is not None and c is not None:
            regress = (b - c) if d == "higher_is_better" else (c - b)
            tol = spec.get("tolerance")
            if spec.get("tolerance_source") != "measured" or tol is None:
                row["rule"] += "tolerance is provisional; "
                if row["result"] == "ok":
                    row["result"] = "REVIEW"
                raise_to("REVIEW")
            elif regress > tol:
                row["rule"] += f"regression {regress:.4g} > tolerance {tol:.4g}; "
                if row["result"] == "ok":
                    row["result"] = "REVIEW"
                raise_to("REVIEW")
            blo, bhi, clo, chi = (base["metrics"].get(f"{name}.ci_low"), base["metrics"].get(f"{name}.ci_high"), cand["metrics"].get(f"{name}.ci_low"), cand["metrics"].get(f"{name}.ci_high"))
            if None not in (blo, bhi, clo, chi) and regress > 0 and max(blo, clo) <= min(bhi, chi):
                row["rule"] += "confidence intervals overlap on a regression; "
                if row["result"] == "ok":
                    row["result"] = "REVIEW"
                raise_to("REVIEW")
        row["rule"] = row["rule"].strip("; ") or ("informational: reported only" if role == "informational" else "met")
        rows.append(row)
    for name, spec in reg.items():                       # a gate that neither snapshot measures belongs to another layer: a note, not a verdict. A gate that only one side measures is a FAIL above (metric sets differ).
        if spec["role"] == "gate" and name not in bv and name not in cv:
            notes.append(f"gate not measured in either snapshot (other layer): {name}")
    return {"outcome": outcome, "baseline": str(base_path), "candidate": str(cand_path), "rows": rows, "notes": notes,
            "rule": "A PASS does not make the candidate the new baseline: only the author accepts a baseline."}


def markdown(res: dict) -> str:
    L = [f"# Comparison: {res['outcome']}", "", f"Baseline: `{res['baseline']}`", f"Candidate: `{res['candidate']}`", ""]
    L += [f"- {n}" for n in res["notes"]] + ([""] if res["notes"] else [])
    L += ["| metric | role | baseline | candidate | delta | result | rule |", "|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    L += [f"| {r['metric']} | {r['role']} | {f(r['baseline'])} | {f(r['candidate'])} | {f(r.get('delta'))} | {r['result']} | {r['rule']} |" for r in res["rows"]]
    return "\n".join(L + ["", res["rule"], ""])


def write(res: dict) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(res['baseline']).stem}_vs_{Path(res['candidate']).stem}"
    md, js = OUT / f"{stem}.md", OUT / f"{stem}.json"
    md.write_text(markdown(res), encoding="utf-8")
    js.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    return md, js


def main(argv: list[str] | None = None) -> None:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 2:
        raise SystemExit("usage: python -m gridsense compare <baseline snapshot.json> <candidate snapshot.json>")
    res = compare(a[0], a[1])
    md, js = write(res)
    print(json.dumps({"outcome": res["outcome"], "notes": res["notes"], "markdown": str(md.relative_to(ROOT)), "json": str(js.relative_to(ROOT))}, indent=1))


if __name__ == "__main__":
    main()
