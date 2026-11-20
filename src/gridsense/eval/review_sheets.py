"""Review aids for the author (no API call, no judge, no label written). They only collect what the author needs on one sheet.

`o13`     the blind labelling sheet for judge validation (O13): 50 generated reports (all 30 main-run reports + 20 seeded noise-run reports) with the event facts, the report text and the
          retrieved passages; the judge scores are NOT on the sheet (they are in a separate file), and the truth is not on it either (the generator and the judge never saw it).
`safety`  the 25 draft safety items with the live result of the three repeats next to each item and an empty verdict column.
`python -m gridsense.eval.review_sheets o13|safety|all`
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from gridsense.config import ROOT, TRACK
from gridsense.eval.snapshot import ITEMS_DIR
from gridsense.s4 import generate as GEN

OUT = TRACK / "reports" / "s4"
SEED = 360131
N_NOISE = 20
RUBRICS = ("prediction_vs_confirmed", "completeness", "adherence", "evidence_use", "no_aggregate_metric_as_confidence")


def latest(pattern: str):
    return sorted(ITEMS_DIR.glob(pattern))[-1]


def _passages(retr, ev_item) -> str:
    from gridsense.s4 import run as R
    out = R.retrieve_all(retr, ev_item, "single")["passages"]
    return "\n\n".join(f"[{p['document_id']} > {' > '.join(p['section_path'][-1:])}]\n{p['text'][:700]}" for p in out)


def o13(retr=None) -> dict:
    from gridsense.s4 import run as R
    ev = R.load_events()
    order = sorted(ev)
    main = latest("*s4_main_live_rescored.jsonl")
    noise = latest("*s4_noise_live_rescored.jsonl")
    rows = [(main, json.loads(x)) for x in main.read_text(encoding="utf-8").splitlines() if x.strip()]
    nz = [(noise, json.loads(x)) for x in noise.read_text(encoding="utf-8").splitlines() if x.strip()]
    pick = np.random.default_rng(SEED).choice(len(nz), size=N_NOISE, replace=False)
    chosen = [r for r in rows if not r[1].get("error")] + [nz[i] for i in sorted(pick) if not nz[i][1].get("error")]
    if retr is None:
        retr = R.retriever_and_tau()[0]
    sheet, scores = [], []
    for n, (path, it) in enumerate(chosen, 1):
        e = ev[it["event_id"]]
        o = e["event_object"]
        ref = f"{path.stem}|{it['anon_id']}|repeat{it.get('repeat', 0)}"            # anonymous: the run ID starts with the true fault bus
        sheet.append({"label_id": f"H-{n:04d}", "report_ref": ref, "k": o["k"], "gate_outcome": o["gate_outcome"], "predicted_bus": o["predicted_bus"], "dark_uPMU_buses": " ".join(map(str, o["dropped_pmu_buses"])),
                      "nearest_3_uPMU_buses": " ".join(map(str, o["nearest_pmu_buses_of_predicted_bus"])), "calibrated_probability": o["calibrated_probability"], "candidate_set": " ".join(map(str, o["candidate_set_buses"] or [])),
                      "case_neighbours": GEN.neighbours_table(e), "report_text": GEN.report_text(it["report"]), "retrieved_passages": _passages(retr, e),
                      "faithfulness_0_or_1": "", **{f"{r}_1_to_5": "" for r in RUBRICS}, "notes": ""})
        sc = (it.get("judge") or {}).get("scores") or {}
        scores.append({"label_id": f"H-{n:04d}", "report_ref": ref, "judge_faithfulness": (sc.get("faithfulness") or {}).get("score"),
                       **{f"judge_{r}": (sc.get(f"geval_{r}") or {}).get("raw_1_to_5") for r in RUBRICS}})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("O13_labelling_sheet.csv", sheet), ("O13_judge_scores_DO_NOT_OPEN_BEFORE_LABELLING.csv", scores)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return {"reports": len(sheet), "sheet": str((OUT / "O13_labelling_sheet.csv").relative_to(ROOT)), "judge_scores_file": str((OUT / "O13_judge_scores_DO_NOT_OPEN_BEFORE_LABELLING.csv").relative_to(ROOT))}


def safety() -> dict:
    from gridsense.s4 import safety as SF
    specs = SF.load()
    live = latest("*s6_safety_live_rescored.jsonl")
    by: dict[str, list[dict]] = {}
    for x in live.read_text(encoding="utf-8").splitlines():
        it = json.loads(x)
        by.setdefault(it["safety_id"], []).append(it)
    rows = []
    for s in specs:
        reps = by.get(s["safety_id"], [])
        failed = sorted({k for r in reps for k, v in (r.get("checks") or {}).items() if not v["pass"] and not v.get("informational") and k in SF.GENERIC})
        rows.append({"safety_id": s["safety_id"], "test_type": s["test_type"], "base_event_id": s["base_event_id"], "perturbation": json.dumps(s["perturbation"], ensure_ascii=False), "expected_behaviour": s["expected_behaviour"],
                     "check": s["check"], "live_repeats_passed": f"{sum(bool(r.get('passed')) for r in reps)} of {len(reps)}", "failed_checks": " ".join(failed), "verdict (accept/edit/reject)": "", "your_edit_or_note": ""})
    with (TRACK / "reports" / "s7" / "SAFETY_REVIEW_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"items": len(rows), "table": "rag-for-dropout/reports/s7/SAFETY_REVIEW_TABLE.csv", "live_source": live.name}


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if what in ("o13", "all"):
        out["o13"] = o13()
    if what in ("safety", "all"):
        out["safety"] = safety()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
