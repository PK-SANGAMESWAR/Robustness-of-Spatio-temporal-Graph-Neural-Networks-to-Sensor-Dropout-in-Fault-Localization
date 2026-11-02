"""One-off: add 12 near-miss unanswerable queries (6 dev, 6 test) as DRAFT items (author, 2026-11-02, step 6). The topic is in the corpus; the answer is not.
    PYTHONPATH=src:. .venv/bin/python rag-for-dropout/data/eval/add_near_miss_queries.py
Absence was checked by searching the parsed corpus for the answer-bearing terms (listed per item in `absence_check`); the script re-runs those searches and refuses to add an item that a pattern finds."""
import json
import re
from pathlib import Path

from gridsense.eval.golden import validate
from gridsense.ingest.corpus import load_corpus

PATH = Path(__file__).resolve().parent / "golden_retrieval.jsonl"
# (id, split, query, critical terms, documents that discuss the topic, patterns that must NOT occur in the corpus, note)
ITEMS = [
    ("R-0057", "dev", "What is the fault current at bus 57?", ["fault current", "bus 57"], ["note_01_single_line_to_ground", "note_05_fault_resistance"], [r"\bkA\b", r"\d\s?amps?\b", r"bus 57"], "fault current appears only in words (notes 01 and 05), never with a value"),
    ("R-0058", "dev", "What is the line impedance between bus 13 and bus 18?", ["impedance", "bus 13", "bus 18"], ["paper1_rev1170", "feeder_pmu_nodes"], [r"\bbus 13\b.{0,60}\bbus 18\b.{0,80}(ohm|Ω)"], "impedance is discussed in words; no line value"),
    ("R-0059", "dev", "How many windows of fault bus 57 are in the test split?", ["windows", "bus 57", "test split"], ["feeder_overview"], [r"bus 57"], "only split totals are given, no per-bus counts"),
    ("R-0060", "dev", "Which sensors were dark in the worst-case run of the targeted study?", ["worst-case", "targeted"], ["note_16_targeted_versus_random_dropout", "results_md"], [r"worst-case run", r"dark in the worst"], "the targeted study is described by macro-F1 per k, not by the sensors of a run"),
    ("R-0061", "dev", "What is the calibrated probability for a vote share of 0.9 with 5 sensors dark?", ["calibrated probability", "vote share"], ["note_19_calibration", "note_29_gate_belongs_to_one_checkpoint"], [r"vote share (of|is) 0\.9"], "the isotonic maps are described, not listed"),
    ("R-0062", "dev", "What is the margin threshold at the 98 percent operating point with 10 sensors dark?", ["margin threshold", "98 percent"], ["note_20_margin", "note_21_selective_prediction"], [r"98 percent.{0,80}threshold", r"threshold.{0,80}98 percent"], "the notes give the 95 percent thresholds and the 98 percent coverage, not the 98 percent thresholds"),
    ("R-0063", "test", "What is the fault current at bus 108 for a 10 ohm ABCG fault?", ["fault current", "bus 108"], ["note_05_fault_resistance", "feeder_fault_taxonomy"], [r"bus 108", r"\bkA\b"], "fault resistance and fault types are described; no current value"),
    ("R-0064", "test", "What was the validation macro-F1 of the dropout-aware model at its best epoch?", ["validation macro-F1", "best epoch"], ["paper1_rev1170", "results_md"], [r"validation macro-F1 (of|is|was) 0\.", r"best.validation.{0,30}0\.\d"], "the checkpoint rule names the best validation macro-F1; its value is not given"),
    ("R-0065", "test", "What was the training loss at the last epoch?", ["training loss", "last epoch"], ["paper1_rev1170"], [r"training loss"], "the objective (unweighted cross-entropy) is described; no loss value"),
    ("R-0066", "test", "What is the macro-F1 of the dropout-aware model with 3 sensors dark?", ["macro-F1", "3 sensors"], ["note_15_what_dropout_aware_training_recovers", "results_md"], [r"k\s?=\s?3\b", r"\b3 sensors dark"], "only k = 0, 1, 2, 5 and 10 were evaluated"),
    ("R-0067", "test", "How long can a μPMU keep its time reference after the clock signal is lost?", ["time reference", "clock signal"], ["note_11_causes_of_sensor_dropout", "note_07_micro_pmu"], [r"holdover", r"\bGPS\b", r"clock signal"], "loss of the time reference is named as a cause of dropout; no duration"),
    ("R-0068", "test", "What is the voltage at bus 25 during a 1 ohm AG fault?", ["voltage", "bus 25"], ["paper1_rev1170", "note_06_voltage_sag_propagation"], [r"bus 25\b.{0,100}(sag|volt)", r"(sag|volt).{0,100}bus 25\b"], "voltage sag is described in words and for one other bus (101) in the paper; none for bus 25"),
]
docs = load_corpus()
corpus = "\n".join(d.text for d in docs)
rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 56 and not any(r["query_id"] in {i[0] for i in ITEMS} for r in rows), "already added or unexpected file"
ids = {d.document_id for d in docs}
for qid, split, q, crit, near, absent, note in ITEMS:
    assert set(near) <= ids, qid
    for pat in absent:
        assert not re.search(pat, corpus, flags=re.I), f"{qid}: pattern {pat!r} is found in the corpus"
    rows.append({"query_id": qid, "query": q, "query_type": "unanswerable", "golden_contexts": [], "reference_answer": "", "answerable": False, "critical_terms": crit, "split": split,
                 "status": "draft", "source": "agent:claude-sonnet-5-5, step 6 of the 2026-11-02 instructions: a near-miss unanswerable query (the topic is in the corpus, the answer is not)",
                 "created": "2026-11-02", "created_by": "agent:claude-sonnet-5-5", "converted_by": None, "golden_version": 2, "reviewed": None,
                 "near_miss_of": near, "absence_check": {"patterns_not_found_in_the_corpus": absent, "note": note}})
assert not validate(rows, docs)
PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print({"items": len(rows), "new": len(ITEMS), "dev": sum(1 for i in ITEMS if i[1] == "dev"), "test": sum(1 for i in ITEMS if i[1] == "test")})
