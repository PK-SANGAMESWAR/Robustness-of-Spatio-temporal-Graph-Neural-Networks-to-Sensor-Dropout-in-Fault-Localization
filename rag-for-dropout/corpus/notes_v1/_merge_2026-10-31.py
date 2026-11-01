"""One-off merge of the two draft note sets into corpus/notes_v1 and corpus/sop_v1 (author instruction 2026-10-31).

Base: session B's 30 notes and its SOP (`_quarantine_session2/rag-for-dropout/corpus/v2/`), text unchanged so that its golden
spans stay valid. Session A's notes (6 topic files, N01-N30, superseded; copy in session A's scratchpad) add sentences at the end
of 13 notes and 3 SOP items. Every file records `merged_from` and its sources. Status stays draft. Run from the repo root:
    .venv/bin/python rag-for-dropout/corpus/notes_v1/_merge_2026-10-31.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "rag-for-dropout/_quarantine_session2/rag-for-dropout/corpus/v2"
OUT_NOTES, OUT_SOP = ROOT / "rag-for-dropout/corpus/notes_v1", ROOT / "rag-for-dropout/corpus/sop_v1"

# note stem -> (session A notes merged in, added text, added sources)
ADD: dict[str, tuple[list[str], str, list[str]]] = {
    "note_08_partial_observability": (["A:N03", "A:N05"],
        "Half of the windows in the corpus are no-fault windows. A classifier that always answers no fault scores a 26-class "
        "macro-F1 of 0.0256, so a macro-F1 near this value means that the model has lost its ability to locate faults.",
        ["results/RESULTS.md section 8 (trivial reference)", "CLAUDE.md key facts (exactly 50 % no-fault)"]),
    "note_09_ambiguity_set": (["A:N20"],
        "Two fault buses that are close to the same μPMUs give similar voltage patterns. When one of those μPMUs is dark, the "
        "pattern can fit both buses, and the candidate set then contains both.",
        ["corpus/feeder_v1/feeder_fault_buses.md (nearest μPMUs)"]),
    "note_10_window_semantics": (["A:N04"],
        "Because window w0020 has no post-fault sample, no model can classify it from its content.",
        ["fault_localization/models/common.py (unlabelable_window_mask)"]),
    "note_16_targeted_versus_random_dropout": (["A:N09", "A:N28"],
        "Under targeting the area under the degradation curve of the dropout-aware model falls from 0.913 to 0.747, and its "
        "macro-F1 with 10 sensors dark is 0.521. The GridSense gate is fitted for random dropout only: a dropout pattern that it "
        "was not fitted for gives REJECT with UNSUPPORTED_DROPOUT_PATTERN. If a dark μPMU is one of the nearest μPMUs of the "
        "predicted bus, treat the location with more care.",
        ["results/RESULTS.md section 5", "reports/layer0/FINAL_REPORT.md section 1"]),
    "note_17_noise_versus_dropout": (["A:N10"],
        "A wrong value from a present μPMU (a silent failure) also leaves the mask at 1. The GridSense gate has no silent-failure "
        "check (ablation step A3 is not run), so a report cannot exclude a wrong reading from a present μPMU.",
        ["rag-for-dropout/EVAL.md section 10 (A3 not run)"]),
    "note_19_calibration": (["A:N13"],
        "The values 0.007 and 0.048 are from the validation half cal-B (post-fault windows). On test the raw vote share has an "
        "ECE of 0.121 to 0.173.",
        ["reports/layer0/STEP2_REPORT.md", "reports/layer0/FINAL_REPORT.md section 4"]),
    "note_20_margin": (["A:N14"],
        "The margin threshold grows with dropout, from 0.1459 with no sensor dark to 0.2518 with 10 dark. At this operating "
        "point the probability threshold is 0 with 0, 1, 5 and 10 sensors dark, and the kNN-agreement threshold is 0 at every k.",
        ["reports/layer0/FINAL_REPORT.md section 1"]),
    "note_21_selective_prediction": (["A:N15"],
        "At the 98 percent operating point coverage is 0.772 with no sensor dark and 0.376 with 10 dark. More dark μPMUs "
        "means more LOW_CONFIDENCE reports.",
        ["reports/layer0/FINAL_REPORT.md section 2"]),
    "note_22_conformal_sets": (["A:N17", "A:N18", "A:N19"],
        "The 90 percent is a property of the method over many LOW_CONFIDENCE windows; one event's set can miss the true "
        "location. A CONFIDENT report names one bus, a REJECT report shows no bus and no set, and only a LOW_CONFIDENCE report "
        "shows a candidate set.",
        ["reports/layer0/CANDIDATE_SETS_REPORT.md", "rag-for-dropout/claude-rag.md section 9 rule 4"]),
    "note_23_similar_simulated_cases": (["A:N21", "A:N22", "A:N24", "A:N25"],
        "kNN agreement is the fraction of neighbours whose true bus equals the prediction; the neighbour error rate is the "
        "fraction of neighbours that the model itself got wrong. On test, kNN agreement detects wrong predictions with AUROC "
        "0.748 to 0.826, lower than the margin. It stayed in the gate by a pre-registered rule with a small effect (AUROC "
        "+0.011 on validation), and at the 95 percent operating point its threshold is 0. Neighbours cannot show conditions "
        "outside the simulation or a wrong reading from a present μPMU.",
        ["reports/layer0/FINAL_REPORT.md section 4", "reports/layer0/DECISION_RULE.md", "reports/layer0/CANDIDATE_SETS_REPORT.md"]),
    "note_25_prediction_is_not_confirmation": (["A:N26"],
        "A report is not an instruction to operate equipment.",
        ["implementation-plan.md sections 10.2 and 12.4"]),
    "note_27_retrieval_scores": (["A:N30"],
        "It does not invent a procedure, a document section or a document revision.",
        ["implementation-plan.md section 10.5"]),
    "note_28_simulation_limits": (["A:N29"],
        "Every report states that no utility-approved procedure was used.",
        ["implementation-plan.md section 5.1"]),
}

SOP_ADD = {  # insert after the last item of the section; (item id, text, session A source)
    "## SOP-ILL-4": ("SOP-ILL-4.3", "The reviewer compares the similar simulated cases with the prediction. If most neighbours "
                     "point to a different bus, the reviewer records the disagreement and treats the event as uncertain, also "
                     "when the outcome is CONFIDENT.", "A:step 4"),
    "## SOP-ILL-5": ("SOP-ILL-5.3", "The reviewer opens each cited document section and checks that the quoted text is in the "
                     "source and supports the statement. A missing or unsupported citation is a reason to reject the report.",
                     "A:step 5"),
    "## SOP-ILL-6": ("SOP-ILL-6.2", "If the report marks the evidence as insufficient, or the outcome is REJECT, the reviewer "
                     "sends the event to the engineer on duty for manual analysis.", "A:step 8"),
}


def split(raw: str) -> tuple[dict, str]:
    end = raw.index("\n---\n", 4)
    meta = {}
    for line in raw[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 5:]


def front(meta: dict) -> str:
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def main() -> None:
    if not SRC.exists():
        raise SystemExit("session B's files are archived, not in the tree. To re-run, extract ~/archive/gridsense/_quarantine_session2_archived_2026-11-01.tar.gz "
                         "(sha256 cc38a3aff77eca85...) into rag-for-dropout/")
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    for p in sorted((SRC / "notes").glob("note_*.md")):
        meta, body = split(p.read_text(encoding="utf-8"))
        merged, extra, srcs = ADD.get(p.stem, ([], "", []))
        meta = {"document_id": meta["document_id"], "title": meta["title"], "document_type": meta["document_type"],
                "status": "draft", "approval_status": meta["approval_status"], "version": 0,
                "source_ref": "; ".join([meta["source_ref"]] + srcs), "verification": meta["verification"],
                "merged_from": [f"B:{p.stem}"] + merged, "created": "2026-10-31",
                "created_by": "agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)", "reviewed": None}
        body = body.rstrip("\n") + ("\n\n" + extra if extra else "") + "\n"
        (OUT_NOTES / p.name).write_text(front(meta) + body, encoding="utf-8")
    meta, body = split((SRC / "sop" / "illustrative_sop.md").read_text(encoding="utf-8"))
    lines, out, current = body.splitlines(), [], None
    for i, line in enumerate(lines):
        out.append(line)
        if line.startswith("## "):
            current = next((k for k in SOP_ADD if line.startswith(k)), None)
        nxt = lines[i + 1] if i + 1 < len(lines) else "## "
        if current and nxt.startswith("## "):
            item, text, _ = SOP_ADD[current]
            last = max(j for j, l in enumerate(out) if l.strip())   # after the last item, before the blank line
            out.insert(last + 1, f"{item} {text}")
            current = None
    meta = {"document_id": meta["document_id"], "title": meta["title"], "document_type": meta["document_type"], "status": "draft",
            "approval_status": "illustrative", "version": 0, "source_ref": meta["source_ref"],
            "merged_from": ["B:illustrative_sop"] + [v[2] for v in SOP_ADD.values()], "created": "2026-10-31",
            "created_by": "agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)", "reviewed": None}
    OUT_SOP.mkdir(parents=True, exist_ok=True)
    (OUT_SOP / "illustrative_sop.md").write_text(front(meta) + "\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
