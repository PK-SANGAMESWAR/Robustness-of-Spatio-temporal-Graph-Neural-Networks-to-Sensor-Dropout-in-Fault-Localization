"""One-off: apply the author's O30 review (2026-11-02) to the draft notes in corpus/notes_v1. Run from the repo root:
    .venv/bin/python rag-for-dropout/corpus/notes_v1/_review_2026-11-02.py

What it does (author instructions; every project number is asserted against the project files BEFORE a file is written):
  - notes 01-07 and 11: source_ref = GENERIC; the book moves to `further_reading` (note 05 and 07 keep their project facts in `project_sources`).
  - notes 19-22: project numbers checked against the reports; the external papers are not checked against a page, so the same rule applies.
  - notes 09, 25, 26, 28: source_ref = the project design documents; external citations removed.
  - notes 14, 15, 16: checked against results/RESULTS.md; no note calls 0.846 the dropout-aware value. Three wording corrections (notes 09, 16, 20).
  - the notes that the author did not mention (08, 10, 12, 13, 17, 18, 23, 24, 27, 29, 30) and the SOP stay `draft`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
NOTES = ROOT / "rag-for-dropout/corpus/notes_v1"
TODAY = "2026-11-02"
GENERIC = "general background, not checked against a specific page"
ANDERSON = "Anderson, P. M. (1995). Analysis of Faulted Power Systems. IEEE Press"
PHADKE = "Phadke, A. G. and Thorp, J. S. Synchronized Phasor Measurements and Their Applications. Springer"

# note -> (source_ref, further_reading, project_sources, verification)
BACKGROUND = f"general background; not checked against a specific page; author review O30 {TODAY}"
CHECKED = f"project numbers checked against the named files {TODAY}; general background not checked against a specific page; author review O30 {TODAY}"
DESIGN = f"checked against the named project documents {TODAY}; author review O30 {TODAY}"
META: dict[str, dict] = {
    "note_01_single_line_to_ground": dict(source_ref=GENERIC, further_reading=[ANDERSON], verification=BACKGROUND),
    "note_02_line_to_line": dict(source_ref=GENERIC, further_reading=[ANDERSON], verification=BACKGROUND),
    "note_03_double_line_to_ground": dict(source_ref=GENERIC, further_reading=[ANDERSON], verification=BACKGROUND),
    "note_04_three_phase_fault": dict(source_ref=GENERIC, further_reading=[ANDERSON], verification=BACKGROUND),
    "note_05_fault_resistance": dict(source_ref=GENERIC, further_reading=[ANDERSON], verification=CHECKED,
                                     project_sources="results/RESULTS.md § 1 (macro-F1 by fault resistance, standard RGSAGE-Max)"),
    "note_06_voltage_sag_propagation": dict(source_ref=GENERIC, further_reading=["Kersting, W. H. Distribution System Modeling and Analysis. CRC Press"], verification=BACKGROUND),
    "note_07_micro_pmu": dict(source_ref=GENERIC, further_reading=[PHADKE, "IEEE Std C37.118.1"], verification=CHECKED,
                              project_sources="fault_localization/config/constants.py (NUM_PMU_NODES = 25, NUM_PHASES = 3, NUM_MASK_CHANNELS = 3, "
                                              "WINDOW_TENSOR_SHAPE = (25, 20, 6)); dataset_fullscale_50/processed/node_mapping.json (25 buses)"),
    "note_09_ambiguity_set": dict(source_ref="implementation-plan.md § 11.1 (ambiguity set) and § 11.3 (gate outcomes); rag-for-dropout/configs/gate_v1.json and "
                                             "configs/candidate_sets_v1.json; reports/layer0/CANDIDATE_SETS_REPORT.md; corpus/feeder_v1/feeder_fault_buses.md (nearest μPMUs)",
                                  verification=DESIGN),
    "note_11_causes_of_sensor_dropout": dict(source_ref=GENERIC, further_reading=[PHADKE], verification=BACKGROUND),
    "note_14_standard_training_collapse": dict(verification=DESIGN),
    "note_15_what_dropout_aware_training_recovers": dict(verification=DESIGN),
    "note_16_targeted_versus_random_dropout": dict(
        source_ref="results/RESULTS.md § 5; results/dropout_study_targeted/targeted_curves_standard.csv and targeted_curves_dropout_aware.csv "
                   "(the +0.307 to +0.473 range is derived from these two files); reports/layer0/FINAL_REPORT.md § 1", verification=DESIGN),
    "note_19_calibration": dict(source_ref=GENERIC, further_reading=["Guo et al. (2017) On Calibration of Modern Neural Networks", "Zadrozny and Elkan (2002)"], verification=CHECKED,
                                project_sources="reports/layer0/STEP2_REPORT.md; reports/layer0/FINAL_REPORT.md § 4"),
    "note_20_margin": dict(source_ref=GENERIC, further_reading=["Geifman and El-Yaniv (2017)"], verification=CHECKED,
                           project_sources="reports/layer0/FINAL_REPORT.md § 1 and § 4; rag-for-dropout/configs/gate_v1.json"),
    "note_21_selective_prediction": dict(source_ref=GENERIC, further_reading=["Geifman and El-Yaniv (2017)"], verification=CHECKED,
                                         project_sources="reports/layer0/FINAL_REPORT.md § 2"),
    "note_22_conformal_sets": dict(source_ref=GENERIC, further_reading=["Vovk, Gammerman and Shafer (2005)", "Angelopoulos and Bates (2021)"], verification=CHECKED,
                                   project_sources="reports/layer0/CANDIDATE_SETS_REPORT.md; reports/layer0/FINAL_REPORT.md § 0 and § 3; rag-for-dropout/claude-rag.md § 9 rule 4"),
    "note_25_prediction_is_not_confirmation": dict(source_ref="implementation-plan.md § 10.2 (prompt design), § 10.3 (report schema, pending_human_review), "
                                                             "§ 12.4 (human review) and § 13.5 (report quality)", verification=DESIGN),
    "note_26_gate_is_not_changed_by_text": dict(source_ref="implementation-plan.md § 1, § 3, § 12.4 and § 17; rag-for-dropout/claude-rag.md § 4 rule 3; "
                                                           "rag-for-dropout/configs/gate_v1.json (the gate rule is code)", verification=DESIGN),
    "note_28_simulation_limits": dict(source_ref="implementation-plan.md § 2.2 and § 5.1; rag-for-dropout/dropout-rag-intro.md § 6 (limitations 1 to 3); results/RESULTS.md",
                                      verification=DESIGN),
}

BODY_EDITS = {
    "note_09_ambiguity_set": [("Ambiguity grows when sensors are missing.",
                               "Which candidates are indistinguishable depends on which sensors are available.")],
    "note_16_targeted_versus_random_dropout": [("With one sensor lost the two models are not distinguishable.",
                                                "With one sensor lost, the dropout-aware model under targeted loss (0.755) is not distinguishable from the standard model under "
                                                "random loss (0.736; difference +0.018, p = 0.33 over three seeds). From two sensors lost, the dropout-aware worst case (0.695) "
                                                "is above the standard average case (0.590).")],
    "note_20_margin": [("At the 95 percent operating point of the frozen gate the margin is the only threshold that is used.",
                        "At the 95 percent operating point of the frozen gate the margin threshold is set at every k, the probability threshold only at k = 2, "
                        "and the kNN-agreement threshold at no k."),
                       ("The margin threshold grows with dropout, from 0.1459 with no sensor dark to 0.2518 with 10 dark. At this operating point the probability "
                        "threshold is 0 with 0, 1, 5 and 10 sensors dark, and the kNN-agreement threshold is 0 at every k.",
                        "The margin threshold is 0.1459 with 0 and 1 sensors dark, 0.1368 with 2, 0.1633 with 5 and 0.2518 with 10 dark. At this operating point the "
                        "probability threshold is 0 with 0, 1, 5 and 10 sensors dark and 0.5424 with 2 dark, and the kNN-agreement threshold is 0 at every k.")],
}
UNMENTIONED = {"note_08_partial_observability", "note_10_window_semantics", "note_12_missing_data_representation", "note_13_dropout_aware_training",
               "note_17_noise_versus_dropout", "note_18_soft_vote", "note_23_similar_simulated_cases", "note_24_gate_outcomes",
               "note_27_retrieval_scores", "note_29_gate_belongs_to_one_checkpoint", "note_30_aggregate_versus_individual"}


def split(raw: str) -> tuple[dict, str]:
    end = raw.index("\n---\n", 4)
    return yaml.safe_load(raw[4:end]), raw[end + 5:]


def verify_project_numbers() -> list[str]:
    """Assert the numbers that notes 07, 14-16, 20 quote against project files. Returns the log lines."""
    log = []
    from fault_localization.config import constants as C
    assert C.NUM_PMU_NODES == 25 == len(C.PMU_BUSES) and C.NUM_PHASES == 3 and C.NUM_MASK_CHANNELS == 3 and C.WINDOW_TENSOR_SHAPE == (25, 20, 6)
    assert len(json.loads((ROOT / "dataset_fullscale_50/processed/node_mapping.json").read_text())) == 25
    ev = sorted((ROOT / "dataset_fullscale_50/processed/windows").glob("*.npz"))[0]
    with np.load(ev) as z:
        assert z["windows"].shape[1:] == (25, 20, 6), z["windows"].shape
    log.append("note_07: 25 μPMUs, 3 voltage + 3 mask channels, window [25, 20, 6] = constants.py = node_mapping.json = an event file")
    g = json.loads((ROOT / "rag-for-dropout/configs/gate_v1.json").read_text())["per_k"]
    t = {int(k): v["acc95"] for k, v in g.items()}
    assert [round(t[k]["tau_m"], 4) for k in (0, 1, 2, 5, 10)] == [0.1459, 0.1459, 0.1368, 0.1633, 0.2518]
    assert [round(t[k]["tau_p"], 4) for k in (0, 1, 2, 5, 10)] == [0.0, 0.0, 0.5424, 0.0, 0.0] and all(t[k]["tau_k"] == 0.0 for k in t)
    log.append("note_20: gate_v1 acc95 thresholds equal the corrected sentences (tau_m 0.1459/0.1459/0.1368/0.1633/0.2518; tau_p 0 except 0.5424 at k=2; tau_k 0)")
    res = (ROOT / "results/RESULTS.md").read_text()
    for needle in ("| **RGSAGE-Max** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | 0.402 → **0.913** |", "| Standard, **targeted** | 0.846 | **0.448 ± 0.039** |",
                   "| Standard, random (ref) | 0.846 | 0.736 ± 0.013 |", "| Dropout-aware, **targeted** | 0.839 | 0.755 | 0.695 | 0.600 | **0.521** |",
                   "(0.695 vs 0.590 at k=2, p=0.042)", "At k=1 the two are indistinguishable (+0.018, p=0.33)", "AUDC 0.913 → **0.747**",
                   "| **Dropout-aware training** | **0.913** | 0.839 | 0.827 | 0.815 | 0.774 | **0.674** | **80.3 %** |", "(−0.008, p=0.181)",
                   "| Standard (zeroing) | 0.402 | 0.846 | 0.736 | 0.590 | 0.236 | 0.048 | 5.7 % |"):
        assert needle in res, needle
    log.append("notes 14-16: every quoted number found in results/RESULTS.md; the k=1 'indistinguishable' pair is dropout-aware TARGETED vs standard RANDOM (+0.018, p=0.33)")
    import csv
    def curve(f):
        return {int(r["k"]): float(r["f1_mean"]) for r in csv.DictReader((ROOT / "results/dropout_study_targeted" / f).open())}
    s, d = curve("targeted_curves_standard.csv"), curve("targeted_curves_dropout_aware.csv")
    diffs = [d[k] - s[k] for k in (1, 2, 5, 10)]
    assert round(min(diffs), 3) == 0.307 and round(max(diffs), 3) == 0.473, diffs
    log.append(f"note_16: dropout-aware minus standard under the same targeting = {min(diffs):.4f} to {max(diffs):.4f} (k = 1..10) = +0.307 to +0.473")
    rep = (ROOT / "rag-for-dropout/reports/layer0/FINAL_REPORT.md").read_text()
    for needle in ("| ECE isotonic | 0.0084 | 0.0056 | 0.0064 | 0.0086 | 0.0057 |", "| ECE raw vote share | 0.1211 | 0.1247 | 0.1292 | 0.1443 | 0.1726 |",
                   "| margin | 0.9152 (", "| isotonic_prob | 0.9109 (", "0.8868 (0.883–0.891)", "0.8850 (0.881–0.889)",
                   "0.8449 (0.838–0.852)", "k = 0: coverage 0.772"):
        assert needle in rep, needle
    step2 = (ROOT / "rag-for-dropout/reports/layer0/STEP2_REPORT.md").read_text()
    assert "ECE 0.007 at k = 0 and 0.048 at k = 10" in step2
    cs = (ROOT / "rag-for-dropout/reports/layer0/CANDIDATE_SETS_REPORT.md").read_text()
    assert "0.9058" in cs and "4.27" in cs and "3.13" in cs
    log.append("notes 19-22: ECE, AUROC, coverage and candidate-set numbers found in the Layer 0 reports")
    return log


def main() -> None:
    log = verify_project_numbers()
    plan = (ROOT / "rag-for-dropout/implementation-plan.md").read_text()
    for h in ("### 2.2 Non-goals", "### 5.1 Approved source types", "### 10.2 Prompt design", "### 10.3 Suggested report schema", "### 11.1 Checks to implement",
              "### 11.3 Gate policy", "### 12.4 Human-in-the-loop", "### 13.5 Report quality", "## 1. Executive summary", "## 17. Security"):
        assert h in plan, h
    log.append("plan sections cited in source_ref exist: 1, 2.2, 3, 5.1, 10.2, 10.3, 11.1, 11.3, 12.4, 13.5, 17")
    allnotes = {p.stem: p for p in NOTES.glob("note_*.md")}
    assert len(allnotes) == 30
    for stem, p in sorted(allnotes.items()):
        front, body = split(p.read_text(encoding="utf-8"))
        if stem in UNMENTIONED:
            continue
        m = dict(META.get(stem, {}))
        for a, b in BODY_EDITS.get(stem, []):
            assert body.count(a) == 1, (stem, a[:50])
            body = body.replace(a, b)
        new = {}
        for k, v in front.items():
            new[k] = v
            if k == "source_ref":
                if "source_ref" in m:
                    new["source_ref"] = m["source_ref"]
                if "further_reading" in m:
                    new["further_reading"] = m["further_reading"]
                if "project_sources" in m:
                    new["project_sources"] = m["project_sources"]
        new["verification"] = m.get("verification", DESIGN)
        new["status"], new["version"], new["reviewed"] = "reviewed", 1, TODAY
        new["review"] = f"accepted by the author in the O30 review ({TODAY})" + ("; wording corrected by session A" if stem in BODY_EDITS else "")
        txt = "---\n" + yaml.safe_dump(new, sort_keys=False, allow_unicode=True, width=1000) + "---\n" + body
        p.write_text(txt, encoding="utf-8")
    out = [stem for stem in sorted(allnotes) if stem not in UNMENTIONED]
    print(json.dumps({"reviewed": len(out), "still_draft": sorted(UNMENTIONED), "checks": log}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
