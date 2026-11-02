"""Draft golden events for S4 (O36, author 2026-11-02): 60 events from the TEST split at the report window w = 25.

    PYTHONPATH=src:. nice -n 19 .venv/bin/python -m gridsense.events.draft_events

Built from the stored Layer-0 artifacts of seed 42 (soft votes and latents of the test split, mask offset 0): no new model run is needed for the
selection. One forward pass over the batches of the 60 chosen windows replays the dropout masks with the paper's own `apply_sensor_dropout_batch`
and checks that the stored votes are reproduced (a check, not a new result). CPU only, 2 threads.

Strata, 12 events per k in {0, 1, 2, 5, 10}: 4 `confident_correct` (gate CONFIDENT, top-1 right), 4 `high_conf_error` (gate CONFIDENT, top-1 wrong, calibrated p >= 0.9),
4 `low_confidence` (gate LOW_CONFIDENCE; 2 with top-1 right, 2 with top-1 wrong). A seeded greedy picker takes, for each slot, a candidate whose fault type
(then fault bus) is the least used so far, ties broken by the seed; one event per run. kNN agreement does not change the gate at the 95 % point (tau_k = 0).
Reference report points that come from the event object are written with status `reviewed` (author instruction); judgement points and passages stay `draft`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from gridsense.config import ROOT, TRACK, load_config, torch_threads
from gridsense.data.splits import DATA_ROOT, event_of, scenario_of, window_index
from gridsense.eval import golden as G
from gridsense.ingest.corpus import load_corpus
from gridsense.verification.candidate_sets import CandidateSets
from gridsense.verification.gate import OUTCOME_NAMES, Gate, reasons_to_names

SEED = 36
SEED_MODEL = 42
KS = (0, 1, 2, 5, 10)
PER_SLOT = {"confident_correct": 4, "high_conf_error": 4, "low_confidence_correct": 2, "low_confidence_wrong": 2}
P_HIGH = 0.9
K_NN = 10
STRATUM_OF = {"confident_correct": "confident_correct", "high_conf_error": "high_conf_error", "low_confidence_correct": "low_confidence", "low_confidence_wrong": "low_confidence"}
LAT = TRACK / "eval" / "latents" / f"seed{SEED_MODEL}"
OUT_JSONL = TRACK / "data" / "eval" / "golden_events.jsonl"
OUT_DIR = TRACK / "reports" / "events"
TODAY = "2026-11-02"
FIXED_KINDS = ("not_confirmation", "aggregate", "limits")     # the same statement in every event; the author approves each once (2026-11-02)
S4_SEED_BASE = 360000                                          # cell seed = base + 10 * k + stratum index (author 2026-11-02: fixed seed inside each cell, no hand selection)
S4_PER_CELL = 2
STRATA = ("confident_correct", "high_conf_error", "low_confidence")


# ---- selection ------------------------------------------------------------------------------------------------------------------
def pool_rows(window_ids: list[str], index: int) -> np.ndarray:
    return np.flatnonzero(np.array([window_index(w) for w in window_ids]) == index)


def classify(gate: Gate, cs: CandidateSets, votes: dict[int, np.ndarray], y: np.ndarray) -> dict[int, dict]:
    """Gate decision (agreement 0: tau_k = 0 at the 95 % point), calibrated p, margin, candidate sets, correctness; per k, for the pool windows."""
    out = {}
    for k, v in votes.items():
        v = np.asarray(v, dtype=float)
        r = gate.decide(k, v, np.zeros(len(v)))
        out[k] = {"outcome": r["outcome"], "reason": r["reason"], "p": r["p"], "margin": r["margin"], "pred": v.argmax(1), "correct": v.argmax(1) == y,
                  "set": cs.sets(k, v), "top1_share": np.sort(v, 1)[:, -1] / 25.0}
    return out


def slot_masks(c: dict) -> dict[str, np.ndarray]:
    conf, low = c["outcome"] == 0, c["outcome"] == 1
    return {"confident_correct": conf & c["correct"], "high_conf_error": conf & ~c["correct"] & (c["p"] >= P_HIGH),
            "low_confidence_correct": low & c["correct"], "low_confidence_wrong": low & ~c["correct"]}


def pick(pool_ids: list[str], y: np.ndarray, cls: dict[int, dict], meta_type: dict[str, str], seed: int = SEED) -> list[dict]:
    """Seeded greedy balancing: fault types first, then fault buses; one event per run."""
    rng = np.random.default_rng(seed)
    runs = np.array([event_of(w) for w in pool_ids])
    types = np.array([meta_type[r] for r in runs])
    used_runs, n_type, n_bus, chosen = set(), Counter(), Counter(), []
    slots = [(k, s) for k in KS for s in PER_SLOT for _ in range(PER_SLOT[s])]
    for i in rng.permutation(len(slots)):
        k, s = slots[i]
        ok = np.flatnonzero(slot_masks(cls[k])[s])
        ok = np.array([j for j in ok if runs[j] not in used_runs])
        if len(ok) == 0:
            raise RuntimeError(f"no candidate left for k={k} {s}")
        key = np.array([(n_type[types[j]], n_bus[int(y[j])]) for j in ok])
        best = np.flatnonzero((key[:, 0] == key[:, 0].min()))
        best = best[key[best, 1] == key[best, 1].min()]
        j = int(ok[rng.choice(best)])
        used_runs.add(runs[j]); n_type[types[j]] += 1; n_bus[int(y[j])] += 1
        chosen.append({"row": j, "k": k, "slot": s, "stratum": STRATUM_OF[s]})
    return sorted(chosen, key=lambda c: (c["k"], list(PER_SLOT).index(c["slot"]), c["row"]))


# ---- neighbours (memory-light, held-out) ---------------------------------------------------------------------------------------
def neighbours(q_vecs: np.ndarray, q_scn: list[str], n: int = K_NN, chunk: int = 100_000):
    """Exact cosine top-n of each query against the TRAIN window store at k = 0 (representation b), own scenario removed (held-out, O20)."""
    import torch
    from gridsense.model.latent_repr import represent

    torch.set_num_threads(torch_threads())
    d = LAT / "train"
    ids = (d / "window_ids.txt").read_text().split()
    h = np.load(d / "h_k0_o0.npy", mmap_mode="r")
    q = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(q_vecs, dtype=np.float32)), dim=1)
    sim = torch.empty(len(q), len(ids))
    for a in range(0, len(ids), chunk):
        blk = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(h[a:a + chunk], dtype=np.float32)), dim=1)
        sim[:, a:a + blk.shape[0]] = q @ blk.T
    scn = np.array([scenario_of(event_of(w)) for w in ids])
    for i, s in enumerate(q_scn):
        sim[i, torch.from_numpy(scn == s)] = float("-inf")
    sc, ix = sim.topk(n, dim=1)
    y = np.load(d / "labels.npy").astype(int)
    pred_ok = np.load(d / "votes_k0_o0.npy", mmap_mode="r")
    ix = ix.numpy()
    ok = np.array([[int(np.argmax(pred_ok[j])) == y[j] for j in row] for row in ix])
    return sc.numpy(), ix, ids, y, ok, scn


# ---- mask replay (the paper's own function) ----------------------------------------------------------------------------------
def replay_masks(rows: list[int], ks: list[int], n_test: int, expect_ids: list[str], expect_votes: dict[int, np.ndarray]) -> list[dict]:
    """Dropped node indices per chosen window, by re-running the paper's masking on the real batch, plus a check of the stored votes."""
    import torch
    from fault_localization.data.dataset import EventChunkBatchSampler, _collate
    from fault_localization.dropout.masking import apply_sensor_dropout_batch
    from fault_localization.models.voting import soft_vote
    from gridsense.model.predictor import lean_dataset, load_model

    torch.set_num_threads(torch_threads())
    ds = lean_dataset("test", DATA_ROOT)
    batches = list(EventChunkBatchSampler(ds, 32, False, 0))
    start = np.cumsum([0] + [len(b) for b in batches])
    assert [ds.records[i][0] for b in batches for i in b] == expect_ids, "sampler order differs from the stored window order"
    model, _ = load_model(SEED_MODEL, ds.edge_index, "cpu")
    ei = getattr(model, "edge_index", None)
    res = []
    with torch.no_grad():
        for row, k in zip(rows, ks):
            bi = int(np.searchsorted(start, row, side="right") - 1)
            batch = _collate([ds[i] for i in batches[bi]])
            x = batch["x"]
            if k == 0:
                xd, dropped = x, [[] for _ in range(len(x))]
            else:
                xd, _, dropped = apply_sensor_dropout_batch(x, k=k, seed=(10000 * k + 0) * 10000 + bi, policy="zero_messages", edge_index=ei)
            scores, _ = soft_vote(model(xd))
            pos = row - int(start[bi])
            assert batch["window_id"][pos] == expect_ids[row]
            diff = float(np.abs(scores[pos].numpy() - expect_votes[k][row]).max())
            res.append({"batch_idx": bi, "position": pos, "mask_seed": (10000 * k) * 10000 + bi, "dropped_nodes": [int(n) for n in dropped[pos]], "max_vote_diff": diff})
    return res


# ---- text helpers ---------------------------------------------------------------------------------------------------------------
def nearest_pmus(docs: dict, bus: int) -> list[int]:
    """The 3 nearest μPMU buses of a fault bus, read from the corpus document (feeder_fault_buses)."""
    for line in docs["feeder_fault_buses"].text.splitlines():
        m = re.match(rf"\|\s*{bus}\s*\|\s*bus (\d+) \(.*?\)\s*\|\s*bus (\d+) \(.*?\)\s*\|\s*bus (\d+) \(", line)
        if m:
            return [int(g) for g in m.groups()]
    raise KeyError(bus)


PASSAGES = {  # one sentence per note; each occurs exactly once in its note (checked at build time)
    "gate": ("note_24_gate_outcomes", "The gate returns CONFIDENT, LOW_CONFIDENCE or REJECT."),
    "not_confirmation": ("note_25_prediction_is_not_confirmation", "The model predicts a candidate fault bus. It does not confirm a fault."),
    "conformal": ("note_22_conformal_sets", "A conformal prediction set contains the classes whose score passes a threshold fitted on held-out data, so that the true class is inside the set with a chosen probability, for example 90 percent."),
    "neighbours": ("note_23_similar_simulated_cases", "Neighbours are precedents, not proof that a prediction is correct."),
    "missing": ("note_12_missing_data_representation", "A dropped μPMU has its three voltage channels set to 0 and its three mask channels set to 0, giving the mask pattern (0,0,0)."),
    "near_sensors": ("note_06_voltage_sag_propagation", "Sensors close to the fault therefore carry the most information about its location."),
    "limits": ("note_28_simulation_limits", "Results therefore do not show safety on a real grid, and the documents in this corpus were curated for research and are not approved by a utility."),
    "aggregate": ("note_30_aggregate_versus_individual", "They must not be quoted as the confidence of the current prediction."),
    "ambiguity": ("note_09_ambiguity_set", "When the evidence is ambiguous a report should show the candidates and not a single bus."),
    "nearest_dark": ("note_16_targeted_versus_random_dropout", "If a dark μPMU is one of the nearest μPMUs of the predicted bus, treat the location with more care."),
}


def passage(docs: dict, key: str, grade: int) -> dict:
    did, span = PASSAGES[key]
    hits = G.locate(docs[did], span)
    assert len(hits) == 1, (did, len(hits))
    return {"document_id": did, "section_path": list(docs[did].section_at(hits[0][0])), "span": span, "span_hash": G.span_hash(span), "grade": grade}


def _bus_counts(buses: list[int]) -> str:
    return ", ".join(f"bus {b} ({n} of {len(buses)})" for b, n in Counter(buses).most_common())


def build_item(i: int, ev: dict, docs: dict) -> dict:
    o = ev["event_object"]
    auto, judge, pts = [], [], []
    def A(kind, text): auto.append({"point_id": f"E-{i:04d}-A{len(auto) + 1}", "kind": kind, "text": text, "status": "reviewed", "derived_from": "event_object"})
    def J(kind, text): judge.append({"point_id": f"E-{i:04d}-J{len(judge) + 1}", "kind": kind, "text": text, "status": "draft", "derived_from": "agent judgement"})
    A("location", f"The report states the top-ranked fault location: bus {o['predicted_bus']} (class {o['predicted_class']})." if o["predicted_bus"] else "The report states that the top-ranked class is 'no fault' (class 0).")
    A("gate_outcome", f"The report states the gate outcome {o['gate_outcome']} with the reason code(s) {', '.join(o['reason_codes'])}.")
    if o["candidate_set_buses"] is not None:
        A("candidate_set", f"The report lists the candidate set of {len(o['candidate_set_buses'])} bus(es): {', '.join(map(str, o['candidate_set_buses']))}.")
    A("neighbour_locations", f"The report states the fault buses of the {K_NN} most similar simulated cases (held-out): {_bus_counts([n['bus'] for n in o['knn']['neighbours']])}.")
    A("dark_sensors", "The report states that no μPMU is dark (k = 0)." if o["k"] == 0 else f"The report lists the {o['k']} dark μPMU(s): bus {', bus '.join(map(str, o['dropped_pmu_buses']))}.")
    J("not_confirmation", "The report says that the result is a prediction and not a confirmation of a fault.")
    J("aggregate", "The report does not present an aggregate accuracy as the confidence of this prediction.")
    J("limits", "The report keeps the simulation limit in view (no claim of safety on a real grid).")
    pas = [passage(docs, "gate", 3), passage(docs, "not_confirmation", 3), passage(docs, "neighbours", 2), passage(docs, "aggregate", 2), passage(docs, "limits", 1)]
    if o["gate_outcome"] == "LOW_CONFIDENCE":
        J("ambiguity", "The report shows the candidates and does not present one bus as the answer.")
        pas += [passage(docs, "conformal", 3), passage(docs, "ambiguity", 3)]
    if o["k"] > 0:
        pas += [passage(docs, "missing", 2)]
        if set(o["dropped_pmu_buses"]) & set(o["nearest_pmu_buses_of_predicted_bus"]):
            J("nearest_dark", f"The report notes that a dark μPMU is among the nearest μPMUs of the predicted bus ({', '.join(map(str, sorted(set(o['dropped_pmu_buses']) & set(o['nearest_pmu_buses_of_predicted_bus']))))}) and advises more care.")
            pas += [passage(docs, "nearest_dark", 3), passage(docs, "near_sensors", 2)]
    if ev["stratum"] == "high_conf_error" or (o["knn"]["agreement"] < 0.5):
        J("neighbour_disagreement", f"The report points out that only {int(round(o['knn']['agreement'] * K_NN))} of the {K_NN} neighbours have the predicted bus." if o["knn"]["agreement"] < 0.5 else
          "The report states that the neighbours support the predicted bus and still does not call it confirmed.")
    return {"event_id": event_of(ev["window_id"]), "window_id": ev["window_id"], "true_label": ev["true_label"], "true_bus": ev["true_bus"], "fault_type": ev["fault_type"],
            "resistance_ohm": ev["resistance_ohm"], "load_multiplier": ev["load_multiplier"], "k": o["k"], "mask_seed": ev["mask_seed"], "dropped_pmu_nodes": o["dropped_nodes"],
            "stratum": ev["stratum"], "slot": ev["slot"], "prediction_correct": ev["prediction_correct"], "truth_in_candidate_set": ev["truth_in_candidate_set"],
            "event_object": o, "golden_passages": pas, "reference_report_points": auto + judge,
            "split": "test", "status": "draft", "created": TODAY, "created_by": "agent:claude-sonnet-5-5", "reviewed": None,
            "review": {"by": "author", "date": TODAY, "round": "O36", "verdict": "author instruction 2026-11-02: reference points derived from the event object have status reviewed; "
                       "judgement points and golden passages stay draft", "changes": []}, "golden_version": 1}


# ---- main ---------------------------------------------------------------------------------------------------------------------------
def build(index: int | None = None) -> tuple[list[dict], dict]:
    import pandas as pd
    from gridsense.model.latent_repr import represent
    from gridsense.model.predictor import checkpoint_sha256
    from fault_localization.config.constants import FAULT_BUSES

    cfg = load_config()
    index = index or int(cfg["splits"]["report_window"]["chosen_index"])
    docs = {d.document_id: d for d in load_corpus()}
    d = LAT / "test"
    ids = (d / "window_ids.txt").read_text().split()
    rows = pool_rows(ids, index)
    pool_ids = [ids[r] for r in rows]
    y = np.load(d / "labels.npy").astype(int)[rows]
    assert (y > 0).all(), "the report window must be post-fault"
    votes = {k: np.load(d / f"votes_k{k}_o0.npy", mmap_mode="r") for k in KS}
    gate_p, cs_p = TRACK / "configs" / "gate_v1.json", TRACK / "configs" / "candidate_sets_v1.json"
    gate, cs = Gate.load(gate_p), CandidateSets.load(cs_p)
    cls = classify(gate, cs, {k: np.asarray(votes[k][rows]) for k in KS}, y)
    types = {event_of(w): event_of(w).split("_")[1] for w in pool_ids}
    chosen = pick(pool_ids, y, cls, types)
    sel_rows = [int(rows[c["row"]]) for c in chosen]
    meta = pd.read_csv(DATA_ROOT / "processed" / "metadata.csv", usecols=["window_id", "fault_type", "resistance", "load_multiplier"]).set_index("window_id")
    # neighbours
    qv = np.stack([represent(np.asarray(np.load(d / f"h_k{c['k']}_o0.npy", mmap_mode="r")[r])[None], "b_mean_max")[0] if c["k"] == 0 else np.asarray(np.load(d / f"h_k{c['k']}_o0.npy", mmap_mode="r")[r])
                   for c, r in zip(chosen, sel_rows)])
    sc, ix, s_ids, s_y, s_ok, s_scn = neighbours(qv, [scenario_of(event_of(ids[r])) for r in sel_rows])
    test_runs = {event_of(w) for w in ids}
    replay = replay_masks(sel_rows, [c["k"] for c in chosen], len(ids), ids, {k: votes[k] for k in KS})
    node_bus = {int(i): int(b) for b, i in json.loads((DATA_ROOT / "processed" / "node_mapping.json").read_text()).items()}
    ck = checkpoint_sha256(SEED_MODEL)
    items = []
    for n, (c, r, nb, rp) in enumerate(zip(chosen, sel_rows, range(len(chosen)), replay), start=1):
        k, j = c["k"], c["row"]
        cl = cls[k]
        wid = ids[r]
        pred = int(cl["pred"][j])
        pred_bus = None if pred == 0 else int(FAULT_BUSES[pred - 1])
        neigh = []
        for t in range(K_NN):
            s = int(ix[nb, t])
            assert event_of(s_ids[s]) not in test_runs and s_scn[s] != scenario_of(event_of(wid))      # held-out, train only
            neigh.append({"window_id": s_ids[s], "class": int(s_y[s]), "bus": int(FAULT_BUSES[s_y[s] - 1]) if s_y[s] else 0, "similarity": round(float(sc[nb, t]), 4), "store_prediction_correct": bool(s_ok[nb, t])})
        agree = float(np.mean([m["class"] == pred for m in neigh]))
        low = int(cl["outcome"][j]) == 1
        cset = [int(FAULT_BUSES[q - 1]) for q in np.flatnonzero(cl["set"][j]) if q > 0] if low else None
        pm = meta.loc[wid]
        dropped = rp["dropped_nodes"]
        obj = {"window_id": wid, "report_window_index": index, "k": k, "dropped_nodes": dropped, "dropped_pmu_buses": [node_bus[x] for x in dropped],
               "predicted_class": pred, "predicted_bus": pred_bus, "top1_vote_share": round(float(cl["top1_share"][j]), 4), "calibrated_probability": round(float(cl["p"][j]), 4),
               "margin": round(float(cl["margin"][j]), 4), "gate_outcome": OUTCOME_NAMES[int(cl["outcome"][j])], "reason_codes": reasons_to_names(int(cl["reason"][j])),
               "candidate_set_buses": cset, "candidate_set_includes_no_fault": bool(cl["set"][j][0]) if low else None,
               "nearest_pmu_buses_of_predicted_bus": nearest_pmus(docs, pred_bus) if pred_bus else [],
               "knn": {"held_out": True, "k_nn": K_NN, "agreement": round(agree, 3), "neighbours": neigh},
               "provenance": {"model_seed": SEED_MODEL, "checkpoint_sha256": ck, "gate_config_sha256": gate.cfg.get("config_sha256"), "candidate_sets_sha256": cs.cfg.get("config_sha256")}}
        ev = {"window_id": wid, "true_label": int(y[j]), "true_bus": int(FAULT_BUSES[y[j] - 1]), "fault_type": str(pm["fault_type"]), "resistance_ohm": float(pm["resistance"]),
              "load_multiplier": round(float(pm["load_multiplier"]), 6), "mask_seed": rp["mask_seed"] if k else None, "stratum": c["stratum"], "slot": c["slot"],
              "prediction_correct": bool(cl["correct"][j]), "truth_in_candidate_set": (bool(cl["set"][j][y[j]]) if low else None), "event_object": obj}
        it = build_item(n, ev, docs)
        it["replay"] = {"batch_idx": rp["batch_idx"], "position_in_batch": rp["position"], "max_vote_diff_vs_stored": round(rp["max_vote_diff"], 7)}
        items.append(it)
    info = {"seed": SEED, "report_window_index": index, "pool_windows": int(len(rows)), "p_high": P_HIGH, "per_slot": PER_SLOT, "k_nn": K_NN,
            "max_vote_diff": max(i["replay"]["max_vote_diff_vs_stored"] for i in items)}
    return items, info


def validate(items: list[dict]) -> list[str]:
    bad = []
    if len(items) != 60: bad.append(f"{len(items)} items, expected 60")
    if len({i["event_id"] for i in items}) != len(items): bad.append("an event occurs twice")
    for i in items:
        o = i["event_object"]
        if i["split"] != "test" or window_index(i["window_id"]) != o["report_window_index"]: bad.append(f"{i['event_id']}: split or window index")
        if (o["gate_outcome"] == "LOW_CONFIDENCE") != (o["candidate_set_buses"] is not None): bad.append(f"{i['event_id']}: candidate set only for LOW_CONFIDENCE")
        if len(o["dropped_nodes"]) != o["k"]: bad.append(f"{i['event_id']}: dropped nodes != k")
        if i["replay"]["max_vote_diff_vs_stored"] > 1e-3: bad.append(f"{i['event_id']}: replayed votes differ from the stored votes")
        for p in i["reference_report_points"]:
            if (p["derived_from"] == "event_object") != (p["status"] == "reviewed"): bad.append(f"{p['point_id']}: status does not follow the origin")
    return bad


def write(items: list[dict], info: dict) -> None:
    OUT_JSONL.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "judgement_points_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "k", "stratum", "point_id", "kind", "text", "verdict (accept/edit/reject)", "edit"])
        for i in items:
            for p in i["reference_report_points"]:
                if p["status"] == "draft" and p["kind"] not in FIXED_KINDS:
                    w.writerow([i["event_id"], i["k"], i["stratum"], p["point_id"], p["kind"], p["text"], "", ""])
    fixed = {}
    for i in items:
        for p in i["reference_report_points"]:
            if p["kind"] in FIXED_KINDS:
                fixed.setdefault(p["kind"], {"text": p["text"], "ids": []})["ids"].append(p["point_id"])
    (OUT_DIR / "fixed_statements.md").write_text("# The 3 fixed judgement statements (status draft until the author approves each once)\n\n"
        "Each statement is the same text in all 60 events (180 copies). After approval the copies are set to `reviewed`. Passages for them: note_25 (not confirmation), note_30 (aggregate), note_28 (limits).\n\n"
        + "\n".join(f"{n}. **{k}** ({len(v['ids'])} copies): \"{v['text']}\"" for n, (k, v) in enumerate(fixed.items(), 1)) + "\n", encoding="utf-8")
    (OUT_DIR / "s4_selection.json").write_text(json.dumps(select_s4(items), indent=1, ensure_ascii=False) + "\n")
    (OUT_DIR / "EVENTS_SUMMARY.md").write_text(summary_md(items, info), encoding="utf-8")
    (OUT_DIR / "selection.json").write_text(json.dumps({**info, "events": [{"event_id": i["event_id"], "k": i["k"], "slot": i["slot"]} for i in items],
                                                         "sha256_of_jsonl": hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()}, indent=1) + "\n")


def select_s4(items: list[dict]) -> dict:
    """30 events for S4: 2 per cell (k x stratum; no REJECT event exists, so the cell is k x stratum). Seeded inside each cell; events sorted by id first. No hand selection."""
    chosen, cells = [], []
    for k in KS:
        for si, st in enumerate(STRATA):
            pool = sorted((i["event_id"] for i in items if i["k"] == k and i["stratum"] == st))
            seed = S4_SEED_BASE + 10 * k + si
            pick_ix = sorted(np.random.default_rng(seed).choice(len(pool), size=S4_PER_CELL, replace=False).tolist())
            cells.append({"k": k, "stratum": st, "seed": seed, "pool": pool, "selected": [pool[j] for j in pick_ix]})
            chosen += [pool[j] for j in pick_ix]
    return {"rule": "2 events per cell, cell = k x stratum (15 cells); numpy default_rng(seed) inside each cell on the sorted event IDs", "seed_base": S4_SEED_BASE, "cells": cells, "selected": chosen}


def summary_md(items: list[dict], info: dict) -> str:
    pts = [p for i in items for p in i["reference_report_points"]]
    n_rev, n_dr = sum(p["status"] == "reviewed" for p in pts), sum(p["status"] == "draft" for p in pts)
    strata = ["confident_correct", "high_conf_error", "low_confidence"]
    L = ["# Golden events, draft (O36)", "",
         "Written for: the author. Generated by `gridsense.events.draft_events` (seed 36, deterministic: two runs give the same file hash). Items: `data/eval/golden_events.jsonl`. Review sheet: `judgement_points_review.csv`. "
         "No new model run for the selection and no API call; CPU, 2 threads.", "",
         f"## What was drafted\n{len(items)} events from the **test split** at the report window w = {info['report_window_index']} (one window per run; {info['pool_windows']} runs in the pool), checkpoint seed 42, mask offset 0. "
         f"Per k in 0, 1, 2, 5, 10: 4 CONFIDENT and right, 4 CONFIDENT and wrong with calibrated p >= {info['p_high']} (my threshold for \"high probability\"; you can change it), "
         "4 LOW_CONFIDENCE (2 right, 2 wrong). The picker is a seeded greedy: for each slot it takes a window of the least used fault type, then the least used fault bus; one event per run. "
         "REJECT events are not drafted (none occurs in this pool).", "",
         "## Coverage", "| k | " + " | ".join(strata) + " |", "|---|---|---|---|"]
    for k in KS:
        L.append(f"| {k} | " + " | ".join(str(sum(i["k"] == k and i["stratum"] == s for i in items)) for s in strata) + " |")
    ft = Counter(i["fault_type"] for i in items)
    L += ["", "Fault types (11 of 11): " + ", ".join(f"{t} {n}" for t, n in sorted(ft.items())) + ".",
          f"Fault buses: {len({i['true_bus'] for i in items})} of 25 (2 or 3 events each); {sum(i['true_bus'] in (13, 18, 25, 35, 47, 97) for i in items)} events lie on the 6 μPMU buses.",
          "Fault resistance: " + ", ".join(f"{r:g} Ω {n}" for r, n in sorted(Counter(i['resistance_ohm'] for i in items).items())) + " (follows where the model is confident or wrong; not balanced on purpose).",
          f"Top-1 right: {sum(i['prediction_correct'] for i in items)} of {len(items)}. LOW_CONFIDENCE events whose candidate set holds the true bus: "
          f"{sum(bool(i['truth_in_candidate_set']) for i in items if i['stratum'] == 'low_confidence')} of {sum(i['stratum'] == 'low_confidence' for i in items)}. "
          f"Events where a dark μPMU is one of the 3 nearest μPMUs of the predicted bus: {sum(any(p['kind'] == 'nearest_dark' for p in i['reference_report_points']) for i in items)}.", "",
          "kNN agreement (10 held-out neighbours; information only, because tau_k = 0 at the 95 % point and the gate ignores it):", "| stratum | median | events with agreement < 0.5 |", "|---|---|---|"]
    for s in strata:
        a = [i["event_object"]["knn"]["agreement"] for i in items if i["stratum"] == s]
        L.append(f"| {s} | {float(np.median(a)):.2f} | {sum(x < 0.5 for x in a)} of {len(a)} |")
    L += ["", "## Checks", f"- Mask replay: for all {len(items)} windows the paper's `apply_sensor_dropout_batch` was run on the real batch (sampler order equals the stored order) and the model's soft vote was compared with the stored vote: "
          f"largest difference {info['max_vote_diff']:.1e}. So the dark nodes of each event are the ones the stored votes were made with.",
          "- Neighbours: held-out (own scenario removed), train windows only; the code asserts that no neighbour comes from a test run.",
          "- Each golden passage is a sentence that occurs exactly once in its note (checked at build time). Notes used: 24, 25, 23, 30, 28; with LOW_CONFIDENCE 22, 09; with dark μPMUs 12, and 16 and 06 when a nearest μPMU is dark.", "",
          "## Status of the reference points (your instruction)", f"- {n_rev} points come from the event object (location, gate outcome and reason codes, candidate set, neighbour locations, dark μPMUs): status `reviewed`, `derived_from: event_object`.",
          "- Judgement points by kind: " + ", ".join(f"{k} {n}" for k, n in Counter(p["kind"] for p in pts if p["status"] == "draft").most_common()) + f". The {sum(p['status'] == 'draft' and p['kind'] in FIXED_KINDS for p in pts)} points of the kinds {', '.join(FIXED_KINDS)} are 3 fixed statements (`fixed_statements.md`); the author approves each once. The other {sum(p['status'] == 'draft' and p['kind'] not in FIXED_KINDS for p in pts)} are in `judgement_points_review.csv`.",
          f"- {n_dr} judgement points (not confirmation, aggregate metric, simulation limit, ambiguity, nearest μPMU dark, neighbour disagreement) and all golden passages: status `draft`. The events themselves stay `draft` until you review these.",
          "- The truth (`true_bus`, `prediction_correct`, `truth_in_candidate_set`) is kept outside `event_object`; the generator never sees it.", "",
          "## Open for you", "1. Review `judgement_points_review.csv` (accept, edit or reject each point) and the golden passages.", "2. Approve the 3 fixed statements; then their copies are set to `reviewed`.",
          "3. \"High probability\" = calibrated p >= 0.9 (accepted by the author 2026-11-02).", "4. S4 selection: `s4_selection.json` (30 events, 2 per k x stratum cell, seeded per cell; no REJECT event exists, so the cell is k x stratum)."]
    return "\n".join(L) + "\n"


def main() -> None:
    items, info = build()
    problems = validate(items)
    if problems:
        raise SystemExit("\n".join(problems))
    write(items, info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
