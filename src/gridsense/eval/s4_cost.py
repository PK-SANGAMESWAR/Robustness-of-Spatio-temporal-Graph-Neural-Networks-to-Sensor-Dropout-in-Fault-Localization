"""S4 cost estimate (rag-cost-check). NO API call: it renders the draft generator prompt and the real DeepEval judge templates with realistic inputs, counts tokens locally and prices them.

    PYTHONPATH=src:. .venv/bin/python -m gridsense.eval.s4_cost

Run set (author, 2026-11-02): deterministic checks on all events (cost 0); LLM-judged metrics on 30 events, one run; 3 repeat runs on 10 events for the noise estimate (counted as 30 additional
generated and judged reports, the cautious reading); the evidence-status check on every golden retrieval query. Only retrieved chunks and the structured event fields are sent.
Judge metrics per report (DeepEval templates, steps given for the GEval rubrics): faithfulness 4 calls, answer relevancy 3, contextual relevancy 3 + 1, five GEval rubrics 5 = 16 calls.
Assumptions (stated, not hidden): tokenizer = tiktoken o200k_base as a proxy (the tokenizer of the gpt-6 models is not known here; the ceiling adds 15 % to every input count);
outputs: the expected value is an assumed typical size per call type, the ceiling is its cap; the events are fixtures built from real retrieved passages, since the golden events do not exist yet.
Batch API: priced at the confirmed Batch rates (50 % of standard); no discount for the response cache is assumed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jinja2
import tiktoken

from gridsense.config import ROOT, TRACK, load_config
from gridsense.cost import Ledger
from gridsense.eval.precision_check import load_run

DE = ROOT / ".venv/lib/python3.12/site-packages/deepeval/metrics"
ENC = tiktoken.get_encoding("o200k_base")
N_EVENTS, N_NOISE_EVENTS, N_NOISE_RUNS = 30, 10, 3
INPUT_MARGIN = 1.15

SAMPLE_REPORT = """Event 7_ABCG_10p0_0014, window w0025 (5 post-fault samples). Predicted location: fault bus 57. This is a candidate and not a confirmed fault.
Gate outcome: LOW_CONFIDENCE (reason code LOW_MARGIN). Candidate set: buses 53, 55, 57 and 62.
Sensor coverage: 5 of 25 μPMUs are dark (k = 5): the μPMUs at buses 36, 45, 54, 60 and 69. Two of them are among the μPMUs nearest to bus 57 on the operated feeder.
Case evidence: 6 of the 10 most similar simulated training windows had true fault bus 57, 3 had bus 55 and 1 had bus 53. The neighbours agree with the prediction only in part.
Document evidence: Targeted loss of the sensors nearest to the fault is harder than random loss (note_16_targeted_versus_random_dropout, "Targeted dropout is harder than random dropout").
The gate was fitted for random dropout and returns LOW_CONFIDENCE when its margin condition fails (note_24_gate_outcomes). Test-set averages of the model, such as its macro-F1, are not the confidence of this
prediction (note_30_aggregate_versus_individual).
Limitations: The data are simulated and the documents are curated for research, not approved by a utility. No check for a wrong reading from a present μPMU was run, so such a reading cannot be excluded.
Recommended operator review: Treat buses 53, 55, 57 and 62 as candidates, read the cited sections, and confirm the location with the processes that the utility authorises before any action.
Evidence status: sufficient. Approval status: pending_human_review."""

# assumed typical output tokens per call type, and the cap that the call would carry
OUT = {"generator": (None, 900), "evidence_check": (45, 120), "f_truths": (200, 500), "f_claims": (200, 500), "f_verdicts": (200, 500), "f_reason": (70, 200),
       "a_statements": (150, 400), "a_verdicts": (150, 400), "a_reason": (60, 200), "c_verdicts": (180, 450), "c_reason": (70, 200), "g_result": (90, 250)}


def n(text: str) -> int:
    return len(ENC.encode(text))


def sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 15]


def split_prompt(path: Path) -> tuple[str, str]:
    body = path.read_text(encoding="utf-8").split("\n---\n", 1)[1]
    sys_, user = body.split("## User", 1)
    return sys_.replace("## System", "").strip(), user.strip()


def render(rel: str, **kw) -> str:
    return jinja2.Template((DE / rel).read_text(encoding="utf-8")).render(multimodal=False, multimodal_instruction="", _fragments={}, limit="", _additional_context="", **kw)


def rubric_parts(name: str) -> tuple[str, str]:
    t = (TRACK / "rubrics" / f"{name}.md").read_text(encoding="utf-8")
    crit = t.split("## Criteria", 1)[1].split("## Score scale", 1)[0].strip()
    scale = t.split("## Score scale", 1)[1].split("## Good example", 1)[0].strip()
    return crit, scale


def samples() -> list[dict]:
    items, chunks = load_run("frozen_final_v2_test")
    out = []
    for it in sorted(items, key=lambda x: x["query_id"]):
        if it["split"] == "test" and it["answerable"] and len(out) < 3:
            out.append({"query": it["query_id"], "passages": [chunks[p["chunk_id"]] for p in it["retrieved"][:3]]})
    return out


def prompts_for(sample: dict, question: str) -> dict[str, list[tuple[str, int]]]:
    """token count of every call of one report (generation, 16 judge calls, evidence check) for one sample."""
    gsys, guser = split_prompt(TRACK / "prompts" / "generator_v0.md")
    esys, euser = split_prompt(TRACK / "prompts" / "evidence_check_v0.md")
    passages = "\n\n".join(f"[{c['document_id']} › {' › '.join(c['section_path'][-1:])}]\n{c['text']}" for c in sample["passages"])
    event = json.dumps({"event_id": "7_ABCG_10p0_0014", "window_id": "7_ABCG_10p0_0014_w0025", "report_window_post_fault_samples": 5, "k": 5, "dark_uPMU_buses": [36, 45, 54, 60, 69],
                        "predicted_bus": 57, "vote_share": 0.46, "calibrated_probability": 0.58, "margin": 0.11, "gate": {"outcome": "LOW_CONFIDENCE", "reason_codes": ["LOW_MARGIN"], "policy_version": "gate_v1"},
                        "candidate_set": [53, 55, 57, 62], "sensor_coverage": {"available": 20, "total": 25}, "model_version": "dropout_aware_rgsage_max_seed42"}, indent=1)
    nb = "case | true bus | predicted | correct | fault type | k\n" + "\n".join(f"c{i:05d} | {b} | {p} | {'yes' if b == p else 'no'} | {t} | 5" for i, (b, p, t) in enumerate(
        [(57, 57, "ABCG"), (57, 55, "AG"), (55, 55, "BC"), (57, 57, "CAG"), (53, 57, "AB"), (57, 57, "BG"), (55, 55, "ABC"), (57, 57, "CA"), (57, 62, "BCG"), (57, 57, "AG")]))
    schema = json.dumps({"event_id": "str", "predicted_location": "int", "prediction_status": "candidate_not_confirmed", "gate_outcome": "str", "gate_reason_codes": ["str"], "candidate_set": ["int"],
                         "sensor_coverage_summary": "str", "case_evidence_summary": "str", "document_evidence": [{"claim": "str", "document_id": "str", "section": "str", "excerpt": "str"}],
                         "limitations": ["str"], "recommended_operator_review": ["str"], "evidence_status": "sufficient|insufficient", "evidence_status_reason": "str"}, indent=1)
    gen = n(gsys) + n(guser.format(event_json=event, neighbours_table=nb, passages=passages, schema_json=schema))
    ev = n(esys) + n(euser.format(question=question, passages=passages))
    claims, truths = sentences(SAMPLE_REPORT)[:8], [s for c in sample["passages"] for s in sentences(c["text"])][:12]
    ctx = [c["text"] for c in sample["passages"]]
    j: list[tuple[str, int]] = []
    j += [("f_truths", n(render("faithfulness/templates/generate_truths.txt", retrieval_context="\n\n".join(ctx)))), ("f_claims", n(render("faithfulness/templates/generate_claims.txt", actual_output=SAMPLE_REPORT))),
          ("f_verdicts", n(render("faithfulness/templates/generate_verdicts.txt", claims=json.dumps(claims), retrieval_context="\n\n".join(truths)))),
          ("f_reason", n(render("faithfulness/templates/generate_reason.txt", contradictions=json.dumps([claims[0]]), score=0.9)))]
    j += [("a_statements", n(render("answer_relevancy/templates/generate_statements.txt", actual_output=SAMPLE_REPORT))),
          ("a_verdicts", n(render("answer_relevancy/templates/generate_verdicts.txt", input=question, statements=json.dumps(claims)))),
          ("a_reason", n(render("answer_relevancy/templates/generate_reason.txt", input=question, irrelevant_statements=json.dumps([claims[-1]]), score=0.9)))]
    j += [("c_verdicts", n(render("contextual_relevancy/templates/generate_verdicts.txt", context=c, context_type="text", empty_context_instruction="", extraction_instructions="", input=question, statement_or_image="statement"))) for c in ctx]
    j += [("c_reason", n(render("contextual_relevancy/templates/generate_reason.txt", input=question, irrelevant_statements=json.dumps(truths[:3]), relevant_statements=json.dumps(truths[3:8]), score=0.6)))]
    content = f"Input: {question}\nActual Output: {SAMPLE_REPORT}\nRetrieval Context: " + "\n\n".join(ctx)
    for r in ("prediction_vs_confirmed", "completeness", "adherence", "evidence_use", "no_aggregate_metric_as_confidence"):
        crit, scale = rubric_parts(r)
        j.append(("g_result", n(render("g_eval/templates/generate_evaluation_results.txt", evaluation_steps=crit, rubric=scale, parameters="Input, Actual Output, Retrieval Context", score_range=(1, 5), test_case_content=content))))
    return {"generator": [("generator", gen)], "evidence_check": [("evidence_check", ev)], "judge": j}


def estimate() -> dict:
    cfg = load_config()
    led = Ledger()
    gen_m, judge_m = cfg["generation"]["model"], cfg["judge"]["model"]
    ss = samples()
    q = {s["query"]: s for s in ss}
    per = [prompts_for(s, f"Draft the operator report for event {s['query']}") for s in ss]          # question text length is representative; the real text comes from the event
    mean = lambda f: sum(f(p) for p in per) / len(per)
    gen_in = mean(lambda p: p["generator"][0][1])
    ev_in = mean(lambda p: p["evidence_check"][0][1])
    judge_calls = [(t, mean(lambda p, i=i: p["judge"][i][1])) for i, (t, _) in enumerate(per[0]["judge"])]
    gen_out = (round(1.3 * n(SAMPLE_REPORT)), OUT["generator"][1])
    n_reports_main, n_reports_noise = N_EVENTS, N_NOISE_EVENTS * N_NOISE_RUNS
    n_retr = sum(1 for _ in open(ROOT / cfg["golden_sets"]["retrieval"]["path"]))
    groups = []
    def add(name, model, calls, tok_in, out_exp, out_cap):
        for batch in (False, True):
            pass
        groups.append({"group": name, "model": model, "calls": calls, "input_tokens": round(calls * tok_in), "output_expected": round(calls * out_exp), "output_cap": round(calls * out_cap)})
    add("generation, main run (30 events)", gen_m, n_reports_main, gen_in, gen_out[0], gen_out[1])
    add("generation, noise runs (10 events × 3)", gen_m, n_reports_noise, gen_in, gen_out[0], gen_out[1])
    for label, nrep in (("main run", n_reports_main), ("noise runs", n_reports_noise)):
        tin = sum(t for _, t in judge_calls); oe = sum(OUT[k][0] for k, _ in judge_calls); oc = sum(OUT[k][1] for k, _ in judge_calls)
        add(f"judge, {label} ({nrep} reports × {len(judge_calls)} calls)", judge_m, nrep * len(judge_calls), tin / len(judge_calls), oe / len(judge_calls), oc / len(judge_calls))
    add(f"evidence-status check ({n_retr} golden retrieval queries)", gen_m, n_retr, ev_in, OUT["evidence_check"][0], OUT["evidence_check"][1])
    for g in groups:
        for tag, ratio_in, out in (("expected", 1.0, g["output_expected"]), ("ceiling", INPUT_MARGIN, g["output_cap"])):
            for mode, batch in (("standard", False), ("batch", True)):
                g[f"usd_{tag}_{mode}"] = led.price(g["model"], round(g["input_tokens"] * ratio_in), out, batch)
    tot = {k: sum(g[k] for g in groups) for k in groups[0] if k.startswith("usd_")}
    return {"groups": groups, "totals": tot, "judge_calls_per_report": [(t, round(v)) for t, v in judge_calls], "gen_input_tokens": round(gen_in), "gen_output": gen_out, "evidence_input_tokens": round(ev_in),
            "spent_so_far_usd": led.spent(), "limit_usd": led.limit, "samples": list(q)}


def write(res: dict) -> Path:
    t = res["totals"]
    L = ["# S4 cost estimate (no API call has been made)", "", "Written for: the author. Generated by `gridsense.eval.s4_cost` on 2026-11-02. Prices per 1M tokens (your numbers; confirmed on the OpenAI pricing page and the luna model page on 2026-11-02, to be confirmed again before the first call): "
         "gpt-6-luna 0.10 in / 0.50 out (Batch 0.05 / 0.25); gpt-6.1-terra 1.00 in / 5.00 out (Batch 0.50 / 2.50). Only retrieved chunks and the structured event fields are sent; the corpus is never sent.", "",
         "## Run set and assumptions", "", "Deterministic checks on all events: no LLM call, cost 0. LLM-judged metrics on 30 events, one run. Noise estimate: 3 repeat runs on 10 events (30 more generated and judged reports; the cautious reading). "
         "Judge metrics per report (DeepEval templates; GEval with steps given): faithfulness 4 calls, answer relevancy 3, contextual relevancy 4, five GEval rubrics 5 = 16 calls. The evidence-status check runs once on every golden retrieval query (68 incl. drafts).", "",
         f"Input tokens are counted from the real prompts (draft generator prompt and DeepEval's own templates, 3 real retrieved passage sets, a fixture event and report; tiktoken `o200k_base` as a proxy tokenizer). Mean generator prompt {res['gen_input_tokens']:,} tokens; "
         f"expected report {res['gen_output'][0]} tokens (1.3 × the fixture report), cap {res['gen_output'][1]}. Output sizes of the judge calls are assumed (expected and cap per call type: see the code, `OUT`). The ceiling adds 15 % to every input count and sets every output to its cap. "
         "The events do not exist yet: they are fixtures.", "",
         "| group | model | calls | input tokens | output tokens (expected / cap) | standard: expected / ceiling USD | Batch: expected / ceiling USD |", "|---|---|---|---|---|---|---|"]
    for g in res["groups"]:
        L.append(f"| {g['group']} | {g['model']} | {g['calls']:,} | {g['input_tokens']:,} | {g['output_expected']:,} / {g['output_cap']:,} | {g['usd_expected_standard']:.4f} / {g['usd_ceiling_standard']:.4f} | {g['usd_expected_batch']:.4f} / {g['usd_ceiling_batch']:.4f} |")
    L += [f"| **total** | | {sum(g['calls'] for g in res['groups']):,} | {sum(g['input_tokens'] for g in res['groups']):,} | | **{t['usd_expected_standard']:.4f} / {t['usd_ceiling_standard']:.4f}** | **{t['usd_expected_batch']:.4f} / {t['usd_ceiling_batch']:.4f}** |", "",
          f"Budget: limit {res['limit_usd']:.2f} USD for the whole track; spent so far {res['spent_so_far_usd']:.6f} USD (the OpenAI embedding candidate). Even the standard-rate ceiling ({t['usd_ceiling_standard']:.2f} USD) is below the limit. "
          f"With the Batch API (where the call does not need an immediate answer: all of these) the expected cost is {t['usd_expected_batch']:.2f} USD and the ceiling {t['usd_ceiling_batch']:.2f} USD.", "",
          "Judge calls of one report, mean input tokens: " + ", ".join(f"{k} {v:,}" for k, v in res["judge_calls_per_report"]) + ".", "",
          "Not included: a judge-validation set labelled by you beyond the 30 reports (O13), retries (the client retries at most twice; the ledger counts every try), and the response cache saving (no credit is assumed). "
          "A dependency before S4 can start: the 30+ golden events do not exist; they must be drafted (seeded sample from the test split) and reviewed by you. No paid call is made before your approval."]
    p = TRACK / "reports" / "s4" / "COST_ESTIMATE.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (p.parent / "COST_ESTIMATE.json").write_text(json.dumps(res, indent=1, default=float))
    return p


if __name__ == "__main__":
    r = estimate()
    print(write(r))
    print(json.dumps({k: round(v, 4) for k, v in r["totals"].items()}, indent=1))
