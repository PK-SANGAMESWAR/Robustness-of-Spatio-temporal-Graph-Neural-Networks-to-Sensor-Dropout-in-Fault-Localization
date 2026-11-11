"""LLM judge for S4: DeepEval metrics on top of our own client, so that every judge call is priced, cached, budgeted and approval-gated.

Per report (16 calls in the cost estimate): faithfulness, answer relevancy, contextual relevancy and the five GEval rubrics (`rubrics/*.md`).
Context given to the judge: faithfulness and the rubrics read the EVENT FACTS plus the retrieved passages (the report states facts of the event that no passage holds);
contextual relevancy reads the retrieved passages only (it measures retrieval). A metric that raises is a failure with its cause; it is never scored 0 or 1.
`status: draft` rubrics are refused for a live run (`llm.require_live_approval`); the hashes of the rubric files enter the snapshot.
"""
from __future__ import annotations

import json
import re
from typing import Any

from gridsense import settings  # noqa: F401
from gridsense.config import TRACK

RUBRICS = ("prediction_vs_confirmed", "completeness", "adherence", "evidence_use", "no_aggregate_metric_as_confidence")


def parse_rubric(name: str) -> dict:
    t = (TRACK / "rubrics" / f"{name}.md").read_text(encoding="utf-8")
    crit = t.split("## Criteria", 1)[1].split("## Score scale", 1)[0]
    steps = [re.sub(r"^\s*\d+\.\s*", "", l).strip() for l in crit.splitlines() if re.match(r"^\s*\d+\.", l)]
    scale = t.split("## Score scale", 1)[1].split("## Good example", 1)[0]
    rows = [(int(a), b.strip()) for a, b in re.findall(r"^\|\s*(\d)\s*\|\s*(.+?)\s*\|\s*$", scale, flags=re.M)]
    return {"name": name, "steps": steps, "scale": rows}


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if not m:
            raise ValueError("judge reply holds no JSON")
        return json.loads(m.group(1))


def make_adapter(client, salt: str = "main"):
    from deepeval.models import DeepEvalBaseLLM

    class Adapter(DeepEvalBaseLLM):
        def __init__(self):
            self.client, self.salt, self.usage = client, salt, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
            super().__init__(model="judge")

        def load_model(self, *a, **k):
            return self

        def get_model_name(self, *a, **k) -> str:
            return getattr(self.client, "model", "judge")

        def supports_json_mode(self):
            return True

        def supports_structured_outputs(self):
            return False

        def supports_temperature(self):
            return True

        def generate(self, prompt: str, schema=None, **kw):
            r = self.client.complete([{"role": "user", "content": prompt}], purpose="judge", json_mode=True, max_tokens=500, salt=self.salt)
            self.usage["calls"] += 1; self.usage["input_tokens"] += r.input_tokens; self.usage["output_tokens"] += r.output_tokens; self.usage["usd"] += r.usd
            if schema is None:
                return r.text
            return schema.model_validate(_extract_json(r.text))

        async def a_generate(self, prompt: str, schema=None, **kw):
            return self.generate(prompt, schema=schema)

    return Adapter()


def run_judge(client, report_text: str, question: str, event_facts: str, passages: list[str], salt: str = "main") -> dict:
    """Scores for one report. Returns {"scores": {metric: {"score", "reason"}}, "failures": {metric: cause}, "usage": {...}}."""
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.metrics.g_eval.utils import Rubric
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    m = make_adapter(client, salt)
    full_ctx = [event_facts, *passages]
    cases = {"faithfulness": LLMTestCase(input=question, actual_output=report_text, retrieval_context=full_ctx),
             "answer_relevancy": LLMTestCase(input=question, actual_output=report_text, retrieval_context=full_ctx),
             "contextual_relevancy": LLMTestCase(input=question, actual_output=report_text, retrieval_context=passages or ["(no passages)"])}
    metrics: dict[str, Any] = {"faithfulness": FaithfulnessMetric(model=m, async_mode=False, include_reason=True),
                               "answer_relevancy": AnswerRelevancyMetric(model=m, async_mode=False, include_reason=True),
                               "contextual_relevancy": ContextualRelevancyMetric(model=m, async_mode=False, include_reason=True)}
    for r in RUBRICS:
        p = parse_rubric(r)
        rub = [Rubric(score_range=(s, s), expected_outcome=text) for s, text in sorted(p["scale"])]
        metrics[f"geval_{r}"] = GEval(name=r, evaluation_steps=p["steps"], rubric=rub, model=m, async_mode=False, evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT])
        cases[f"geval_{r}"] = LLMTestCase(input=question, actual_output=report_text, retrieval_context=full_ctx)
    scores, failures = {}, {}
    for name, metric in metrics.items():
        try:
            metric.measure(cases[name])
            scores[name] = {"score": float(metric.score), "reason": str(getattr(metric, "reason", ""))[:400]}
            if name.startswith("geval_"):            # DeepEval maps the rubric range 1..5 to 0..1: (raw - 1) / 4; the raw 1..5 score is kept for reading
                scores[name]["raw_1_to_5"] = round(1 + 4 * float(metric.score), 2)
        except Exception as e:                                                        # a failure with a cause, never a score
            failures[name] = f"{type(e).__name__}: {str(e)[:200]}"
    return {"scores": scores, "failures": failures, "usage": m.usage}


# ---- mock judge (dry runs and tests): a valid reply for whatever schema DeepEval asks for -------------------------------------------
def default_for_schema(schema) -> dict:
    """A minimal valid object for a pydantic schema: empty lists, short strings, middling numbers. Verdict lists are empty, so a metric computes its empty-case score."""
    out = {}
    for name, f in schema.model_fields.items():
        ann = str(f.annotation)
        if "List" in ann or "list" in ann:
            out[name] = []
        elif "int" in ann:
            out[name] = 3
        elif "float" in ann:
            out[name] = 0.5
        elif "bool" in ann:
            out[name] = True
        else:
            out[name] = "mock reason"
    return out


class MockJudgeResponder:
    """Callable for MockClient. One reply that holds every key any DeepEval parser asks for (extra keys are ignored), so every metric runs end to end at zero cost."""

    def __init__(self, geval_score: int = 4):
        self.geval_score = geval_score

    def __call__(self, messages: list[dict], purpose: str) -> str:
        return json.dumps({"truths": [], "claims": [], "statements": [], "verdicts": [], "reason": "mock reason", "score": self.geval_score})
