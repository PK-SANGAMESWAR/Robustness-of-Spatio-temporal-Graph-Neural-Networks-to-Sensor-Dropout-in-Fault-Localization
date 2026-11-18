"""LLM clients for S4 (S4_PLAN.md). Three rules are enforced in code, not by habit:

1. A LIVE call needs `rag-for-dropout/s4_approval.json` written by the author (rubrics, SOP decision, cost). Without it `OpenAIClient` raises `ApprovalMissing` before any network call.
2. Every live call is priced from configs/eval.yaml (`Ledger`), reserved before the call and committed with the real usage after it. The run budget is the approved cost; the track limit is 5 USD.
3. Keys come only from `gridsense.settings`. No key, prompt text or path is logged.

`MockClient` makes no network call, books nothing and returns scripted text: it is for dry runs and tests.
The response cache key holds the model, the temperature, the messages, the JSON flag and a `salt` (so that repeat runs for the noise estimate are not served from the cache of the first run).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from gridsense import settings  # noqa: F401  (loads .env, switches telemetry off; must come first)
from gridsense.config import ROOT, TRACK, load_config
from gridsense.cost import CostLimitError, Ledger

APPROVAL = TRACK / "s4_approval.json"
CACHE = TRACK / "eval" / "llm_cache"
APPROVED_RUBRIC_STATUS = ("reviewed", "approved")
INPUT_MARGIN = 1.15


class ApprovalMissing(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float = 0.0
    cached: bool = False
    temperature_sent: float | None = None
    model: str = ""
    latency_s: float = 0.0           # wall time of the live API call(s); 0 for a cached or mock reply (Layer 6 excludes those)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], *, purpose: str, json_mode: bool = True, max_tokens: int = 900, salt: str = "main") -> Reply: ...


def count_tokens(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))            # proxy tokenizer (the gpt-6 tokenizer is not known here)


def load_approval(path: Path | None = None) -> dict:
    """The author's approval for the live run. Needs: approved_by == 'author', date, rubrics_approved, sop_decision ('keep'|'exclude'), cost_approved_usd > 0."""
    path = Path(path) if path else APPROVAL                      # read the module value at call time (tests replace it)
    if not path.exists():
        raise ApprovalMissing(f"{path.name} does not exist: the author has not approved the S4 live run (rubrics O38, SOP O42, cost). No call was made")
    a = json.loads(path.read_text())
    problems = [k for k, ok in {"approved_by": a.get("approved_by") == "author", "date": bool(a.get("date")), "rubrics_approved": a.get("rubrics_approved") is True,
                                "sop_decision": a.get("sop_decision") in ("keep", "exclude"), "cost_approved_usd": float(a.get("cost_approved_usd", 0)) > 0}.items() if not ok]
    if problems:
        raise ApprovalMissing(f"s4_approval.json is incomplete or not from the author: {problems}")
    return a


def rubrics_status(rubric_dir: Path = TRACK / "rubrics") -> dict[str, str]:
    import re
    out = {}
    for p in sorted(rubric_dir.glob("*.md")):
        m = re.search(r"^status:\s*(\S+)", p.read_text(encoding="utf-8"), flags=re.M)
        out[p.stem] = m.group(1) if m else "missing"
    return out


def require_live_approval() -> dict:
    a = load_approval()
    bad = {k: v for k, v in rubrics_status().items() if v not in APPROVED_RUBRIC_STATUS}
    if bad:
        raise ApprovalMissing(f"rubrics not approved (status must be reviewed/approved): {bad}")
    return a


def _key(model: str, temperature, messages: list[dict], json_mode: bool, salt: str, effort=None, max_tokens=None) -> str:
    return hashlib.sha256(json.dumps([model, temperature, messages, json_mode, salt, effort, max_tokens], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class Budget:
    """One budget shared by the generator and the judge client of a run (the approved cost)."""
    limit_usd: float
    spent_usd: float = 0.0


@dataclass
class MockClient:
    """Scripted client. `responder(messages, purpose) -> str`. Books nothing; counts tokens so that a dry run shows the size of the real run."""
    responder: Callable[[list[dict], str], str]
    model: str = "mock"
    calls: list = field(default_factory=list)

    def complete(self, messages, *, purpose, json_mode=True, max_tokens=900, salt="main") -> Reply:
        text = self.responder(messages, purpose)
        r = Reply(text, count_tokens("\n".join(m["content"] for m in messages)), count_tokens(text), 0.0, False, 0.0, self.model)
        self.calls.append((purpose, r.input_tokens, r.output_tokens))
        return r


class OpenAIClient:
    """Live client: standard (synchronous) prices. The Batch API is not used: the DeepEval judge steps depend on each other (see S4_PLAN.md)."""

    def __init__(self, model: str, temperature: float | None, ledger: Ledger, budget: Budget, label: str, cache: bool = True, reasoning_effort: str | None = None):
        require_live_approval()
        from openai import OpenAI
        self.model, self.temperature, self.ledger, self.budget, self.label, self.use_cache = model, temperature, ledger, budget, label, cache
        self.effort = reasoning_effort      # the gpt-6 models reason before they answer; the reasoning tokens count as output tokens (and as max_completion_tokens)
        self.client = OpenAI(api_key=settings.secret("OPENAI_API_KEY"), max_retries=2)
        self.temperature_refused = self._marker().exists()          # a model that once refused a temperature is not asked again: the cache keys then stay the same from the first call on
        self.default_temperature_note = "model refused an explicit temperature earlier; the model default is used (O35)" if self.temperature_refused else None

    def _marker(self) -> Path:
        return CACHE / self.model / "_temperature_refused"

    def _cache_path(self, key: str) -> Path:
        return CACHE / self.model / f"{key}.json"

    def complete(self, messages, *, purpose, json_mode=True, max_tokens=900, salt="main") -> Reply:
        temp = None if self.temperature_refused else self.temperature
        key = _key(self.model, temp, messages, json_mode, salt, self.effort, max_tokens)
        p = self._cache_path(key)
        if self.use_cache and p.exists():
            d = json.loads(p.read_text())
            return Reply(d["text"], d["input_tokens"], d["output_tokens"], 0.0, True, d.get("temperature_sent"), self.model)
        est_in = int(count_tokens("\n".join(m["content"] for m in messages)) * INPUT_MARGIN)
        est = self.ledger.price(self.model, est_in, max_tokens)
        if self.budget.spent_usd + est > self.budget.limit_usd:
            raise CostLimitError(f"{purpose}: run budget {self.budget.limit_usd:.2f} USD would be exceeded (spent {self.budget.spent_usd:.4f}); stop and ask the author")
        self.ledger.reserve(self.model, est_in, max_tokens, purpose)
        kw = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if temp is not None:
            kw["temperature"] = temp
        if self.effort:
            kw["reasoning_effort"] = self.effort
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kw)
        except Exception as e:                                                   # a model that refuses a temperature: send none, record the default (O35)
            if temp is not None and "temperature" in str(e).lower():
                self.temperature_refused = True
                self._marker().parent.mkdir(parents=True, exist_ok=True)
                self._marker().write_text("the model refused an explicit temperature\n")
                self.default_temperature_note = "model refused an explicit temperature; the model default was used (O35)"
                kw.pop("temperature")
                resp = self.client.chat.completions.create(**kw)
                temp = None
            else:
                self.ledger._reserved = max(0.0, self.ledger._reserved - est)
                raise
        latency = time.perf_counter() - t0
        u = resp.usage
        usd = self.ledger.commit(self.model, u.prompt_tokens, u.completion_tokens, f"{self.label}:{purpose}")
        self.budget.spent_usd += usd
        text = resp.choices[0].message.content or ""
        if self.use_cache and text.strip():                                       # an empty reply is never cached
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"text": text, "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens, "temperature_sent": temp}))
        return Reply(text, u.prompt_tokens, u.completion_tokens, usd, False, temp, self.model, latency)


def make_clients(live: bool, label: str, mock_gen: Callable | None = None, mock_judge: Callable | None = None, repeat_budget: float | None = None):
    """(generator client, judge client). live=False returns mocks. live=True needs the author's approval file and approved rubrics."""
    cfg = load_config()
    if not live:
        return MockClient(mock_gen, cfg["generation"]["model"] + " (mock)"), MockClient(mock_judge, cfg["judge"]["model"] + " (mock)")
    a = require_live_approval()
    led = Ledger(label=label)
    budget = Budget(min(float(a["cost_approved_usd"]), float(repeat_budget) if repeat_budget else 1e9))
    return (OpenAIClient(cfg["generation"]["model"], cfg["generation"].get("temperature"), led, budget, label, reasoning_effort=cfg["generation"].get("reasoning_effort")),
            OpenAIClient(cfg["judge"]["model"], cfg["judge"].get("temperature"), led, budget, label, reasoning_effort=cfg["judge"].get("reasoning_effort")))
