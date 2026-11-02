"""Cost ledger for every paid call of the track (IMPLEMENTATION.md § 7; O9: 5 USD in total; rag-eval.md § 10.2).

Prices come only from configs/eval.yaml `cost.prices` = {model: {input_per_1m, output_per_1m, date}}, given by the author.
Never a remembered price: a model without a price is refused. `reserve` refuses a call that would take the total above
`cost.limit_usd_total`; `commit` appends the actual usage to eval/cost_ledger.jsonl (append-only).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gridsense.config import ROOT, load_config


class CostLimitError(RuntimeError):
    pass


class MissingPriceError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path | None = None, prices: dict | None = None, limit_usd: float | None = None, label: str = ""):
        cfg = load_config()["cost"]
        self.path = Path(path) if path else ROOT / cfg["ledger"]
        self.prices = prices if prices is not None else (cfg.get("prices") or {})
        self.limit = float(limit_usd if limit_usd is not None else cfg["limit_usd_total"])
        self.label = label
        self._reserved = 0.0

    def price(self, model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
        p = self.prices.get(model)
        if not p or p.get("input_per_1m") is None or not p.get("date"):
            raise MissingPriceError(f"no dated price for {model!r} in configs/eval.yaml cost.prices; ask the author (O1)")
        if batch:
            if p.get("batch_input_per_1m") is None:
                raise MissingPriceError(f"no Batch price for {model!r}: the call must use the standard price or the author must give the Batch price")
            return input_tokens / 1e6 * float(p["batch_input_per_1m"]) + output_tokens / 1e6 * float(p.get("batch_output_per_1m") or 0.0)
        return input_tokens / 1e6 * float(p["input_per_1m"]) + output_tokens / 1e6 * float(p.get("output_per_1m") or 0.0)

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.path.read_text().splitlines() if line.strip())

    def reserve(self, model: str, input_tokens: int, output_tokens: int, purpose: str, batch: bool = False) -> float:
        usd = self.price(model, input_tokens, output_tokens, batch)
        if self.spent() + self._reserved + usd > self.limit:
            raise CostLimitError(f"{purpose}: {usd:.6f} USD would exceed the track limit {self.limit:.2f} USD "
                                 f"(spent {self.spent():.6f}); stop and ask the author")
        self._reserved += usd
        return usd

    def commit(self, model: str, input_tokens: int, output_tokens: int, purpose: str, batch: bool = False) -> float:
        usd = self.price(model, input_tokens, output_tokens, batch)
        self._reserved = max(0.0, self._reserved - usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "label": self.label, "purpose": purpose,
               "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens, "usd": usd, "batch": batch,
               "price_date": str(self.prices[model]["date"])}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return usd
