---
name: rag-cost-check
description: Before any paid GridSense run, count the LLM judge, generation and embedding calls and tokens per layer, multiply by the author's dated prices, compare with the cost limit, and stop for approval.
---

# rag-cost-check

Use this skill before any run that calls an API (judge, generator, OpenAI embedding, LangSmith online judges).

1. Read `rag-for-dropout/configs/eval.yaml` `cost` and `generation` / `judge` / `embedding`. If `cost.prices` or the
   limits are `null` (O1, O9), stop and ask the author for the prices with their date. Never use a price that you
   remember or find without the author.
2. Count, per layer: items × repeats × calls per item, for generation and for each judge metric. Subtract cached
   judgments (key: judge model, rubric hash, input hash).
3. Estimate tokens per call from a real prompt: build 3 prompts with the real templates and count tokens with the
   provider tokenizer. State the method.
4. Cost = Σ (input tokens × input price + output tokens × output price). Give one full run, the 5-item pilot, the
   3–5 repeat runs for tolerance (S7), and the deterministic-only run (must be 0).
5. Compare with the remaining budget: `cost.limit_usd_total` (5 USD for the whole track) minus the sum in
   `cost.ledger`, and with `cost.limit_usd_per_run`. Over a limit → say so and propose a smaller
   scope (fewer items, fewer metrics, cheaper judge for triage).
6. Show the table to the author: layer, calls, input tokens, output tokens, cost, assumptions. Stop. Wait for approval.
7. During the run, the LLM client stops at the limit. After the run, record the actual cost next to the estimate in
   `EXPERIMENT_LOG.md`.
