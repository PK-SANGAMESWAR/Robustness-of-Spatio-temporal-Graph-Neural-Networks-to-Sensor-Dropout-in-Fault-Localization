# dropout-rag-intro.md — GridSense: from a dropout-robust prediction to a verified, human-approved report

Written 2026-10-28. Figure: `RAG-arch.png` (its "0.846" label is the standard model; see `PROPOSED_CHANGES.md` P1).

## 1. The starting point: Paper 1 and dropout-aware training
Paper 1 (`paper/overleaf.tex`, ICMLDE Paper ID 1170) adopts the STGNN fault-localization methodology of Karabulut et
al. (2026) on the IEEE-123 feeder with 25 μPMUs, and reports our own numbers. Its contribution is sensor dropout:
μPMUs that stop reporting **after** deployment.

- A window is `[N=25 nodes, S=20 steps, F=6]`: 3 RMS voltages and 3 static mask channels per μPMU.
- The model has 26 classes: class 0 = no fault, class c = fault bus `FAULT_BUSES[c-1]`.
- Each node gets a GRU-128 embedding, 4 GraphSAGE-max hops (64-d), and a per-node head. The window decision is the
  soft vote: the sum of the 25 node softmaxes, then argmax.
- Dropout at inference zeroes all 6 channels of k random μPMUs, so the mask becomes (0,0,0). Standard training never
  sees that pattern. **Dropout-aware training** masks k ∈ {0, 1, 2, 5} random nodes per training batch, so the model
  learns that (0,0,0) means "dark sensor".

| RGSAGE-Max, macro-F1 (3 seeds) | k=0 | k=1 | k=2 | k=5 | k=10 | AUDC | Ret.@10 |
|---|---|---|---|---|---|---|---|
| Standard training | 0.846 | 0.736 | 0.590 | 0.236 | 0.048 | 0.402 | 5.7 % |
| Dropout-aware training | 0.839 | 0.827 | 0.815 | 0.774 | 0.674 | 0.913 | 80.3 % |

Source: `results/RESULTS.md` §2–4. Dropout-aware training costs no clean accuracy that is significant (−0.008,
p = 0.181) and no inference latency. It does not help against measurement noise (the mask stays 1).

## 2. The gap that GridSense fills
A macro-F1 of 0.674 at k = 10 is an average over 247,000 test windows. It says nothing about **one** prediction.
An operator who receives "fault at bus 57" needs to know three things:
1. Is this prediction likely right? (calibrated confidence, similar past cases, sensor coverage)
2. What does this fault and this dropout pattern mean? (retrieved evidence with citations)
3. Who approves an action? (a human, always)

GridSense answers these questions with separate components. The STGNN predicts. Python code verifies. Retrieval
finds evidence. An LLM drafts the report. A human approves. The LLM never decides whether to trust the STGNN.

## 3. Pipeline (rev 2 design, `spec/impl_02.md` §3)
| Step | Component | Output |
|---|---|---|
| 1 | Input validation | Valid window, mask, model version — or REJECT |
| 2 | STGNN (dropout-aware RGSAGE-Max) | Soft-vote scores, top-1 bus, latent vector |
| 3 | Deterministic gate | CONFIDENT / LOW_CONFIDENCE (candidate set) / REJECT, with reason codes |
| 4 | Case retrieval (latent kNN over the train split) | Neighbours, kNN agreement, neighbour error rate |
| 5 | Document retrieval (dense, BM25, hybrid, reranker) | Cited passages from the model-card corpus |
| 6 | Report drafting (LangGraph, OpenAI model) | Pydantic report, `approval_status: pending_human_review` |
| 7 | Report validation | Field consistency, schema, citation substring checks |
| 8 | Human approval | Logged decision. No equipment control |

## 4. Why the case store matters
Document retrieval explains what a fault type means. It cannot say whether **this** prediction is right. The case store
holds latent vectors of training windows with their true bus and the model's own errors. "8 of the 10 most similar
training windows had a different fault bus" is direct evidence of doubt. The cheapest experiment of the track (stage
S2) measures whether kNN agreement detects errors better than the soft-vote score. If it does not, the gate is
simplified and the result is reported as it is.

## 5. What is new compared with Paper 1
- Per-prediction reliability: calibration, selective risk, coverage, error-detection AUROC, at every k.
- A component-level evaluation of a RAG stack on a technical corpus: chunking, embedder, vector store, retrieval,
  reranker, generator LLM, judge LLM — each chosen by measurement on a reviewed golden set.
- End-to-end tests that a report does not rubber-stamp a wrong prediction (counterfactual swap, evidence flip,
  empty evidence, injection, aggregate-metric-as-confidence trap).
- LangSmith tracing and online LLM-as-judge evaluation on a replay stream of events.

## 6. Known limitations (state them in every result)
1. The data is simulated (OpenDSS, noise-free corpus). No result shows safety on a real grid.
2. The corpus is research material (Paper 1, `RESULTS.md`). No utility approved it. Every report says so.
3. All 825 scenarios are in both train and test (the split divides runs). Latent-kNN evidence is optimistic.
   The scenario-held-out split (`dataset_fullscale_50_scenario/`) is the informational control.
4. The `val` split chose the checkpoint. It is split by run into cal-A (calibration) and cal-B (thresholds), but
   both halves were seen by checkpoint selection. This gives a small optimistic bias.
5. An LLM judge can be wrong. A judge metric counts only after its agreement with human labels is measured.
6. The report window is one fixed index chosen on `val` (window w has w − 20 post-fault samples; w0020 has none).
   Its accuracy is not the paper's macro-F1, which averages all 40 windows. Layer 0 reports accuracy by window index.
7. **CPU vs paper predictions.** The pipeline uses its own CPU predictions. On `test` at k = 0 (seed 42), 6 of 247,000
   window predictions differ from the paper's CUDA run: `21_ABG_0p1_0042_w0020`, `21_BCG_0p1_0030_w0020`,
   `51_AG_0p1_0030_w0020`, `62_BG_10p0_0034_w0031`, `65_CA_10p0_0031_w0029`, `93_BCG_0p1_0013_w0020`. All are near-ties
   (vote gap 1.9e-6 to 1.835e-3; votes sum to 25). Macro-F1 differs by −7.04e-6 on `test` (0.8515212 vs 0.8515282)
   and −2.6e-7 on `val`. The tolerance rule (differ only where the gap is < 2e-3) was set **after** these data were
   seen. A re-run on the same CPU is byte-identical, so the CPU predictions are deterministic.
8. n = 3 training seeds. Seed-level comparisons are trends unless a paired test with Holm correction says otherwise.
