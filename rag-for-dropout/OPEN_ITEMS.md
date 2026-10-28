# OPEN_ITEMS.md — decisions that block later stages

Written 2026-10-28 (S0). Do not choose any of these values. Stop and ask the author. When the author answers, write
the answer and the date here, copy it to `claude-rag.md` § 3 or `configs/eval.yaml`, and move the item to "Closed".

## Open
| ID | Question | Blocks | Proposal (not approved) |
|---|---|---|---|
| O1 | OpenAI generator, judge and (optional) embedding models: exact snapshot names, price per 1M input/output tokens, with the date | S4 | Cheap generator candidates; judge candidates of a higher tier; prices from the provider page, given by the author |
| O6 | Gate target for τ_p and τ_k (selected on cal-B) | S2 | Accuracy ≥ 95 % on CONFIDENT windows, per k if one threshold does not hold |
| O7 | Extra corpus sources: generated feeder docs, curated notes, illustrative SOP | S3 | Not approved on 2026-10-28. Without an SOP, the control-command trap uses an injected chunk |
| O8 | LangSmith: project name, workspace, approval of the first upload. (Key: in `rag-for-dropout/.env`, valid — HTTP 200 on the US endpoint, 2026-10-28; ignored by Git) | S8 | Project `gridsense-dev`; reviewed datasets only; traces redacted |
| O16 | `OPENAI_API_KEY` (the author adds it when coding starts) | S4 | Same `.env`; `gridsense.settings` reads it |
| O11b | Which file is "the current paper version": `resubmit/1170_revised.tex` in the revision worktree (newest, 2026-10-21, uncommitted edits, blob `ee097f8`), or `paper/overleaf.tex` on `main` (blob `88ce732`, commit `f4b7e7b`) | S3 | `1170_revised.tex`, copied into `rag-for-dropout/corpus/v1/` at ingestion, tag `paper1-rev1170@ee097f8` |
| O12 | Proposed defaults marked (P): bootstrap 2,000 resamples; ECE 15 equal-mass bins; k_rrf = 60; judge κ bar 0.6; cal-A/cal-B = 50/50 by run, stratified by fault bus, seed 42; O3 "stops rising" rule (below) | S2, S4 | As listed |
| O13 | Human labels for judge validation: who labels, how many (≈50), which rubric scale | S4 | The author labels 50 reports on the rubric scales in `rubrics/` |
| O14 | Event-level aggregation rule for the informational Layer 0 metric | S2 | Sum of window vote shares over the fault windows of a run, argmax |
| O15 | RAM for the O4 concatenation method: 25 × 64 float16 per window = 3.7 GB for all 1,155,000 train windows; about 4 GB RAM is free | S2 | Compare (a) and (b) with the same train store: a 10 % subsample of train runs, stratified by fault bus, seed 42. If (a) wins, ask before the full store (options: wait for RAM, PCA fitted on train, or a run subsample) |

**O3 "stops rising" rule (P):** compute top-1 accuracy per window index w = 21…39 on `val` at k = 0, with run-grouped
bootstrap CIs. The report window is the smallest w where the gain acc(w+1) − acc(w) is not above 0 at 95 %
confidence. Report the per-k curves too. If another k gives a different w, report it and ask.

## Closed (author answers 2026-10-28)
| ID | Answer |
|---|---|
| O2 | Vote share = soft vote / 25. Fit a monotone map on the top-1 share: Platt and isotonic, compare both. No temperature on node logits; predictions stay identical to the paper. Also record the top-1 − top-2 margin |
| O2b | Split `val` by run into cal-A and cal-B. Fit calibration on cal-A; select thresholds on cal-B. State the checkpoint-selection bias in the limitations |
| O3 | Report window = a rule on post-fault samples. Measure accuracy per window index on `val`; use the earliest window where accuracy stops rising. Never use `test` for this choice. (Window w has w − 20 post-fault samples; w0020 has 0, so w ≥ 21) |
| O4 | Per-node embeddings before the head (`h` [25, 64]). Compare by kNN-vote accuracy on `val`: (a) concatenation in fixed node order (1,600-d), (b) mean and max pooling (128-d). Use the better method. RAM: O15 |
| O5 | The system uses seed 42. Repeat Layer 0 for seeds 43 and 44 if the CPU time is below 30 min for each |
| O9 | Cost limit: **5 USD in total for the track** (author update 2026-10-28; replaces "5 USD per run, 20 USD total"). Stop and ask before more |
| O10 | Commit to a new branch `gridsense-eval`. Never commit to `main` |
| O11 | Use the current paper version with a version tag. Its numbers must match `results/RESULTS.md` for the checkpoint in use (a test checks it). Add the camera-ready version later as a new corpus version. Which file: O11b |
