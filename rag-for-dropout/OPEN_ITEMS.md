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
| O12r | Remaining proposed defaults marked (P): bootstrap 2,000 resamples; ECE 15 equal-mass bins; k_rrf = 60; judge κ bar 0.6. (Report-window rule and cal split: closed as O12) | S2 report, S4 | As listed |
| O13 | Human labels for judge validation: who labels, how many (≈50), which rubric scale | S4 | The author labels 50 reports on the rubric scales in `rubrics/` |
| O14 | Event-level aggregation rule for the informational Layer 0 metric | S2 | Sum of window vote shares over the fault windows of a run, argmax |

| O17 | k = 10 test macro-F1 (seed 42, batch 32) is 1.39e-3 from the stored curve; the tolerance is 1e-3. k = 1, 2, 5 pass (1.8e-4, 2.9e-4, 4.1e-4). Diagnostic: batch size alone moves it by 8.0e-4; one mask offset has std 1.31e-3 (`reports/s2/S2_REPORT.md` § 2) | S2 close | Accept as mask-draw noise (the pipeline uses its own predictions); or tolerance 2e-3 at k = 10; or re-evaluate at another batch size |
| O18 | Seeds 43 and 44 for Layer 0: extraction alone is about 34 min each (k = 0 ≈ 4 min, k > 0 ≈ 30 min), above the 30-min rule (O5) | S2 close | Run both, one after the other, nice 19, 4 threads; or skip |
| O19 | Next step after S2: Layer 0 metrics, saved case store (FAISS), leakage assertion and sanity run, scenario-held-out control (uses the `scenario_heldout_*` checkpoints; confirm). The gate target O6 is still open | S2 close | As listed |

## Closed (author answers 2026-10-28)
| ID | Answer |
|---|---|
| O11b | Use `resubmit/1170_revised.tex`; copied to `corpus/v1/` with its SHA-256 in `corpus/v1/MANIFEST.json` (`RESULTS.md` copied too). `tests/gridsense/test_corpus.py` checks hashes and curve numbers |
| O12 | Report window = the earliest w ≥ 21 whose `val` accuracy is within 1 percentage point of the maximum `val` accuracy over w ≥ 21 (applied at k = 0; per-k curves reported). cal-A/cal-B = 50/50 by run, stratified by scenario. `unlabelable_window_mask` (drops w0020) in all Layer 0 metrics |
| O15 | 10 % subsample of train runs: whole runs, stratified by scenario (a scenario fixes bus, fault type and R_f, so fault type is stratified too); one subsample for all methods. Compare (a) concatenation (1,600-d), (b) mean + max pooling (128-d), (c) concatenation + PCA to 256-d fitted on the subsample. Ask before building a full store |
| S2-id | Identity gate option 1 (author, 2026-10-28): CPU predictions are the pipeline predictions; differ only where the vote gap < 2e-3 (rule set after seeing the data; record the 6 window IDs, max gap 1.8e-3 and the macro-F1 difference in the limitations); byte-identical determinism re-run; k > 0 masks use the paper's seeded generator and macro-F1 within 1e-3; PCA arm standardized, fitted on the train subsample only; CPU only, nice 19, 4 threads, 2 if the S4.4 epoch time rises > 10 %; order: report window, latent comparison, calibration, k > 0; commit S2 to `gridsense-eval`; stop after S2 |
| S2 | Approved 2026-10-28: `rag` optional dependency group; `nice -n 19`, `torch.set_num_threads(4)`; after extraction the predictions must equal the paper predictions, else stop and report |
| O2 | Vote share = soft vote / 25. Fit a monotone map on the top-1 share: Platt and isotonic, compare both. No temperature on node logits; predictions stay identical to the paper. Also record the top-1 − top-2 margin |
| O2b | Split `val` by run into cal-A and cal-B. Fit calibration on cal-A; select thresholds on cal-B. State the checkpoint-selection bias in the limitations |
| O3 | Report window = a rule on post-fault samples. Measure accuracy per window index on `val`; use the earliest window where accuracy stops rising. Never use `test` for this choice. (Window w has w − 20 post-fault samples; w0020 has 0, so w ≥ 21) |
| O4 | Per-node embeddings before the head (`h` [25, 64]). Compare by kNN-vote accuracy on `val`: (a) concatenation in fixed node order (1,600-d), (b) mean and max pooling (128-d). Use the better method. RAM: O15 |
| O5 | The system uses seed 42. Repeat Layer 0 for seeds 43 and 44 if the CPU time is below 30 min for each |
| O9 | Cost limit: **5 USD in total for the track** (author update 2026-10-28; replaces "5 USD per run, 20 USD total"). Stop and ask before more |
| O10 | Commit to a new branch `gridsense-eval`. Never commit to `main` |
| O11 | Use the current paper version with a version tag. Its numbers must match `results/RESULTS.md` for the checkpoint in use (a test checks it). Add the camera-ready version later as a new corpus version. Which file: O11b |
