# Paper 1170 – Revision Plan (Reviewer Changes Only)

**Paper:** Dropout-Aware Training for Sensor-Robust Fault Localization with Spatio-Temporal Graph Neural Networks
**Venue:** ICMLDE 5.0 → Procedia Computer Science
**Deadline:** 7 days from the date of the decision email
**Scope rule:** Change only what the reviewers asked for. Do not add the newer project stages. Do not change results that the reviewers did not question.

---

## 0. Before you start

1. Find the git commit (or folder copy) that produced the **submitted** paper. Make a new branch from it: `revision-1170`. Run all new experiments from this branch only.
2. Make a copy of the submitted LaTeX source. Edit the copy.
3. Keep a change log. For each edit, record: reviewer, comment number, section, page, old text, new text. You need this for the response letter.
4. **Number check:** the project notes have the imputation result as AUDC 0.495 / Ret@10 13.1%. The paper has AUDC 0.476 / Ret@10 11.8%. Confirm which run is the final one before you touch Section 4.3, Table 6 and Section 4.5.

---

## 1. Summary table

| ID | Reviewer comment (short) | Type | Effort | Sections affected |
|---|---|---|---|---|
| R1.1 | How the 25 µPMU locations and edges were selected | Text | Low | 3.1, Fig. 1 caption |
| R1.2 | Evaluate more feeders / topologies | Text (limitation + justification) | Low | 5.3, 6, response |
| R1.3 | Three seeds acknowledged | No change needed | — | Response only |
| R2.1 | Wider training dropout levels, incl. k = 10 | **Experiment** | Medium | 3.5, 4.3/4.4, new table, 5.3 |
| R2.2 | Correlated / geographically clustered failures | **Experiment** (inference only) | Low–Medium | 3.4, 4.3, 5.3, 6 |
| R2.3 | Do the tests account for mask realizations and seeds? | Text | Low | 3.4, 4.5, Table 7 caption |
| R2.4 | Compare with stronger learned imputation | **Experiment** (one learned imputer) | Medium–High | 4.3, Table 6, 5.2, 5.3 |
| R2.5 | Effect of 4-layer depth chosen on RGSAGE-Max | Text (+ optional small experiment) | Low | 3.2, 4.1, 5.1, 5.3 |
| R3.1 | Emphasize zeroed measurements and availability mask | Text | Low | 3.1 (new paragraph), 3.4 |
| R3.2 | Verify stats, report adjusted p-values consistently | Text + recompute | Medium | Abstract, 4.3, 4.5, 5.1, 5.2, 6 |
| R3.3 | How 26-class macro-F1 is calculated from windows | Text (verify code) | Low | 3.4 (or 3.7) |
| R3.4 | State RMS phasor limitation prominently | Text | Low | Abstract?, 5.1, 5.3, 6 |
| R3.5 | Qualify "40% sensor loss" claims (one feeder) | Text | Low | Abstract, 5.1, 6 |
| G | English proofreading, Procedia template, similarity ≤ 10 %, response letter | Admin | Medium | Whole paper |

---

## 2. Reviewer #1

### R1.1 – Selection of the 25 µPMU locations and the topology-derived edges
**Problem:** Section 3.1 says "following the placement of [1]" and "connectivity is inherited from the feeder topology". This is not precise.

**Change (Section 3.1, paragraph 2):**
- State that the 25 µPMU buses and their phase configuration are taken **unchanged** from [1]. List the bus IDs (they are in Fig. 1) in the text or in a footnote.
- State the exact edge rule. Check the code first. The expected rule is: *two µPMU buses are connected if the unique path between them in the radial feeder tree passes through no other µPMU bus.* This gives a tree with 24 edges.
- Add one sentence: the placement is not optimized in this work; placement optimization is out of scope.

**Change (Fig. 1 caption):** add "edges connect µPMU pairs whose feeder path contains no other µPMU".

**Code task:** Read the graph-building function. Confirm the rule. If the rule is different, write the real rule.

### R1.2 – Additional feeders or topologies
**Decision:** No new feeder in 7 days (new simulation corpus + 5 architectures × 2 trainings × 3 seeds is not possible on one T1000).

**Change:**
- Section 5.3: make the single-feeder limitation the **first** item. Name concrete next feeders (e.g., IEEE 34-bus, the reconfigured "green" topology of [1]).
- Section 6: keep "larger and more diverse feeder datasets, cross-topology generalization" in future work.
- Response letter: explain why one feeder was used (direct comparison with [1] needs the same feeder and placement), and that the method itself does not depend on the feeder (it changes only the input masking).

### R1.3 – Three seeds
No change. In the response, thank the reviewer.

---

## 3. Reviewer #2

### R2.1 – Wider training dropout grid, including k = 10
**Current state:** training uses K = {0, 1, 2, 5}. Section 5.3 already says this is an open question.

**Experiment (RGSAGE-Max backbone only):**
| Variant | Training grid K |
|---|---|
| DA-current (exists) | {0, 1, 2, 5} |
| DA-light | {0, 1, 2} |
| DA-wide | {0, 1, 2, 5, 10} |

- 3 seeds (42/43/44) each, same 30-epoch recipe, same evaluation sweep k ∈ {0, 1, 2, 5, 10}, 3 mask realizations.
- That is 6 new training runs. Estimate the time from one existing run before you start. If time is short, run DA-wide first (it is the one the reviewer named).

**Paper changes:**
- Section 3.5: say that the main grid is {0,1,2,5} and that a grid ablation is in Section 4.x.
- New short table (or extra rows in Table 6): clean F1 (k=0), F1 at k=10, AUDC, Ret@10 for each grid.
- Add paired tests (DA-wide vs DA-current) at k=0 and k=10, Holm-adjusted.
- Section 5.3: remove the sentence "we ran no ablation over the training dropout grid". Replace it with the result.

### R2.2 – Correlated / geographically clustered failures
**Experiment (inference only, no retraining):**
- New mask policy "clustered": pick a random seed µPMU, then add its nearest µPMUs by hop distance on the measured-only graph (BFS) until k nodes are dropped. Ties broken by the shared RNG seed.
- Optional second policy "shared communication path": drop a whole branch/subtree of the measured-only graph.
- Run on all 5 standard and 5 dropout-aware checkpoints (same frozen test split, same k grid, 3 mask realizations).

**Paper changes:**
- Section 3.4: add the clustered policy to the missingness model.
- Section 4.3: add a "Clustered dropout" paragraph after "Targeted (worst-case) dropout". Report AUDC and Ret@10, standard vs DA, with Holm-adjusted tests.
- Section 5.3: change "spatial correlation between the failed units remains unmeasured" to report the result.
- Section 6: update future work.

### R2.3 – Do the statistical tests account for mask realizations and seeds?
**Change (Section 3.4 and Section 4.5, Table 7 caption):**
State the procedure exactly:
1. For each training seed, the 3 mask realizations are averaged first.
2. The paired test then uses n = 3 seed-level values (df = 2).
3. So the test variance is the training-seed variance; mask variance is averaged inside each seed, not treated as independent samples (this avoids pseudo-replication).
- Optional (strong answer): report the mask-realization std inside each seed, to show it is small compared to seed std. Compute it from existing evaluation outputs — no retraining.

**Code task:** confirm in the evaluation script that masks are averaged before the t-test.

### R2.4 – Stronger learned imputation
**Experiment (pick one, in this order of cost):**
1. **Learned GNN imputer (recommended):** a small GraphSAGE/GCN on the measured-only graph, trained on the training split to reconstruct the voltages of randomly masked nodes (masked-autoencoder objective). At test time, fill dropped nodes, set mask = 1, feed the standard-trained RGSAGE-Max. Same frozen test split and dropout masks as Table 6.
2. Also report an **imputer + dropout-aware model** row if time allows.
3. Do not try to reimplement the full pipeline of [10]; say why in the response (it assumes a fixed, known set of unmeasured buses).

**Paper changes:**
- Section 4.3 "Imputation baseline": add the learned imputer. Report reconstruction error (RMSE on held-out masked nodes) and the F1 / AUDC / Ret@10 numbers.
- Table 6: add a row "Learned GNN imputation (standard)".
- Table 7: add "DA vs learned imputation, F1 at k=10". Update the Holm family size.
- Section 5.2, point two: update the argument with the new result.
- Section 5.3: change the imputation limitation sentence.

### R2.5 – Effect of the 4-layer depth chosen on RGSAGE-Max
**Current state:** Sections 4.1 and 5.1 already say the depth was tuned on RGSAGE-Max and may hurt RGATv2.

**Change (text):**
- Section 3.2 ("Spatial depth"): say clearly that depth was selected on RGSAGE-Max only and then fixed for all models, and that this can favor RGSAGE-Max in the clean ranking.
- Section 5.1: add one paragraph — the clean-accuracy ranking and the mid-tier robustness ranking are conditional on depth 4; the main claims (dropout-aware gain on every architecture, SDR-STGNN vs DA) are within-architecture comparisons and do not depend on cross-architecture depth fairness.
- Section 5.3: add as a limitation.

**Optional experiment (if time):** single-seed depth sweep L ∈ {1, 2, 3, 4} for RGCN and RGATv2 (reduced budget, same as the RGSAGE-Max diagnostic in 4.1). Report in one sentence or a small table.

---

## 4. Reviewer #3

### R3.1 – Emphasize zeroed measurements and the availability mask in Methods
**Change:** Add a short, named paragraph in Section 3.1, for example **"Encoding of missing measurements."** Move the content now spread across 3.1 and 3.4 into it:
- Mask m_p = 1 → measurement present; m_p = 0 → phase not installed or sensor dropped.
- A dropped sensor: all 6 channels (3 voltages + 3 mask) are set to 0.
- A zero voltage alone is ambiguous (it can be a real deep-sag reading after normalization); the mask is the only definitive missingness signal.
- In the clean training data the mask is static per node (only 14 of 25 µPMUs are three-phase); this is why the standard model has never seen mask (0,0,0) — link forward to Section 5.2.
- Section 3.4 then only refers back to this paragraph.

### R3.2 – Verify statistical claims; report adjusted p-values consistently
**Rule for the revision:** every p-value in the text is either Holm-adjusted (written p_Holm) with its family stated, or clearly marked as a single pre-planned test. No unlabeled raw p-values for multiple comparisons.

**Places to fix (search the .tex for `p =`, `p <`, `p \le`):**
| Location | Current | Change |
|---|---|---|
| Abstract | "p < 0.01" | "p_Holm = 0.003" |
| 4.3 clean F1 | "p = 0.181" | add p_Holm = 0.196 |
| 4.3 imputation | "not significant ... Section 4.5" | give p_Holm |
| 4.3 targeted dropout | p = 0.010, 0.017, 0.003, ≤0.014, ≤0.008, 0.042, 0.004, 0.33 | Define a "targeted" test family; Holm-adjust; report p_Holm |
| 4.3 persistent dropout | min p_Holm = 0.46, p ≤ 0.0006 | Check the second one is adjusted; label it |
| 4.5 | p = 0.098, 0.102, 0.023, 0.076 | Report p_Holm and state the family for each |
| 5.1 | p = 0.076 | p_Holm |
| 5.2 | p = 0.098, p = 0.102 | p_Holm |
| Table 7 | — | Recompute with new rows (R2.1, R2.4); update family size in caption |

**Code task:** write one script that recomputes every reported test from the saved per-seed results and prints raw p, p_Holm, Δ, 95 % CI. Check every number in the paper against this output.

### R3.3 – How the 26-class macro-F1 is calculated from window-level predictions
**Change (add to Section 3.4, "Dropout sweep and metrics"):** state, after checking the code:
1. Each test window gets one graph-level prediction by soft voting over its 25 nodes (Eq. 5).
2. Predictions of all 247,000 test windows are pooled (no aggregation over the 40 windows of an event).
3. Per-class F1 is computed for each of the 26 classes (25 fault locations + no-fault), and macro-F1 is their unweighted mean (e.g., scikit-learn `f1_score(average="macro")`).
4. The no-fault class is included; because it is 50 % of the windows, macro (not micro) averaging is used so the 25 fault classes dominate the score.
5. Node-level macro-F1 (Table 4) uses each node's prediction as a separate sample — state this too.
6. Optionally add one formula: F1_macro = (1/26) Σ_c 2·P_c·R_c / (P_c + R_c).

**Code task:** confirm windows vs events, inclusion of no-fault, and the averaging mode.

### R3.4 – RMS phasor limitation stated prominently
**Current state:** only in Section 3.1.

**Change:**
- Section 5.3: add as an explicit limitation — the data are quasi-static RMS phasors; no electromagnetic transients, traveling waves, DC offset, or sub-cycle content; real-time localization on field µPMU streams is not demonstrated.
- Section 5.1, the "Practically, on this feeder an operator could lose..." sentence: add "in RMS phasor simulation".
- Section 6: add one clause.
- Consider one short clause in the Abstract ("on simulated RMS phasor data").

### R3.5 – Qualify the "40 % sensor loss" claims
**Places to change:**
- Abstract: "raises retention at 40 % loss from 5.7 % to 80.3 %" → add "on the simulated IEEE 123-bus feeder".
- Section 5.1: "an operator could lose roughly 10 of 25 µPMUs ... while macro-F1 stays above 0.67" → "on this simulated feeder, ... under random, independent loss".
- Section 6: same qualification.
- Remove or soften "supports resilient monitoring ... without additional sensing hardware" (5.1 end) → "suggests ... ; field validation is needed".

---

## 5. General requirements from the editor

- [ ] **English proofreading** of the full manuscript (a professional or a careful second pass).
- [ ] **Procedia template:** header, footer, margins, column spacing, figure/table captions, equation numbers. Use the current Elsevier Procedia CS LaTeX template. Check that new tables fit the page.
- [ ] **Page count:** new tables and paragraphs add length. Check the page limit and the over-length fee. Shorten other text if needed (do not cut reviewer-requested content).
- [ ] **Similarity ≤ 10 %:** run the similarity check on the revised manuscript. Upload the report as a supplementary file.
- [ ] **Response letter:** point-by-point, page-by-page. For each comment: comment text → response → change made → section + page number. Put it at the start or end of the same file as the manuscript.
- [ ] **Single file name:** `1170_revised_version.pdf` (or .docx).
- [ ] **Upload:** use the *edit* option on the same submission portal. Do not delete existing files.
- [ ] **Acknowledge email:** send a short email to info@icmlde.org with Paper ID 1170 to confirm you received the decision.
- [ ] **Figures:** update Fig. 2 only if you add new curves (e.g., clustered dropout or DA-wide). Keep the old figure otherwise.

---

## 6. Suggested order (7 days)

| Day | Work |
|---|---|
| 1 | Send acknowledgement email. Restore submitted code branch. Resolve the imputation-number mismatch. Start R2.1 training runs (longest job). Implement R2.2 clustered masks. |
| 2 | Run R2.2 on all 10 checkpoints. Implement and train the R2.4 learned imputer. Continue R2.1 runs. |
| 3 | Finish R2.1 and R2.4 evaluations. Write the statistics script (R3.2, R2.3). Confirm R1.1 edge rule and R3.3 metric code. |
| 4 | Text edits: R1.1, R1.2, R2.3, R2.5, R3.1, R3.3, R3.4, R3.5. Add new results: R2.1, R2.2, R2.4. |
| 5 | Recompute all p-values; fix every p in the text (R3.2). Update tables, abstract, conclusion. Template check. |
| 6 | Proofreading. Similarity check. Write the response letter with page numbers from the final PDF. |
| 7 | Final read, merge letter + manuscript into one file, upload, upload similarity report. |

**If time runs out:** the minimum is R2.1 (DA-wide only) and R2.2 (clustered, inference only) as experiments, all text items, and R2.4 answered as a stated limitation with a clear reason. Say this honestly in the response letter.

---

## 7. Response letter skeleton

```
Response to Reviewers – Paper ID 1170

We thank the reviewers ... Changes in the manuscript are marked in blue.

Reviewer #1
Comment 1.1: Please clarify how the 25 µPMU locations and their topology-derived edges were selected.
Response: ...
Change: Section 3.1, page X, paragraph Y; Fig. 1 caption, page X.

Comment 1.2: ...
...

Reviewer #2
Comment 2.1: ...
...

Reviewer #3
Comment 3.1: ...
...

Editorial requirements
- English proofreading: ...
- Template: ...
- Similarity: X % (report attached).
```
