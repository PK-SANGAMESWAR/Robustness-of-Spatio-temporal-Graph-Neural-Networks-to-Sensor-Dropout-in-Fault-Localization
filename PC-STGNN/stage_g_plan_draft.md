# Stage G plan — draft (2026-11-02)

**Status: plan only.** Nothing was run. The GPU, the queue and the watcher were not used or changed. `data/ood_test/` was not
opened, read or scored. `08_STAGE_G_eval.md`, `14_PREREGISTRATION.md`, `07_STAGE_F_gate.md` and
`stage_f_gtf2_preregistration.md` are not changed. Every setting below is a **proposal** until the author accepts it.
Rates are measured values from `results/s4_4/*_progress.json` unless a line says "estimate".

## 0. Findings that change the plan

1. **There is no "Gate 6".** The gate map stops at G5 (`00_INDEX.md:45`). `08_STAGE_G_eval.md` has no gate box. The nearest
   criteria are in § 1.2. Proposal: the author defines Gate 6 by a dated amendment before the test split is scored.
2. `14_PREREGISTRATION.md` names **R0** as the pair for H1–H6 (lines 51–56). Gate 4 and `12_CHECKLISTS.md:87` pair with **R0ⁿ**.
   14 has no amendment yet (line 115: "None").
3. **The grounded / ungrounded split is not registered in 14.** `08:38-40` requires it before the sealed set is opened.
4. **The sealed set holds raw simulation only** (`scripts/generate_ood_sets.py:12-14`). Windowing and labels are not made yet.
5. **The Gate-4 scorer does not save predictions.** It keeps them in memory only (`score_s4_4_gate4_run.py:179`). The event
   bootstrap of 14 § 3 needs them.
6. **08 and 13 give costs that are out of date.** 08:4-5 and 08:98 say "≈ 7 GPU-h" per retrain. 13:31 says 2 retrains = 14 GPU-h.
   The measured fusion run is 33.6 GPU-h (26.16 train + 7.44 score).
7. **The operator R axis has 18 points and 84,167 solves** (`operator/grid.json`), not 15 and 70 k as in `01_SPEC.md` § 2.
   The OOD values 3, 5, 20, 30 Ω lie near grid points 3.05, 4.82, 19.0, 30.0.

## 1. Current definitions (quoted)

### 1.1 Stage G steps — `08_STAGE_G_eval.md`
- **S6.1 Master sweep** (line 7). "Every model × every condition. Shared seeds throughout — mask seeds, corruption seeds, noise
  seeds identical across models." (9–10). Conditions (12–20): random dropout k ∈ {0,1,2,5,10}; persistent; structured; targeted;
  silent failures "4 types × s ∈ {1,2,3}"; measurement noise σ ∈ {5e-4, 2e-3, 1e-2}; OOD "{3,5,20,30} Ω × k ∈ {0,5,10} — open the
  sealed set here, once. Report split by grounded vs ungrounded".
- Models (22–24): "B1, B1ᵃ (matched filter), B2 (PSM), logreg/RF, both impute-then-infer rows, all 5 baselines standard, all 5
  dropout-aware, SDR-STGNN, R1–R7, B2ᵃ, G0–G3. … Only the new models need the existing conditions re-run".
- Tables T5, T6, T8, T9, T10. Figures F-G1 to F-G5 (42–46).
- **S6.2 Model-mismatch study** (48), see § 1.4. **S6.3 Statistics** (107): mean ± std over 3 seeds; "Paired bootstrap over test
  **events** … for the pre-registered comparisons only: R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2"; 95 % CIs; table T13.
- **S6.4 Cost accounting** (121): parameters, FLOPs, training GPU-h, latency per window "(including the residual and, for G2, the
  LOO pass and second forward)"; table T12.

### 1.2 Gate criteria (no "Gate 6" exists)
- Gate map, `00_INDEX.md:45`: "G0 assets → G1 fidelity → G2 go/no-go → G3 infrastructure → G4 PC-STGNN → G5 trust gate."
- `13_BUDGET_TIMELINE.md:87-98`, "Success criteria": "Judge against all six: 1. The gain concentrates in the pre-registered
  conditions … 2. Clean k = 0 accuracy does not drop by more than one seed standard deviation. 3. Sign-consistent across all three
  seeds. 4. Paired bootstrap p < 0.05 on the pre-registered comparisons only. 5. R6 (shuffled physics) removes the gain and R2
  (learned decoder) does not reproduce it. 6. It survives ±5 % model mismatch."
- `14:79-80`: "A hypothesis is "supported" only if the sign is consistent in all 3 seeds **and** the bootstrap CI excludes zero.
  Anything else is reported as observed, not claimed."
- `14:90-103` § 5: the fallback boxes are ticked "**after** the sealed set is opened, with the date".

### 1.3 H3 — `14_PREREGISTRATION.md`
- Line 53: `| H3 | Physics fusion extrapolates to unseen resistance | R5 vs R0 | R ∈ {3,5,20,30} Ω, k = 0 | F1 (sealed OOD) | + |`
- Rule: lines 75–80 (bootstrap over 6,175 test events, 10,000 resamples, 95 % CI, only R5 vs R0, G2 vs G0, R5 vs R6, R5 vs R2);
  line 78 "families = {H1–H4}, {H5}, {A1–A3}"; lines 79–80 as in § 1.2.
- Gaps: H3 does not say pooled or per-R F1, does not name σ, and pairs R0 (see § 0.2). A4 ("B2ᵃ collapses on OOD R", line 70)
  does not define "collapses".

### 1.4 Model-mismatch study — `08:48-105`
- Which mismatch: rebuild the signature table "from a perturbed feeder model. Leave the corpus and the trained models untouched"
  (70–71). Levels (73–78): regulator tap ±1 step on each regulator ("lead with this one"); line impedances random per segment ±2 %,
  ±5 %, ±10 %; load composition (constant-P vs a ZIP mix), one alternative; one unmodelled lateral or small DER. That is 6 or 7
  tables (7 if tap +1 and −1 count separately).
- Which models (93–94): "Gain over R0 as a function of mismatch level, for PSM (B2), fusion-only (S2.6 on R0) and R5."
- How many runs (96–98): "≈ 30 min of solves per perturbed table (70 k solves). The fusion-only variant needs no retraining.
  One retrain of R5 at ±5 % confirms the trained variant behaves the same way (≈ 7 GPU-h)." So 08 says **1 retrain**.
  `13:31` says **2 retrains** (±5 % plus one of ±2 %/±10 %, 14 GPU-h). The two files disagree.
- Outputs: T11, F-G6. `14:87-88`: the curve is reported "even if it collapses".

### 1.5 Opening the sealed set
- `08:27-28`: "Before opening the sealed OOD set, verify its hash against `data/ood_test.sha256`. Record the date it was opened in
  `14_PREREGISTRATION.md`." `08:38-40`: "Register the split in `14_PREREGISTRATION.md` before the sealed set is opened".
- `12_CHECKLISTS.md:63-65`: "The sealed OOD set was opened once, in Stage G. The Gate-2 pilot used different seeds and is not
  reported"; "never used for tuning, checkpoint selection, or any decision". `12:105`: "`data/ood_test.sha256` verified before the
  set is opened". `12:106`: 14 committed before any test-split result was viewed.
- `CLAUDE.md`: do not open without the author's go-ahead; record any opening in 14 § 6.
- **Log row** (`14:107-111`; columns Event | Date | By | Hash verified). The empty row today:
  `| `data/ood_test/` opened (Stage G, once) | | | |`
  Sealed hash on record (line 110): sha256 `2a61cff095b36caf254457d090610794ee53a48ab0451b4525afec8d0cd91bd7`; 11,000 events,
  R ∈ {3,5,20,30} Ω, run_ids 2000–2009.

## 2. Effect of the Stage E and F results

### 2.1 Steps that change

| # | Stage G item | What changed | Proposed new setting |
|---|---|---|---|
| 1 | "R5" everywhere (S6.1–S6.3) | R5 = λ₁ = 0, λ₂ = 0.1, γ = 1, fusion on, 95,259 parameters (`06:632-635`) | Say it once in the Stage G header. R3 (λ₁ only) and R4 (λ₂ only) did not give the R5 gain (`06:929-930`) |
| 2 | Pair for H1–H4, H6, OOD, S6.3 | Gate 4 pairs R5 with R0ⁿ; 14 says R0 | Pair with **R0ⁿ**. Show R0 as an extra row. Needs a 14 amendment (author) |
| 3 | Model list "R1–R7" | R7 dropped (D6, `06:814`); R6 training collapsed twice | Drop R7. Replace R6 by **R6ⁱ**. Add R3, R4, R5-γ0, R8, which exist |
| 4 | A1 "R5 vs R6: R6 ≈ R0"; bootstrap pair "R5 vs R6" | R6ⁱ tests table content at inference; result pending (queue) | Pair becomes R5 vs R6ⁱ (same weights, shuffled table). Rule: the R6ⁱ rule of `06:967` |
| 5 | A3 "R5 vs R7"; family {A1–A3} | R7 not run | A3 cannot be tested. Family becomes {A1, A2}. State it in the 14 amendment |
| 6 | Silent-failure row, T9, H5, A5 | PCRG (G0–G3) not built; G-TF2 replaces it (`07:8-32`) | Scope = the G-TF2 scope: 5 types (stuck, bias ±2 %, bias ±5 %, Gaussian 0.02 p.u., swap), s = 1,2,3, σ₀ = 1e-2, k = 0. Arms: no gate, G-TF2, oracle, residual-only 0.5 %. Thresholds stay those fitted on validation half A. Run on the **test split** only with the author's go-ahead (the G-TF2 file lists the test split as "not in scope") |
| 7 | S6.3 pair "G2 vs G0" | not testable (`07:29-31`) | "G-TF2 vs no gate" on R5, rule = 9 of 9 for stuck. H5 and A5 are reported as "not tested" |
| 8 | S6.4 "for G2, the LOO pass and second forward" | in-sample residual replaced LOO (`07:36`) | Measure G-TF2: in-sample ρ plus the flatline score. Count its passes. No LOO pass. Measure on a quiet box (`12:37`) |
| 9 | S6.2 "fusion-only (S2.6 on R0)" | R0 is not noise-trained | Fusion-only on **R0ⁿ**, β fixed on validation at the nominal table, then frozen for every perturbed table. 1 seed (S2.6 was seed 42) |
| 10 | S6.2 retrain "≈ 7 GPU-h", 13:31 "14 GPU-h" | measured 33.6 GPU-h per fusion run | 33.6 GPU-h each; count is conditional (§ 2.4) |
| 11 | S6.1 "Compute: mostly inference — minutes per model per condition" | R5 full grid = 7.44 GPU-h per seed | Use § 5 |
| 12 | S6.3 bootstrap | scorer saves no predictions | Save per-window predicted class for every cell scored on the test split (int8, about 15 MB per model-seed) |
| 13 | OOD row | σ not stated; grounded split not registered; B2ᵃ in the list | Propose σ = 0 (H3 says k = 0 only), k ∈ {0,5,10}, 3 offsets. Register the grounded split (flag from `03_STAGE_B` S1.3) and A4's meaning before opening |
| 14 | Budget row "B2ᵃ ≈ 20 GPU-h" | deferred to Stage G (D7) | § 3. 33.6 GPU-h as the upper bound |

### 2.2 Why the inference-only test comes first
R6 trained with a shuffled table and collapsed twice (`06:834-956`). The cause is the unnormalised residual: with the wrong table a
β of 0.001 lowers the true class by about 11 logit units. A retrain at ±5 % can fail the same way. R6ⁱ (R5 weights, shuffled table,
no training) is the cheap form of that question, and it is running now. The mismatch study gets the same form first.

### 2.3 Inference-only mismatch test (M-I) — proposal
- **Models.** R5 checkpoints, seeds 42/43/44. Weights and β unchanged. Plus fusion-only R0ⁿ (seed 42, fixed β). R0ⁿ network-only
  numbers do not depend on the table: reuse its Gate 4 scoring.
- **Change.** The signature table only. Build with `scripts/build_signature_table.py --master <perturbed .dss>`, mode daily, one
  recompile per solve (`12:101`). The scorer swaps `phys_table` in the loaded model, as R6ⁱ does with `shuffle_seed`; a new option
  `table_path` is needed (code not written). No `.dss` perturbation files exist yet; they must be written and checked on CPU.
- **Levels.** Phase 1: tap +1, tap −1, impedance ±5 %. Phase 2: impedance ±2 %, ±10 %, ZIP mix, DER or lateral.
- **CPU pre-check per table** (as in the R6 record, `06:946`): true-class ρ on 320 validation windows. Report it. It shows the
  size of the residual shift before any GPU time is spent.
- **Grid.** k = 10 × σ ∈ {0, 5e-4, 2e-3, 1e-2}; k = 0 × σ ∈ {0, 5e-4, 2e-3, 1e-2}; 3 mask offsets. That is the 7 H2/H4 cells of
  Gate 4 plus clean k = 0. Validation split (as Gate 4 and R6ⁱ).
- **Self-checks.** Network-only F1 equals R5's exactly. The table differs from the nominal one. The PSM-only F1 changes.
- **Rule (proposal; the author sets it before the run).** "Graceful at a level" if R5 (perturbed table) fused F1 > R0ⁿ fused F1 in
  all 3 seeds in the 7 H2/H4 cells (21 comparisons). Sign rule, no significance test. Report the whole curve ΔF1 vs level, also
  when it collapses (`14:87-88`).
- **Expectation, not a result.** The effect should be larger at σ = 0 than at σ = 1e-2, because the residual denominator is
  σ_ε² + σ₀². The 18 corpus events with one tap step (`08:81-91`) are named in the report.
- **Cost.** One 8-cell pass = 24 forwards. At the R5 rate (1.80 ms per window-forward: 26,785 s ÷ (60 × 248,000)) this is 2.98 GPU-h per
  seed and table.

| Item | GPU-h |
|---|---|
| Per table: R5 × 3 seeds (8.93) + fusion-only R0ⁿ × 1 seed (2.98) | 11.9 |
| Phase 1 (3 tables) | 35.7 |
| Phase 2 (4 more tables) | 47.6 |
| All 7 tables | 83.3 |
| CPU, tables | about 30 min each by 08:96; `operator/grid.json` records 75 s for the nominal build (84,167 solves, 20 workers) |

### 2.4 Are the mismatch retrains still needed?
**Yes for one retrain; the second is optional.** M-I and a retrain answer different questions.
- M-I: the model learned with the exact table, and the table is wrong at test time. It is the stress test.
- Retrain: the model learned with the approximate table and is scored on the true feeder. This is how a utility would deploy it
  (08:96-98: "confirms the trained variant behaves the same way"). M-I cannot show it.

| M-I outcome | Retrain |
|---|---|
| Graceful (rule holds) | One retrain (seed 42, R5 settings, table = ±5 % impedance, or tap +1). It confirms the trained variant. No second level |
| Not graceful at small levels | One retrain is the only test of whether training adapts. Run it only after the CPU pre-check of true-class ρ. Keep the epoch-3 guard (a failed run costs about 2.6 h, as R6 did) |
| Author drops retrains | State in the paper that mismatch was tested at inference only. Allowed; it weakens the claim "survives ±5 %" (`13:98`) |

A retrain is 1 seed: no significance claim. The second level (13:31) is not needed unless the first one shows an effect.

## 3. B2ᵃ — learned R̂ head (plan, no code)

**Settings.** R5 settings: λ₁ = 0, λ₂ = 0.1, γ = 1, fusion on (β learned from 0), noise and dropout augmentation on, 30 epochs,
1 seed (42), collapse guard on. The only change from R5: a head regresses R, and ρ uses the R-slice at R̂ instead of profiling over
R. Open author items (`S4.4_remaining_arms_plan.md:152-154`): target form, loss weight, and train-time rule.

**Code changes.**
1. `models/pc_stgnn.py`: switch `use_rhat_head`. Create `self.rhat_head = nn.Linear(gnn_dim, 1)` last, after β, type head and
   decoder, so the backbone draws are unchanged (S4.1 byte-identity). Input: node embedding `h` [B,25,64] mean-pooled over nodes.
   Method `_rho_r(x, rhat)`: slice the table at the nearest R-grid point to R̂ (18 points, log distance), keep all 11 types and 17 L
   values (H′ = 187 instead of 3,366), then call the residual on the sliced table. Per-sample table [B,26,187,75] is about 47 MB.
2. `physics/residual.py`: new function `rho_sliced(s, a, table_b, …)` that accepts a per-sample table. `rho` itself is not edited.
   The held-out term `rho_heldout` needs the profiled index mapped back to the full flat index
   (τ·18·17 + i_R·17 + l); a test covers the mapping.
3. `models/common.py::pc_loss`: new term `rhat_loss_weight · MSE(log10 R̂, log10 R)`. Add `rhat_targets_from_metadata`
   (like `type_targets_from_metadata`; the dataset batch already carries `metadata["resistance"]`).
4. `config/s4_4/b2a.yaml` (R5 plus the switch and weight); `scripts/score_s4_4_gate4_run.py` accepts the arm; an
   `rhat` summary (mean and spread of R̂ per true R) is added to the scoring output.

**Loss.** L = CE(z′, y) + 0.1·E_P[ρ(D)] + 1·CE(z, y) + w_R·MSE(log10 R̂, log10 R). Terms as `common.py:119`. The R term counts only
fault windows (label ≠ 0) and excludes w = 20 (`unlabelable_window_mask`). Proposal: target log10 R (the R axis is log-spaced),
w_R = 1.0, pending a balance check like `scripts/check_r2_recon_weight.py` (ratio inside [0.1, 10]).
ρ is computed under `no_grad` (`pc_stgnn.py:164`). So the head learns only from the MSE term, and the table gets no gradient.

**Train-time rule — author decision.**
- Primary (one variable): R̂ in ρ at train and inference. Risk: early R̂ is poor, ρ at the true class is large, and the run may
  collapse as R6 did. The guard stops it at epoch 3 (about 2.6 h lost). Report a collapse as a result.
- Fallback, pre-declared: train with the grid-snapped true R in ρ (teacher forcing), R̂ at inference only. Use it only if the
  primary collapses.

**Null tests (before any run).**
1. Switch off: the model's state dict equals R5's (same seed); with the head on and `w_R = 0`, the backbone outputs are unchanged.
2. Slice = full R axis returns the same ρ as the profiled `rho` (float tolerance).
3. Synthetic window built from table entry (f, τ, R_g, L): with R̂ = R_g, ρ at class f is 0 and equals the profiled value.
4. No gradient reaches `phys_table`. With `w_R = 0` the head gets zero gradient. The MSE mask drops no-fault and w = 20 windows.
5. Oracle check (smoke, labelled as such): score with the true grid-snapped R in place of R̂ on a small validation subset; F1 must be
   within the R5 tolerance. A subset is a code check only, never evidence (`12_CHECKLISTS.md` § 0).

**Data.**
- In-distribution (R ∈ {0.1, 1, 10} Ω): validation grid as S4.4 (no sealed set). Test-split grid inside the master sweep. Also report R̂
  versus true R on validation; a three-cluster R̂ predicts the OOD failure but is not OOD evidence.
- Out-of-distribution (A4, F-G3 curve): **needs the sealed set.** Nothing else holds R ∉ {0.1, 1, 10}. `data/ood_pilot/`
  (R ∈ {5, 20}) is a throwaway set: smoke test only, never reported. B2ᵃ must be trained and frozen before the one opening.
- Cost: 33.6 GPU-h is the upper bound (R5 rate). The sliced ρ is 18× smaller, so the real value is probably lower. Measure with a
  200-batch probe first, as `S4.4_plan_draft.md` did for R1/R2. OOD scoring: § 5.

## 4. Sealed-set checklist (status on 2026-11-02)

| # | Condition before `data/ood_test/` is opened | Status |
|---|---|---|
| 1 | Set unopened and unchanged. Evidence: directory mtime 2026-09-25 (listing of `data/` only); 14 § 6 has no opening row | done |
| 2 | Author's explicit go-ahead for the opening | not done |
| 3 | Hash check ready: `sha256_of_dir` exists in `generate_ood_sets.py:136`; recorded hash in 14:110. The check itself runs at the opening | not done |
| 4 | R0ⁿ ×3, R5 ×3, R0 ×3 trained | done |
| 5 | R1, R2 ×3 trained; R6ⁱ scored; G-TF2 scored (queue and watcher running) | not done |
| 6 | B2ᵃ trained (code not written) | not done |
| 7 | Mismatch retrain, if the author keeps it | not done |
| 8 | R7 | not applicable (dropped, D6) |
| 9 | R6 training arm | not applicable (stopped; replaced by R6ⁱ) |
| 10 | G0–G3 | not applicable (replaced by G-TF2) |
| 11 | Checkpoint SHA-256 list of every model to be scored, written before the opening. Per-arm `manifest.json` files exist; that they hold checkpoint hashes is not checked | not done |
| 12 | Existing code committed: `fault_localization/`, `scripts/`, `config/`, `PC-STGNN/` show no uncommitted changes (`git status`, 2026-11-02) | done |
| 13 | Stage G code written, tested and committed: test-split scorer with saved predictions, dropout-policy scorers, OOD windowing and scorer, `table_path` option, B2ᵃ, bootstrap | not done |
| 14 | OOD windowing and scoring tested end to end on `data/ood_pilot/` (smoke, numbers never reported) | not done |
| 15 | Decision rules committed: 14 itself (9fb50b8, 44d1c0c) | done |
| 16 | Amendments to 14 committed: R0ⁿ pairing, R6ⁱ for A1, A3 not tested, H5/A5 not tested, G-TF2 scope, H3 pooled or per-R, σ and k for OOD, meaning of "collapses" for A4, mismatch rule, Gate 6 | not done |
| 17 | Grounded / ungrounded split registered in 14 (`08:38-40`) | not done |
| 18 | List of models and cells to score on the sealed set fixed in writing, so the opening is once | not done |
| 19 | Log row text ready (draft below) | not done (row in 14 is empty) |
| 20 | Queue complete, GPU free, no other job (CLAUDE.md: one GPU job) | not done |
| 21 | Stage G plan approved by the author (CLAUDE.md: no new GPU run outside an approved plan) | not done |
| 22 | Test-split rule: nothing tuned or selected on the test split or the sealed set (`12:65`) | not applicable until the first test-split run |
| 23 | Pilot numbers (`data/ood_pilot/`) not reported | not applicable |

Draft log row (for the author to paste into 14 § 6 at the opening; not written to 14):
`| data/ood_test/ opened (Stage G, once) | <YYYY-MM-DD HH:MM IST / UTC> | <who>; <script and commit hash> | sha256 <computed> = 2a61cff0…cd91bd7: <yes/no> |`

## 5. Plan — order, GPU-hours, days

**Rates (measured).** R5 training 26.16 h, scoring 7.44 h, total 33.60 h per seed (94,092 / 94,239 / 94,225 s; 26,818 / 26,785 /
26,752 s). R0ⁿ training 8.79 h, scoring 1.34 h, total 10.14 h (4,844 / 4,849 / 4,815 s). Scoring does 60 forwards (5 k × 4 σ × 3
offsets) over 248,000 windows: **R5 1.80 ms, R0ⁿ 0.325 ms per window-forward**. G-TF2 pass: 212.2 s per 124,000 windows (1.71 ms).
Test split 247,000 windows. Sealed set 440,000 windows (derived: 11,000 events × 40; not counted).

**Before Stage G (not in the totals).** Queue: R6ⁱ ×3 = 22.3 h (running since 2026-11-02T06:14Z), R1 about 10.3 h and R2 ×3 about
30.5 h (estimates from the S4.4 probe ratios), then the G-TF2 chain 11.1 h (watcher waiting) = about 74.2 h. Earliest free GPU:
about **5 Nov 2026** (± 1 day). Code, tests and the 14 amendments need no GPU and can proceed now.

| Step | Content | Models × seeds | GPU-h |
|---|---|---|---|
| S1 | M-I Phase 1, validation, 8 cells (§ 2.3) | R5 ×3, fusion-only R0ⁿ ×1, 3 tables | 35.7 |
| S2 | B2ᵃ: train + validation scoring (upper bound; § 3) | 1 seed | 33.6 |
| S3 | Master sweep, test split, random dropout × noise, 60 forwards | R0ⁿ ×3 4.0, R5 ×3 22.2, R2 ×3 4.0, R6ⁱ ×3 (8 cells) 8.9, B2ᵃ ×1 7.4 | 46.5 |
| S4 | Persistent, contiguous, targeted: 3 policies × 4 k × 3 offsets = 36 forwards | R5 ×3 13.3, R0ⁿ ×3 2.4 | 15.7 |
| S5 | Silent failures, test split: R5 63 passes (no gate, G-TF2, oracle, residual-only, clean) | R5 ×3 22.2, R0ⁿ ×3 1.1 | 23.3 |
| S6 | Sealed OOD, 9 forwards (k ∈ {0,5,10} × 3 offsets, σ = 0) | R5 ×3 5.9, R0ⁿ ×3 1.1, R0 ×3 1.1, B2ᵃ ×1 2.0 | 10.1 |
| S7 | T12 timing on a quiet box (assumed, not measured) | — | < 1 |
| | **Core total** | | **165.9 h = 6.9 days** |
| C1 | Conditional: one mismatch retrain (§ 2.4) | 1 seed | +33.6 |
| C2 | Conditional: M-I Phase 2 (4 tables) and a second retrain | | +47.6 + 33.6 |

| Case | GPU-h | Days (T1000, no parallel run) | Ends (from 5 Nov) |
|---|---|---|---|
| Core | 165.9 | 6.9 | about 12 Nov |
| Core + C1 | 199.5 | 8.3 | about 13–14 Nov |
| Core + C1 + C2 | 280.7 | 11.7 | about 17 Nov |

**Order and reason.** S1 first: it is cheap and it decides C1. S2 next (B2ᵃ and any retrain must be frozen before the opening).
Then S3–S5 on the test split (models are frozen; the test split is not sealed). S6 last, one opening, all models in one session.
S7 last, because it needs an idle GPU. S6 comes after C1 if C1 is run.

**Notes.** (a) S3 scores R2 and R6ⁱ because they are in the pre-registered bootstrap pairs; R1, R3, R4, R5-γ0, R8 stay
validation-only (T7), about 21.5 GPU-h saved on the full test grid (1.3 + 1.3 + 4.0 + 7.4 + 7.4). (b) R6ⁱ on 8 cells is 8.9 h; 22.3 h on the full grid. (c) R0 test numbers are
reused from Paper 1 if the metric-consistency check of `12:47-53` passes; otherwise add 4.0 h. (d) The B2ᵃ rate in S3 and S6 is
the R5 rate, an upper bound. (e) Dates are GPU time only. The real critical path is the Stage G code and the amendments.

## 6. Decisions for the author
1. Amend 14: pair with R0ⁿ; A1 via R6ⁱ; A3, H5, A5 not tested; Gate 6; H3 pooled or per-R; OOD σ; "collapses" for A4.
2. Register the grounded / ungrounded split before the opening.
3. Which models go on the test split (proposal: R0ⁿ, R5, R2, R6ⁱ, B2ᵃ).
4. Mismatch: levels of Phase 1; split (validation proposed; a test rerun costs about +35.7 h); M-I rule; one retrain or none.
5. B2ᵃ: target form, w_R, train-time rule.
6. Silent failures on the test split: go-ahead.
7. Approval of this plan, and then of each GPU step.
