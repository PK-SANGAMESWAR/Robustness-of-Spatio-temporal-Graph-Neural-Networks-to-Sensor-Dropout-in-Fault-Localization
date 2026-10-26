# PC-STGNN result — thesis content record (rewritten 2026-10-22)

**State.** The text is **in** `thesis-for-college/MTech_thesis.tex`. The previous version of this
file said "not yet inserted"; that was out of date — the earlier draft was already in the `.tex`.
On 2026-10-22 the four spots were rewritten from the committed Gate 4 result (commit `d40916a`,
`results/s4_4/`). No value from the S5.1 job was used. A copy of the `.tex` before this rewrite is
in the session scratchpad (`MTech_thesis.before_gate4_rewrite.tex`); the previous version of this
file is in git history only if it was committed (it was untracked).

Build check (scratch copy; pdflatex, bibtex, pdflatex ×2): 0 errors, 0 overfull boxes, 5 underfull
boxes (the same 5 as before the change, front matter), no undefined references or citations.
`tab:gate4` = Table 4.14 (p. 57), new `tab:decomp` = Table 4.15 (p. 58). Spell check (aspell,
en_US): technical terms only. **Page count:** main text ends on p. 64 (was p. 62 before this
change, already over the 60-page limit of `thesis-rules.md`). The PDF in `thesis-for-college/`
was **not** replaced.

## 1. What was written, and where

| Spot | `.tex` location | Content |
|---|---|---|
| Abstract | line 104, last 4 sentences | 20/20 conditions, every seed; largest gain 0.20 at k = 10; gain arises in the fusion step; reference still improving; R6 pending |
| § 4.9.4 "Comparison With the Noise-Augmented Reference" | from line 1285 | Decision rule and Holm family; Table 4.14 (per-seed F1, both models, 20 cells, mean difference, p_Holm for the 7-cell family); significance vs family size; reference still improving (best epochs 30/29/27 vs 19/29/17); Table 4.15 (fusion decomposition at each k and σ); attribution paragraph with PENDING markers |
| § 4.10 Discussion, fourth finding | "Fourth, the physical operator ..." | Tuning evidence marked as one seed; three-seed result; decomposition; attribution pending |
| Ch. 5 RQ5 | "\textbf{RQ5: fusion in the network.}" | One-seed design evidence; λ₁ = 0; held-out term untested; 20/20 and 7/7 Holm; p_Holm = 0.009 at k = 10; not significant at k = 0, 0.01 p.u.; fusion-step finding; reference still improving; ablations pending |

## 2. PENDING markers in the `.tex` (no numbers written for them)

| Marker | Where | Sentence needs |
|---|---|---|
| `[PENDING: R6]` | Abstract; § 4.9.4; Discussion; RQ5 | Whether the gain needs the correct signature-to-location map |
| `[PENDING: R4]` | § 4.9.4; Discussion; RQ5 | Effect of the held-out term (λ₂) |
| `[PENDING: R5-$\gamma$0]` | § 4.9.4; Discussion; RQ5 | Effect of the auxiliary loss on the fused score at 30 epochs |
| `[PENDING: R2]` | § 4.9.4; RQ5 | Learned decoder of matched size |
| `[PENDING: R7]` | § 4.9.4; RQ5 | Wider backbone |
| ~~`[PENDING: R8]`~~ | RQ5 | **Filled 2026-10-24** from `results/s4_4/r8/` (1 seed, no significance claimed): no deferral to the physics at gamma=1 without noise; 0.008-0.014 below R5 (seed 42) without test noise; 0.075-0.259 below with test noise (15/15). Wording edits for consistency: Ch. 3 "are both needed" -> "were both included"; subsection title "Are Needed" -> "Were Introduced"; Discussion 4th finding. |

The markers use internal arm names and are placeholder text, which `thesis-rules.md` §3–4 forbid in
the final thesis. They must be replaced (by a result or by a removal of the sentence) before
submission.

## 3. Source numbers (validation macro-F1, `results/s4_4/*/pcstgnn/seed_*/scoring_full_grid.json`)

- 60 of 60 seed–condition differences (R5 − R0ⁿ) are positive; the smallest is +0.0083 (seed 42,
  k = 0, σ = 0.01).
- Holm, 7-cell family (k ∈ {1,2,5,10} at σ = 0; k = 10 at σ ∈ {5e-4, 2e-3, 1e-2}): 0.020, 0.020,
  0.017, 0.009, 0.009, 0.017, 0.020 — all < 0.05.
- Holm, 16 cells with k > 0: 5 of 16 < 0.05 (k = 5 at σ ≤ 5e-4; k = 10 at σ ≤ 2e-3).
- Holm, all 20 cells: 2 of 20 < 0.05 (k = 10 at σ = 0 and 5e-4, both 0.025).
- k = 0, σ = 0.01: +0.025, raw p = 0.140.
- R5 network-only F1 < R0ⁿ F1 in 45 of 60 seed–condition pairs (mean deficit per cell 0.014–0.041;
  seed 43 is the exception at most k ≤ 5).
- Fused F1 − max(network-only, PSM) = +0.038 to +0.132 in every cell.
- Training curves: R0ⁿ best epochs 30, 29, 27; mean(26–30) − mean(21–25) = +0.016, +0.016, +0.010.
  R5 best epochs 19, 29, 17; same difference +0.003, +0.002, +0.005.
- R5 config: λ₁ = 0, λ₂ = 0.1, γ = 1 (`metrics.json`).

## 4. Places where the data supports a weaker claim than the earlier draft text

1. **Abstract (old):** "scores higher ... at every tested combination of noise and sensor loss". True
   for the sign (60/60), but significance after Holm holds only within the 7-cell family; over all
   20 cells only 2 survive. The old text also did not say that the reference was still improving.
2. **§ 4.9.4 (old):** Holm "across the five conditions". That family was chosen for the table, it
   included clean k = 0 (which the decision rule excludes), and it is not the 7-cell family of the
   project record. Replaced.
3. **§ 4.9.4 (old):** "The fusion therefore helps most exactly where the physical signal is most
   needed." Only half supported: the gain is largest under sensor loss, but it is smallest at high
   noise, where PSM is weakest (F1 0.496). Removed.
4. **RQ5 (old):** "$p_{\mathrm{Holm}}=0.006$" came from the 5-cell family. In the 7-cell family it
   is 0.009.
5. **RQ5 (old):** "the one exception, the highest noise level with no sensors lost". Over the full
   grid, the raw p is also above 0.05 at σ = 0.01 for k = 1 (0.12), k = 2 (0.096) and k = 5 (0.051).
   Not one exception.
6. **RQ5 and Discussion (old):** "Without noise augmentation the network learns to defer entirely
   ...", "without an auxiliary loss ... it still fails", "a stable trained fusion needed noise
   augmentation and an auxiliary loss". The evidence is the one-seed tuning (5–15 epochs). Not
   confirmed at 30 epochs (R8, R5-γ0 not run). Now marked "in the tuning at one seed".
7. **§ 4.9.3 (unchanged) and RQ5 (old):** "the held-out term was neutral to slightly positive".
   One seed, margin 0.0019, not a tested difference (§ 4.9.3 says so). The final model keeps it, and
   no arm has isolated it (R4 not run). RQ5 now says "its effect has not been tested". **§ 4.9.3
   still contains "neutral to slightly positive"** — left as it is, because it is qualified there.
8. **§ 4.9.3 (unchanged):** "the fused prediction on fault windows agrees with PSM for 91.5 to
   94.7% ... so at clean k = 0 the localization is still largely physical." This is the 15-epoch,
   one-seed tuning model. For the final 30-epoch models the scorer counts agreement over all
   windows only: 71.3 % with PSM and 92.1 % with the network at clean k = 0. The fault-window figure
   was not computed for the final models, so "largely physical" cannot be carried to them. Left
   unchanged; consider qualifying it.
9. **Attribution (old, implicit):** the old text called the gain the effect of "the fusion". R5
   differs from R0ⁿ in three elements (fusion, λ₂, γ). The data support only "the gain is realized
   in the fusion step" (the network branch alone is below the reference). Whether physics is the
   cause needs R6; λ₂ and γ need R4 and R5-γ0.
10. **Reference not converged (missing in old text):** R0ⁿ was still improving at epoch 30. The
    margins are against a reference that had not reached its final level.
11. **Project record, not thesis — `06_STAGE_E_pcstgnn.md` § GATE 4 (not changed, per instruction):**
    (a) the Holm-adjusted p values there skip the step-down maximum (k=1 0.0175 → 0.020;
    k=10/σ=5e-4 0.0078 → 0.009; k=10/σ=2e-3 0.0149 → 0.017; k=10/σ=1e-2 0.0133 → 0.020); the
    conclusion (all < 0.05) does not change. (b) The text says R5 > R0ⁿ "at every noise operating
    point (k ∈ {1,2,5,10} × σ ∈ {...})", which is 16 cells, but the significance table has 7. The
    sign holds in all 16; Holm significance over 16 holds in 5. (c) The 7-cell family was not
    listed before the runs; the frozen rule says only "in the pre-registered conditions ... never at
    clean k = 0".
