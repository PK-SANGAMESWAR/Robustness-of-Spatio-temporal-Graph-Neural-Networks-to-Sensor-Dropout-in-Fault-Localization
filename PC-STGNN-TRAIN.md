# PC-STGNN Gate 4 — full context dump (written 2026-10-16, assume zero memory of the prior chat)

**Read this whole file before doing anything** if you're picking this up cold. It covers the
entire Gate 4 saga: pre-registration, a real tooling bug and its fix, an epoch-count amendment,
current live status, what to do when the queue finishes, and what NOT to do. Also see the two
related memory files, which may be more current than this snapshot:
`pcstgnn-gate4-status.md` and `thesis-college-workflow.md` under the memory directory.

---

## 0. TL;DR — where things stand as of 2026-10-16T13:51Z

- **Gate 4** (comparing two trained model variants, R0ⁿ and R5) is **5 of 6 runs complete**.
  The 6th and last run, **R5 seed 3** (seed=44), is training (epoch 3/30 as of this writing).
- No collapse, no error, queue has never stopped on this attempt.
- A background watcher (`bash /tmp/wait_for_gate4_completion.sh`, launched via the Bash tool's
  `run_in_background`) polls every 60s and will fire a completion notification the moment the
  queue finishes or stops — **check for that notification before assuming the queue is still
  running**; if a new session has no memory of launching it, check `ps aux | grep
  wait_for_gate4` to see if it's still alive, and re-launch it (see §6) if not.
- **Results so far are good and consistent**: R5 beats R0ⁿ on every checked cell in both
  completed seed pairs, including the conditions Gate 4 actually judges on (not clean k=0) —
  see §7. Not yet a formal pass (need all 3 seeds).
- **When the queue finishes**, there are specific follow-up actions required — see §6. They
  were NOT fully done as of this file being written (the queue hadn't finished yet).

---

## 1. What Gate 4 is and why it exists

This repo has (at least) two research threads:
1. An **older, already-submitted** paper: dropout-aware STGNN fault localization under partial
   µPMU observability (Paper ID 1170, ICMLDE → Procedia Computer Science). It's now in
   post-review revision in a separate worktree — see `resubmit/1170_revision_plan.md`,
   `resubmit/CHANGELOG-1170.md`. Decision email 2026-10-10, revision deadline 2026-10-17.
2. A **newer, in-progress** extension: **PC-STGNN**, which fuses a fixed physics operator
   (precomputed fault-signature table) into the same backbone. This file is about that thread.
   Planning docs live under `PC-STGNN/*.md`; `PC-STGNN/06_STAGE_E_pcstgnn.md` is the living
   source of truth for the current stage (Stage E, the core ablation).

**Gate 4** is a specific, pre-registered decision point inside Stage E: does the physics fusion
(**R5**) actually beat the matched noise-augmented backbone with no physics (**R0ⁿ**, not the
original R0 — R0ⁿ adds noise augmentation, which R0 never had, so it's the fair comparator)?
The rule, quoted verbatim from `06_STAGE_E_pcstgnn.md`'s "GATE 4" section:

> - [ ] **R5 > R0ⁿ** (not R0) on validation, in the pre-registered conditions, sign-consistent
>       across 3 seeds — read at the noise operating points, never at clean k = 0 (S2.0ᵃ)

If it fails ("R5 ≈ R0ⁿ"), the project's own plan says: do NOT proceed to Stage F expecting a
rescue — go straight to Stage G and write the "physics buys generalisation, not in-distribution
accuracy" paper instead.

**Both R0ⁿ and R5 are `PCSTGNN` instances** (`fault_localization/models/pc_stgnn.py`):
- R0ⁿ: `use_fusion=False`, no type head, no decoder, no physics table — structurally identical
  to plain RGSAGE-Max (verified: 95,258 parameters). Noise augmentation is on, which requires
  routing through the `pc_loss`/`PhysicsDropoutLoader` path even though there's no physics term
  (λ₁=λ₂=γ=0) — see `experiments/runner.py:258-259`, "noise_augmentation is only wired through
  the pc_loss path."
- R5: `use_fusion=True` (β learned, init 0), real physics table loaded, λ₁=0, λ₂=0.1, γ=1 — the
  S4.3ᵉ-selected point (95,259 parameters — exactly one scalar, `beta`, more than R0ⁿ).

Both trained for **30 epochs**, 3 seeds each (42, 43, 44 — referred to as "seed 1/2/3" in run
order), dataset `dataset_fullscale_50` (1.65M windows, 825 scenarios × 50 runs, event-level
70/15/15 split, **validation split only** — `training.skip_test_eval: true`, test split never
touched).

---

## 2. Timeline of this whole effort (chronological, so you understand WHY things are the way they are)

1. **2026-10-08**: S4.3ᵉ (the λ-tuning sweep that picked R5's λ₁=0, λ₂=0.1) finished.
2. **2026-10-09**: Gate 4 pre-registered into `06_STAGE_E_pcstgnn.md` (a new subsection, "S4.4 —
   Frozen settings for Gate 4 runs"). Epoch count initially decided as **20**, via a fixed rule:
   mean signed Δ(fused F1 @ k=0, epoch15−epoch10) over the 5 S4.3ᵉ arms = −0.0038 < 0.005 ⟹ 20.
   (Caveat on record: the *absolute*-difference mean is 0.0061, which would have flipped the
   rule to 30 — flagged, not resolved, the signed rule was applied literally as specified.)
   Gate 4 queue started training.
3. **2026-10-09, same day**: after R0ⁿ seed 1 finished training cleanly, the **scorer crashed
   with CUDA OOM** (`scripts/score_s4_4_gate4_run.py`, first version). Root cause: it cached the
   *entire* 248,000-window validation split as GPU tensors twice over (once clean, once per
   noised copy) — ~6GB on an 8GB T1000. This inverted the pattern
   `dropout/evaluator.py::evaluate_dropout_curve` already uses (loop the (k,σ,offset)
   combinations *inside* one streaming pass over the DataLoader).
   - **Fixed 2026-10-12**: rewrote the scorer as a single streaming pass, holding only the
     current 32-window batch on GPU. Also made every window's noise/dropout draw deterministic
     from `(level, mask-seed offset, that window's fixed position in dataset.records)` —
     independent of batch order/size/shared RNG state (previously, `apply_sensor_dropout_batch`
     draws from one shared RNG stream per batch call, so a window's mask depended on its
     *position within whatever batch happened to contain it* — a real but subtle bug).
   - **Equivalence-tested**: old cache-then-loop structure vs. new streaming structure, same
     seed formula, 2% of validation windows, all 60 (σ,k,offset) cells → **max |Δ F1| = 0.0**
     (tolerance was 1e-6). Full split at σ=0,k=0 with the new scorer matched the logged
     `best_validation_graph_f1` exactly (diff 0.0). Both checks passed.
   - Orchestrator (`scripts/run_s4_4_gate4.py`) was made **resumable**: skip training for an
     (arm, seed) only if `checkpoint_best.pt` + `metrics.json` exist AND the saved `config.yaml`
     matches the frozen `config/s4_4/<arm>.yaml` exactly (stops on any mismatch, doesn't
     silently retrain); skip scoring only if `scoring_full_grid.json` exists.
   - Documented in `06_STAGE_E_pcstgnn.md`'s "S4.4 — Tooling incident" subsection.
4. **2026-10-12**: after the fix, resumed — R0ⁿ seed 1 correctly skipped (already done), R5
   seed 1 started training.
5. **2026-10-12, later same day**: **epoch count amended 20 → 30.** Reason given by the
   project's author: the deadline that motivated the time-driven 20-epoch rule was removed, and
   30 matches "Phase-1"'s (the older paper's) own epoch count. **This did not depend on any
   Gate 4 result** — at the time, R5 seed 1 had only reached epoch 3 of 20, unscored.
   - R5 seed 1 (training, epoch 3/20) and R0ⁿ seed 1 (fully trained+scored at 20 epochs,
     best_validation_graph_f1=0.8126004083285524, best epoch 18) were **both stopped and moved
     intact to `results/s4_4_pilot_20ep/`** — nothing deleted, full folder structure preserved,
     with a `README.md` recording why. **Do not use anything in that folder as a Gate 4
     result** — it's an archived pilot at the wrong epoch count.
   - `config/s4_4/{r0n,r5}.yaml` updated `epochs: 20` → `epochs: 30`. Everything else in those
     configs (λ/γ/β/fusion, noise grid, dropout grid, collapse guard at epoch 3, `skip_test_eval`)
     is unchanged.
   - `06_STAGE_E_pcstgnn.md` and `S4.4_plan_draft.md` both amended using strikethrough +
     italicized "superseded" notes — **the old 20-epoch value and its rule are kept, not
     deleted**, per this repo's convention for historical corrections.
   - A **heartbeat** feature was added to the orchestrator: a background thread watches each
     run's `metrics_by_epoch.csv` for new rows and appends `{"epoch": N, "utc": ...}` to that
     run's `heartbeat.log`, so progress is visible without waiting for the run to finish.
   - Queue **restarted from zero** at 30 epochs (no checkpoint reuse across the epoch change).
6. **2026-10-12 through 2026-10-16**: queue has run R0ⁿ seed1 → R5 seed1 → R0ⁿ seed2 → R5 seed2
   → R0ⁿ seed3 → (now) R5 seed3, each followed immediately by its own scoring, fully
   automatically, one at a time, no manual intervention, no collapse, no error.
7. **2026-10-15**: a separate, unrelated task happened in between — a repo-wide markdown
   staleness audit (3 parallel subagents + some files I fixed myself) found and corrected
   several stale docs, none of which touch the Gate 4 mechanics described here. Notably fixed:
   `06_STAGE_E_pcstgnn.md`'s old "S4.4 not started" line, `CLAUDE.md`/`README.md`'s stale "paper
   is a submission draft" claim (it's actually been submitted, reviewed, and is in post-review
   revision), and several PPT files claiming S4.3ᵉ was "still running" (it finished 2026-10-08).
   None of this is committed yet (see §6).

---

## 3. Real measured GPU costs — use these, not the project's own budget doc

`PC-STGNN/13_BUDGET_TIMELINE.md` estimates 30-epoch rates at **6.9h** (non-physics) and
**≈20.3h** (physics, i.e. any arm with `use_fusion=True` or λ₁/λ₂>0). **These are stale/wrong by
~27-29%** — they predate any real `forward_pc` GPU measurement. The actual measured rates from
this Gate 4 run (averaged over the runs completed so far):

| | Training (30 epochs) | Scoring (full val-split grid) | Total/run |
|---|---|---|---|
| **R0ⁿ** | **8.80h** (avg of 3 real runs) | **1.35h** (avg of 2 real runs) | **10.15h** |
| **R5** | **26.16h** (avg of 2 real runs) | **7.45h** (avg of 2 real runs) | **33.59h** |

Scoring cost does **not** depend on epoch count (it's a fixed cost per finished checkpoint, over
the fixed-size validation split) — only training time scales with epochs.

**Implication for anything beyond Gate 4**: the remaining 9 S4.4 ablation arms (R1, R2, R3, R4,
R6, R7, R8, B2ᵃ, R5-γ0 — 15 more run-seeds total, per the seed counts in
`06_STAGE_E_pcstgnn.md`'s S4.4 table) would cost **≈387 GPU-h ≈ 16 sequential days** on this
single T1000, at these real rates. On top of that: **Stage F has no code yet** (only β is
implemented in `pc_stgnn.py`; no α, no c, no gate mechanism), **Stage G's primary OOD set
(`data/ood_test/`) is raw OpenDSS output only, never windowed/labeled**, and **Stage H (writing)
was separately budgeted at 7 weeks** in the project's own original schedule. Full PC-STGNN
(everything, eval to metrics) is **not realistically finishable in any short ~10-day window** —
this was asked and answered explicitly earlier in this effort; the reasoning above is the full
grounding if asked again.

---

## 4. Files and scripts — the map

**Config** (frozen, do not change without a new pre-registration amendment):
- `config/s4_4/r0n.yaml`, `config/s4_4/r5.yaml` — the two Gate 4 configs, `epochs: 30`.

**Scripts**:
- `scripts/run_s4_4_gate4.py` — the orchestrator. Trains+scores R0ⁿ/R5 × 3 seeds in order
  (seed1=42, seed2=43, seed3=44), one at a time, resumable, heartbeat, GPU-memory-releasing,
  stops (does not auto-restart) on collapse/exception/config-mismatch.
- `scripts/score_s4_4_gate4_run.py` — the (fixed, streaming) scorer. Importable as
  `score_run(arm, seed, max_windows=None)`.
- `scripts/measure_s4_4_timing.py`, `scripts/measure_s4_4_scoring_time.py` — one-off speed
  probes from earlier planning rounds, not part of the live queue.

**Results** (all under `results/s4_4/<arm>/pcstgnn/seed_<seed:05d>/`, arm ∈ {r0n, r5}, seed ∈
{00042, 00043, 00044}):
- `config.yaml`, `environment.json`, `dataset_manifest.json` — written at run start.
- `checkpoint_best.pt`, `checkpoint_last.pt`, `metrics_by_epoch.csv`, `metrics.json` — written
  during/after training. `metrics.json` has `best_validation_graph_f1` and
  `collapse_abort: {epoch, max_class0_frac, min_net_f1, triggered}`.
- `heartbeat.log` — one JSON line per epoch, `{"epoch": N, "utc": ...}`.
- `scoring_full_grid.json` — written after scoring. Has `cells` keyed `"k{K}_sigma{S}"`, each
  with `f1_fused_mean`/`f1_fused_std`/`f1_fused_per_offset`, and for R5 additionally
  `f1_network_only_mean`, `f1_psm_mean`, `fused_eq_psm_mean`, `fused_eq_network_only_mean`.
- `results/s4_4/gate4_progress.json` — append-only log of every queue step (training_started,
  training_done, scoring_started, scoring_done, config_mismatch, queue_complete).
- `results/s4_4/QUEUE_STOPPED.json` — **only exists if the queue has stopped on a problem**
  (collapse / exception / config mismatch). Absence of this file = queue is fine (running or
  completed normally).
- `results/s4_4_pilot_20ep/` — the archived 20-epoch pilot (R0ⁿ seed1 complete, R5 seed1
  partial). **Not a Gate 4 result. Do not use.**

**Docs**:
- `PC-STGNN/06_STAGE_E_pcstgnn.md` — live source of truth. "S4.4 — Frozen settings for Gate 4
  runs" has the frozen settings + two amendment notes + the current completion estimate. "S4.4
  — Tooling incident" documents the OOM bug. "GATE 4" (further down) is the original checklist.
- `PC-STGNN/S4.4_plan_draft.md` — the working draft this was all planned in, with more granular
  revision history (6 rounds) than the live doc.

---

## 5. How to check status right now (exact commands)

```bash
cd /home/ubuntu/Fault-Localization
ps aux | grep -E "run_s4_4_gate4|score_s4_4" | grep -v grep   # is it running?
cat results/s4_4/QUEUE_STOPPED.json 2>&1                       # exists? queue stopped on a problem
cat results/s4_4/gate4_progress.json | tail -10                 # last few queue events
tail -3 results/s4_4/r5/pcstgnn/seed_00044/heartbeat.log 2>&1  # current epoch, if R5 seed3 still training
```

To check if the background completion-watcher is still alive:
```bash
ps aux | grep wait_for_gate4 | grep -v grep
```
If it's gone and the queue is still running, re-launch it (adjust paths if this session's
scratchpad differs):
```bash
cat > /tmp/wait_for_gate4_completion.sh << 'EOF'
#!/bin/bash
while true; do
  if [ -f results/s4_4/QUEUE_STOPPED.json ]; then
    echo "GATE4_STOPPED"; cat results/s4_4/QUEUE_STOPPED.json; exit 1
  fi
  if python3 -c "
import json,sys
d = json.load(open('results/s4_4/gate4_progress.json'))
sys.exit(0 if d[-1].get('stage')=='queue_complete' else 1)
" 2>/dev/null; then
    echo "GATE4_QUEUE_COMPLETE"; exit 0
  fi
  sleep 60
done
EOF
chmod +x /tmp/wait_for_gate4_completion.sh
# then launch via the Bash tool with run_in_background: true
```

---

## 6. What to do when the queue finishes — DO NOT SKIP THESE

The user's explicit instruction (2026-10-16): *"after the train and scoring is done go and
update readme and also commit the folders and files needed and then continue with the process
dont wait for me to tell you."* My interpretation (stated to the user, not contradicted): let
the already-running queue finish naturally, then automatically do the wrap-up below — but do
**not** autonomously launch the other 9 S4.4 ablation arms (R1-R8, B2ᵃ, R5-γ0) without being
asked, since several of them still have open, unresolved design questions (see §8).

**On successful completion (`queue_complete`), do, in order:**

1. **Compute the formal Gate 4 verdict.** Pull all 3 seeds' R0ⁿ and R5 results from
   `results/s4_4/{r0n,r5}/pcstgnn/seed_000{42,43,44}/scoring_full_grid.json`. Check
   sign-consistency of (R5 − R0ⁿ) at the pre-registered non-clean-k=0 conditions (k=10 and/or
   the noise levels — re-read `06_STAGE_E_pcstgnn.md`'s S2.0ᵃ cross-reference for exactly which
   cells count as "pre-registered" before asserting a verdict). As of 2/3 seeds (seed42,
   seed43), R5 wins every checked cell by a wide margin (see §7) — seed44 is the deciding one.
2. **Update `PC-STGNN/06_STAGE_E_pcstgnn.md`**: tick the GATE 4 checklist item (or report
   fail, honestly, if seed44 breaks sign-consistency) with the actual 3-seed numbers.
3. **Update `README.md`** with current PC-STGNN/Gate 4 status (what it is, result, date).
4. **Update the M.Tech thesis** (`thesis-for-college/`) — see the `thesis-college-workflow`
   memory file for the exact pending items: Table 4.14 (reference vs PC-STGNN, validation, 3
   seeds), Section 4.9.4 text, Chapter 5's RQ5 conclusion (remove the pending sentence), and the
   abstract. **Use only `results/s4_4/` (the 30-epoch finals), never `results/s4_4_pilot_20ep/`.**
   Follow `thesis-for-college/thesis-rules.md`'s workflow (one chapter, compile, check, report,
   stop for approval) — but per the user's "don't wait for me" instruction for Gate-4-triggered
   updates, at minimum fill the numbers and report, rather than blocking entirely.
5. **Commit.** At minimum: `results/s4_4/` (all 6 runs' final output), the updated
   `06_STAGE_E_pcstgnn.md`, `README.md`, and whatever thesis files changed. Follow the repo's
   existing commit-message convention (see `git log` for examples from this same effort, e.g.
   "S4.4 Gate 4 pre-registration: ...", "S4.4 tooling fix: ...", "S4.4 amendment: ..."). **Do
   not push** unless separately asked — every commit in this whole effort has been local-only.
6. **Also still pending from the 2026-10-15 markdown audit** (unrelated to Gate 4 mechanically,
   but also uncommitted): fixes across ~15 files (`CLAUDE.md`, `README.md`,
   `FAULTQUERY-ARCHITECTURE.md`, `PC-STGNN-PLAN.md`, several `PC-STGNN/*.md`, several
   `PPT*.md`/`PPT2/*.md`, `NEXT.md`, `NEW-STRATEGY.md`). Check `git status -- '*.md'` — these
   are all still just working-tree edits. Consider bundling them into the same commit pass or a
   separate one; nothing about them is time-sensitive the way the Gate 4 result is.
7. **Continue monitoring/reporting as before** after the above — i.e. resume normal
   status-check behavior, don't go silent.

**On failure (`QUEUE_STOPPED.json` exists) — collapse, exception, or config mismatch:**
**Stop. Report it. Do not restart the queue without the user's explicit permission.** This rule
has held throughout this entire effort; nothing changes it.

---

## 7. Results so far (2026-10-16, 2 of 3 seed-pairs complete)

Clean validation F1 (k=0, σ=0):

| Seed | R0ⁿ | R5 | Δ |
|---|---|---|---|
| 42 | 0.8461 | 0.8951 | +0.049 |
| 43 | 0.8309 | 0.9019 | +0.071 |

At the pre-registered (non-clean) conditions:

| Condition | Seed | R0ⁿ | R5 | Δ |
|---|---|---|---|---|
| k=10, σ=0 | 42 | 0.6689 | 0.8563 | **+0.187** |
| k=10, σ=0 | 43 | 0.6497 | 0.8556 | **+0.206** |
| k=10, σ=0.01 | 42 | 0.5948 | 0.6729 | **+0.078** |
| k=10, σ=0.01 | 43 | 0.5721 | 0.6896 | **+0.118** |

R5 wins every checked cell, both seeds, by a wide and *growing* margin under stress (k=10) —
exactly the pattern the project's physics-fusion hypothesis predicts. No collapses in any of
the 5 completed runs. Seed 3 (R5 seed44, currently training) is the deciding sign-consistency
check.

---

## 8. Open items — not resolved, don't guess on these

- **R7's "wider" width** is undefined anywhere in docs or code (confirmed by direct search —
  `gnn_hidden_dim` only ever defaults to 64 everywhere). Any R7 run needs this decided first.
- **B2ᵃ's learned R̂ head has no implementation** (`pc_stgnn.py` has no `r_hat`/resistance-head
  code). A short implementation plan exists in `S4.4_plan_draft.md`'s Part 6/report history but
  nothing is coded.
- **R5-γ0's seed count** was proposed as 1 (by analogy to R8's precedent) but never explicitly
  confirmed by the author as a rule — flagged, not settled.
- **R3/R4's exact fusion-off semantics and the γ-redundancy question** were resolved by an
  explicit author decision (fusion OFF, γ=0 when fusion is off) — this one IS settled, see
  `S4.4_plan_draft.md` Part 2 of the 2026-10-09 round for the full reasoning if it ever needs
  re-deriving.

---

## 9. Style/process notes for whoever (whatever) picks this up

- This repo has an established convention for correcting stale doc content: **strikethrough the
  old text + an italicized "Stale/Superseded as of DATE" note with the reason, keep the
  original, never silently delete.** Bracketed tags like `[Added 2026-09-23]` or `[Corrected
  2026-10-04]` are immutable historical event-timestamps, not "last reviewed" stamps — never
  change them to today's date.
- Every commit in this effort has been **local only**, with explicit, given commit messages,
  following "only commit when asked" / "never push unless asked."
- The user has been very deliberate and specific about every GPU-consuming decision (exact
  epoch counts, exact run order, exact seed-to-label mapping, explicit stop conditions). Don't
  infer authorization for new GPU-hour commitments beyond what's explicitly asked — this
  includes **not** starting any of the remaining 9 S4.4 arms without being told to, even though
  Gate 4 finishing might seem like a natural trigger to "keep going" with the rest of the table.
