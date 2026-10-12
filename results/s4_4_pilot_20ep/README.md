# S4.4 Gate 4 — 20-epoch pilot (superseded)

**Date moved here: 2026-10-12T09:10:59Z (2026-10-12 14:40 IST).**

**Reason.** The Gate 4 epoch count was changed from 20 to 30, per the author's instruction
2026-10-12. The author removed the project deadline that motivated the earlier epoch rule
(the rule in `PC-STGNN/06_STAGE_E_pcstgnn.md`'s "S4.4 — Frozen settings for Gate 4 runs" was a
time-driven rule, picked under deadline pressure that no longer applies) and chose 30 epochs to
match Phase-1's own epoch count. The change does not depend on any Gate 4 result — no R5-vs-R0ⁿ
comparison existed when it was made (R5 seed 1 was stopped mid-epoch-4 training; no scoring had
happened for it at all).

**What this folder holds.** The complete output of the two runs that existed under
`results/s4_4/` at the moment of the change, moved here unmodified (no file deleted, no file
edited) so the 20-epoch pilot data stays available for reference:

- `r0n/pcstgnn/seed_00042/` — R0ⁿ seed 1, 20 epochs, **fully trained and fully scored**
  (`checkpoint_best.pt`, `metrics.json`, `metrics_by_epoch.csv`, `scoring_full_grid.json` all
  present). Best epoch 18, best validation graph-F1 = 0.8126004083285524.
- `r5/pcstgnn/seed_00042/` — R5 seed 1, 20-epoch config, **training only, stopped mid-run**
  (epoch 3 of 20 complete when stopped; no `metrics.json`, no checkpoint-complete marker, no
  scoring was ever run for this arm in the 20-epoch pilot).
- `r0n/manifest.json`, `r5/manifest.json` — the per-arm dataset manifests from the pilot run.
- `gate4_progress.json` — the orchestrator's own step-by-step log of the pilot queue, ending at
  R5 seed 1's `training_started` entry.
- `gate4_run.log` — stdout/stderr of the pilot queue process (includes the earlier tooling
  incident's OOM traceback from the first attempt at this epoch count, and the successful
  resume/skip of R0ⁿ seed 1 on the second attempt).

**Status of each run, for anyone reading this later.** R0ⁿ seed 1's 20-epoch result is a
complete, valid, scored data point — it is simply not part of the (now 30-epoch) Gate 4 record
going forward. R5 seed 1 never finished training at 20 epochs and was never scored; its
checkpoint here is a mid-run snapshot, not a result.

**Current Gate 4 record.** `PC-STGNN/06_STAGE_E_pcstgnn.md`, section "S4.4 — Frozen settings
for Gate 4 runs", amended 2026-10-12 to 30 epochs. `results/s4_4/` now holds the fresh 30-epoch
runs.
