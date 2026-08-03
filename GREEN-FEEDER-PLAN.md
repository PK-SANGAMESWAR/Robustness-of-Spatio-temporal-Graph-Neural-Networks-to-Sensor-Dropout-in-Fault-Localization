# GREEN-FEEDER-PLAN.md — implementation plan for the reconfigured ("green") feeder experiment

**Status:** PLAN ONLY. Execute **after all SDR-contribution training finishes** (SDR-STGNN 42/43/44
and dropout-aware RGSAGE-Max 42/43/44 + the final multi-seed figure rebuild). Do **not** start while
the GPU chain is running. *(Revised to incorporate an expert review — normalization timing verified,
confound decomposition added, graph-gate moved earlier, script choice + size + seed-reporting +
deadline all locked.)*

**One-line goal:** add a second robustness axis — *topology reconfiguration* — to complement the
sensor-dropout study, as a **generalization test with NO retraining**: take the already-trained
checkpoints and evaluate them on a green-feeder test set (its different graph + shifted voltages).

---

## 1. Framing (why this is cheap)
The "green" feeder is the standard IEEE-123 reconfiguration — **open the 60–160 tie switch, close the
54–94 switch** (`GREEN_FEEDER_OPEN_SWITCH_BUSES=(60,160)`, `GREEN_FEEDER_CLOSE_SWITCH_BUSES=(54,94)`
in `config/constants.py`). It changes (a) power flow → measured voltages, and (b) the measured-only
graph edges (a different spanning tree over the **same** 25 μPMUs; placement unchanged).

We do **not** retrain. `run_dropout_study.py` rebuilds each model with `test_ds.edge_index` (which for a
green data-root is the **green** graph) and loads the default-trained weights — the edge set is a
buffer, not a learned weight — so this *is* the "does the model rely on a fixed graph structure?" test.
We can also sweep the dropout levels on green to get green degradation curves.

## 2. What already exists vs. the gap
| Piece | Status |
|---|---|
| Green switch in the simulator (`simulate_fault_event(..., feeder_config="green")`) | ✅ exists; writes raw events with a `_green` suffix (`runner.py: event_filename`) so they coexist with default |
| `stage_full_generation(..., feeder_config="green")` | ✅ exists (the sim stage accepts the flag) |
| Green graph in the builder (`processing/build_dataset.build(..., feeder_config="green")`) | ✅ exists (Step 1 builds the green measured-only graph) |
| Evaluate checkpoints on a green data-root (`run_dropout_study.py --data-root …`) | ✅ works unchanged (rebuilds model on green edge_index) |
| Normalization integrity | ✅ **verified easy** — normalization is **load-time** (`data/dataset.py:61`; windows stored raw), so swapping in the default `norm_stats.json` is sufficient, no `build()` change (§4.3) |
| **Orchestration** to generate a green *corpus* end-to-end | ❌ **gap** — `generate_trend_dataset.py` is hardwired to the default topology (shared raw pool + filename regex assume default). Fix = a **new dedicated script**, not an edit to that one (§4.2) |

## 3. Decisions locked before running
1. **Corpus size = 10 runs/scenario, whole corpus as test.** 825×10 = **8,250 green events**, used
   entirely as the green test set — comparable statistical power to the default test split
   (~6,190 events = 41,250×15%), and since nothing trains on green there is no reason to hold out
   train/val. Configure the green build so `splits/test.txt` contains **all** events (§4.2). This is
   ~1/5 the 50-run sim cost. (Matched 50-run is possible if time allows, but not planned.)
2. **Normalization = reuse DEFAULT stats** (`dataset_fullscale_50/processed/norm_stats.json`). The
   model was trained on default-normalized voltages; green voltages must use the same stats (§4.3).
3. **Graph = green** (the point of the test). Validate it first (§4.0.5).
4. **Models = same set as the dropout study:** 5 baselines (seed 42; RGSAGE-Max 42/43/44),
   dropout-aware RGSAGE-Max (42/43/44), SDR-STGNN (42/43/44).
5. **Headline comparisons only from the multi-seed models.** Four of the five baselines are seed-42
   only, so any default-vs-green delta for them is one number minus one number. **Headline
   default-vs-green claims come only from the 3-seed models** (RGSAGE-Max, dropout-aware, SDR-STGNN);
   report **per-seed deltas plus mean±std**, and make **no significance claims** off n=3. Single-seed
   baselines are shown for context only. (We already track this class of error in `WRONG-CLAIMS.md`.)

## 4. Step-by-step plan (execute after training)

### 4.0 Pre-flight
- Confirm the GPU chain is finished (`sdr_full_chain.sh` done; no `cli train` running).
- Confirm `dataset_fullscale_50/processed/norm_stats.json` exists (default stats to reuse).

### 4.0.5 Validate the green graph FIRST (minutes, no simulation)
The measured-only graph is built from feeder **topology**, not events (`graph/measured_only.py` parses
line connectivity), so it can be built and Gate-2-validated in minutes with **zero sim**. Build the
green graph (`build_and_save_measured_only_graph` under `feeder_config="green"`) and confirm it is a
connected, acyclic 25-node tree. **If the reconfiguration disconnects a μPMU, STOP here** — you learn
this for the cost of a coffee instead of after a spot-check. Proceed only if the gate passes.

### 4.1 Sim-time spot-check (confirm the size budget)
Simulate a **small batch** of green events (e.g. one scenario × a few runs) with
`feeder_config="green"` and time it; multiply out to the 8,250-event estimate. OpenDSS sim is
**CPU-only**, single box. This confirms §3.1 fits the deadline (§8).

### 4.2 Generate the green corpus  *(the one code change to implement later — decided)*
- **Write a dedicated `scripts/generate_green_dataset.py`. Do NOT edit `generate_trend_dataset.py`** —
  twelve days from the deadline, touching its default-path regex and shared-pool logic risks breaking
  the working default pipeline while you're not looking. The new script does three green-only stages:
  1. grow a **green raw pool** via `stage_full_generation(..., feeder_config="green")` (writes
     `_green`-suffixed events),
  2. materialize those events into `dataset_green_<tag>/raw_sim/`,
  3. call `build(data_root, feeder_config="green")`, then set `splits/test.txt` to hold **all** events
     (nothing trains on green).
- Mirror `generate_trend_dataset.py`'s absolute-path handling (OpenDSS `Compile` chdir quirk).
- Output: `dataset_green_<tag>/` with `raw_sim/`, `processed/` (green graph + windows), `splits/`.

### 4.3 Fix normalization (verified implementable — one file copy)
- **Verified:** normalization is applied at **load time** (`data/dataset.py:61` normalizes voltage
  channels 0:2 from `processed/norm_stats.json`); `processed/windows` are stored **raw**. So the fix is
  simply to **overwrite** `dataset_green_<tag>/processed/norm_stats.json` with the copy from
  `dataset_fullscale_50/processed/norm_stats.json` after building — no `build()` change.
- *(Had normalization been baked into the stored windows, this swap would silently do nothing and we'd
  have needed to pass default stats into `build()`. It isn't, so we don't — but this is the item that
  fails silently if skipped.)*
- Document the choice in the paper (Reproduction Notes).

### 4.4 Evaluate trained checkpoints on green (no retraining)
- Run the degradation study with `--data-root dataset_green_<tag>` into green output dirs
  (`results/dropout_study{,_aware,_sdr}_green/`), `--seeds 42 43 44` where multi-seed checkpoints
  exist. Yields green macro-F1-vs-k curves + AUDC/retention, directly comparable to default.
- **Confound decomposition (free — do it).** The k=0 drop conflates two effects: the model relying on a
  fixed graph vs. green voltages simply being distributionally different. Since everything is
  inference-only, evaluate the green windows **twice** — once with the **green** `edge_index` (as built)
  and once with the **default** `edge_index` swapped into `dataset_green_<tag>/processed/edge_index.npy`.
  Define:
  - **A** = default windows + default graph (from the main study),
  - **B** = green windows + default graph,
  - **C** = green windows + green graph.
  Then **A−B isolates the voltage/signal shift** and **B−C isolates the topology component**. Without
  this the section answers a weaker question than §1 poses.
- **Reporting (per §3.5):** headline deltas only from the 3-seed models, per-seed + mean±std, no
  significance claims off n=3.

### 4.5 Figures + comparison
- Point `build_sdr_result_figures.py` / `build_dropout_comparison.py` at the green CSVs for green
  degradation / ablation / retention figures, plus a **default-vs-green** comparison and the A/B/C
  decomposition. Key questions: (i) how much k=0 accuracy is lost purely to the topology change;
  (ii) whether dropout-aware training's robustness advantage **survives** reconfiguration.

## 5. Risks & gotchas (ranked)
- **Normalization (§4.3)** — still the one that fails *silently*: forget the `norm_stats.json` swap and
  the green degradation is inflated with no error. Load-time application makes the fix a one-file copy,
  but it must actually be done.
- **Skipping the confound decomposition (§4.4)** — without B, "accuracy dropped N points" cannot be
  turned into a claim about structural reliance; the section under-delivers on its own premise.
- **Green graph disconnection** — caught in minutes by §4.0.5; treat a Gate-2 failure as signal, not an
  obstacle to bypass.
- **Seed asymmetry (§3.5)** — do not headline single-seed default-vs-green deltas.
- **Sim time** — settle with the §4.1 spot-check before committing.
- **Scope/consistency** — green fully in or fully out; half-in is the worst outcome (§8).

## 6. Deliverables
- `dataset_green_<tag>/` corpus (green graph + raw windows + all-in-test split), default-normalized.
- `results/dropout_study{,_aware,_sdr}_green/degradation_curves.csv` + green figures.
- A default-vs-green comparison **and** the A/B/C signal-vs-topology decomposition (per-seed + mean±std
  for the 3-seed models).
- Paper: fill `\subsection{Effect of Feeder Reconfiguration ("Green" Scenario)}` and a Reproduction-Notes
  line on the default-stats normalization choice.

## 7. Time & sequencing
- **No GPU** — CPU sim + inference. Main cost = the green sim (§4.1 confirms); 10-run green ≈ 1/5 the
  50-run sim; eval sweeps are inference-only (~tens of minutes each, ×2 for the decomposition).
- **Run after** the SDR chain + final multi-seed figures are done and reported.

## 8. Go / no-go — with a date
- **Hard gate:** green corpus **built and the first eval sweep clean by Aug 9**. If not, **scrub every
  green mention** (Results subsection heading, abstract) and ship the sensor-dropout paper clean.
  "≲1–2 days" alone has no forcing function; the date does.
- **Graph gate (§4.0.5) first** — if the reconfiguration disconnects a μPMU, no-go immediately.
- **Decide the narrative now, both ways.** If dropout-aware training's robustness advantage **survives**
  reconfiguration → a strong "robust to two independent stressors" story. If it **does not** → that is a
  perfectly reportable, honest result (a limit of the mitigation) — but only if written deliberately,
  not drafted at midnight on the 14th. Draft both framings before running.

## Pointers
- Sim: `fault_localization/simulation/{feeder.py (configure_green_feeder), runner.py (simulate_fault_event, stage_full_generation, event_filename, feeder_config)}`
- Build: `fault_localization/processing/build_dataset.py (build, feeder_config)`; graph:
  `fault_localization/graph/measured_only.py`; norm: `fault_localization/processing/normalization.py`,
  applied at load in `fault_localization/data/dataset.py:61`
- Generation orchestration (new dedicated script): `scripts/generate_green_dataset.py` (to write);
  reference `scripts/generate_trend_dataset.py` for the absolute-path/pool pattern
- Eval/figures: `scripts/run_dropout_study.py` (`--data-root`, `--seeds`),
  `scripts/build_sdr_result_figures.py`, `scripts/build_dropout_comparison.py`
- Constants: `fault_localization/config/constants.py` (GREEN_FEEDER_* switches)
- Context: `WRONG-CLAIMS.md` (green flagged as promised-but-unrun), `CLAUDE.md` (green code note)
