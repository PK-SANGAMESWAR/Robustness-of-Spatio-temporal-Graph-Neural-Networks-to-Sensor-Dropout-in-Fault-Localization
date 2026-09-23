# S0.0 -- Corpus load model

Generated 2026-09-23 by `scripts/audit_load_model.py`.
Corpus: `/home/ubuntu/Fault-Localization/dataset_fullscale_50`

**Verdict: the load is a single global SCALAR per event.**

## What was checked

- Events scanned: **41250**
- Events missing `load_multiplier`: **0**
- Stored Python types: `{'float': 41250}`
- Events whose value is a list/array (i.e. per-bus): **0**
- Value range: 0.500048 to 1.299998, mean 0.899579
- Distinct values: **41250** of 41250 events (one independent draw per event)

## Why it matters

`simulation/runner.py:116` draws `rng.uniform(0.5, 1.3)` once per event and
`monitors.py:_scale_loads_from_base` applies it uniformly to every load.  The 50 runs
of a scenario therefore lie on a **1-D curve** parametrised by L, which is what makes
the profiled load axis of `01_SPEC.md` §2 the right operator and the first draft's
per-bus load ensemble the wrong one.  It is also what makes S1.5 an exact test: the
stored multiplier is the ground truth that `argmin_L` must recover.

## Paper-1 draft on disk

- `overleaf.tex:205: Two encoding choices behind Table~\ref{tab:dataset} carry methodological weight. First, the availability mask defines what ``missing'' means throughout this paper: $m_p=1$ marks an available measurement and $m_p=0$ a missing phase or a dropped sensor, so a $\mu$PMU that fails is encoded exactly like one that was never installed. Second, the 70/15/15 partition is taken at the \emph{event} level and stratified by fault bus, so the 40 sliding windows of one event never span more than one split and no window of a test event is seen during training. What that partition does and does not hold out should be stated plainly, because it bounds every accuracy we report: the split is over the 50 independent runs of each scenario, not over the scenarios themselves, so each of the 825 $(\text{bus},\text{fault type},\text{resistance})$ configurations is represented in all three splits, and consecutive runs of one configuration differ only in the load multiplier drawn per run. Test accuracy therefore measures generalization to unseen \emph{operating conditions} of a known fault configuration, which is the regime the base setup evaluates and the one a feeder-specific deployed model actually faces; it does not measure extrapolation to a fault type or a fault resistance never seen in training. The robustness comparisons that follow are unaffected, since every model and mitigation is trained and evaluated under this identical partition. Voltage channels are z-score normalized using training-split statistics only. We adopt the base paper's dataset-generation methodology but build our own corpus (Section~\ref{sec:repro}); since every model is trained and evaluated on it with identical splits, the comparisons remain fair. Fault severity varies sharply with fault resistance: for a three-phase-to-ground fault at bus 101, the deepest-sagging $\mu$PMU channel drops $0.83$\,p.u. below its pre-fault level at $0.1\,\Omega$, $0.22$\,p.u. at $1\,\Omega$, and only $0.01$\,p.u. at $10\,\Omega$, which is why high-impedance faults are the hardest class to localize throughout this study.`
- `overleaf.tex:224: Load multiplier           & $\mathcal{U}(0.5,\,1.3)$ per run \\`

- Lines claiming a **per-bus** load: **0** -- none, so no erratum is needed for the draft on disk.

> Paper 2 must never claim a per-bus load model (`12_CHECKLISTS.md` §D).  If the
> *submitted* copy of Paper 1 differs from this draft, grep it once and record the
> result here.
