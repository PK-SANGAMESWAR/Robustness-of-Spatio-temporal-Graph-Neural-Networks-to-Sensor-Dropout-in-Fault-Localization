# Guide Review — PPT Build Guide

**Talk:** Dropout-Aware Training for Sensor-Robust Fault Localization with Spatio-Temporal GNNs
**Presenter:** P K Sangameswar · **Guide:** Dr. M Anbazhagan
**Build:** 12 slides + 3 hidden backup · 4 figures · ~16 minutes

---

## CONTENTS

- **Part 1 — Setup** (do once, 10 minutes)
- **Part 2 — The 12 slides** (exact content for each)
- **Part 3 — Backup slides**
- **Part 4 — Q&A preparation**
- **Part 5 — Final checklist**

**How each slide is specified:**

| Block | What it means |
|---|---|
| **TITLE** | Type this as the slide heading, exactly |
| **BULLETS** | Type only these onto the slide. Nothing else |
| **VISUAL** | Figure or table to insert, with position |
| **SAY** | Your spoken words — paste into the Notes pane, never onto the slide |
| **CLICKS** | Animation order |
| **TIME** | Target duration |

> **The one rule that matters:** the slide carries the *claim*, your voice carries the
> *explanation*. Nothing on a slide should be longer than about 10 words.

---
---

# PART 1 — SETUP

### Step 1. Page setup
`Design → Slide Size → Widescreen (16:9)`

### Step 2. Fonts (set in the Slide Master, not per slide)
`View → Slide Master`

| Element | Font | Size |
|---|---|---|
| Slide title | Calibri **Bold** | 32 pt |
| Body bullets | Calibri | 20 pt |
| Table text | Calibri | 16 pt |
| Figure caption | Calibri *Italic* | 14 pt |
| Footer | Calibri | 12 pt, grey |

**Never go below 14 pt.** If text does not fit, cut words — do not shrink the font.

### Step 3. Colours
Set these as theme colours so they are one click away:

| Role | Hex | Used for |
|---|---|---|
| Background | `#FFFFFF` | every slide |
| Body text | `#1A1A1A` | all text |
| Accent | `#17789B` teal | titles, key boxes, positive results |
| Warning | `#E8A33D` amber | failure numbers (the 0.048 cell) |
| Positive | `#1B9E63` green | significant-result rows |

These match the colours already inside the figures, so inserted images will not clash.

### Step 4. Footer
In the Slide Master, add to every slide except slide 1:
`Sensor-Robust Fault Localization · ICMLDE 2026` (left) and the slide number (right).

### Step 5. Table style
Use a plain style: **no vertical lines**, a single horizontal rule under the header row, header
row in accent teal with white text. Bold every number. Leave units plain.

### Step 6. How to insert figures correctly
`Insert → Pictures → This Device` → select the file → **drag a corner handle only**
(never a side handle — that distorts the image). All figures are 400 dpi and will stay sharp.

### Step 7. How to mask the heatmap row (needed on slide 6)
`Insert → Shapes → Rectangle` → draw it over the bottom row of the image →
`Shape Fill: White` · `Shape Outline: No Outline`.
This hides the SDR-STGNN row, which is the result you reveal later.

---
---

# PART 2 — THE 12 SLIDES

## SLIDE 1 — Title

**TITLE**
```
Dropout-Aware Training for Sensor-Robust Fault Localization
with Spatio-Temporal Graph Neural Networks
```

**BULLETS** (centred, no bullet marks)
```
P K Sangameswar
Guide: Dr. M Anbazhagan

Amrita Vishwa Vidyapeetham, Coimbatore
ICMLDE 2026
```

**VISUAL** — none.

**SAY:** "Good morning. My work is on locating faults in power distribution grids using
spatio-temporal graph neural networks, and specifically on making that work when the sensors start
failing. The one-line summary is: we make GNN fault localization survive losing its sensors."

**CLICKS** — none. **TIME** — 0:20

---

## SLIDE 2 — The Problem

**TITLE**
```
Locating Faults With Almost No Sensors
```

**BULLETS**
```
•  Find which bus faulted — 26 classes (25 buses + no-fault)
•  Only 25 μPMUs on a 123-bus feeder
•  Just 6 of 25 fault buses are measured — 19 are invisible
•  Real μPMUs fail: comms, GPS sync, power, maintenance
•  Published STGNN work assumes sensors never fail
```

**VISUAL** — a highlighted box at the bottom, accent teal, white text:
```
THE GAP:  No one has measured the damage.  No one has fixed it.
```

**SAY:** "Fault localization means identifying *where* on the feeder the fault happened, not just
that one happened — a 26-class problem, 25 candidate buses plus a no-fault class. The difficulty
is partial observability. We have 25 μPMUs on a 123-bus feeder, and only six of them sit on buses
that can actually fault. So 19 fault locations are never directly observed; the model must infer
them through topology. And every published STGNN result assumes those sensors are always up. Real
ones are not — they lose comms, they lose GPS sync, they get taken down for maintenance. Nobody
has measured what that does to these models, and nobody has proposed a fix."

**CLICKS** — bullets together, then the GAP box. **TIME** — 1:00

---

## SLIDE 3 — Dataset and Graph

**TITLE**
```
IEEE 123-Bus Feeder, 1.65 Million Windows
```

**VISUAL** — two-column layout.
**Left 55 %:** `paper/figures/measured_only_graph.png`, vertically centred, **do not crop**
(the legend is part of the image).
Caption below, 14 pt italic: *"25 μPMU nodes · red ring = also a fault location (6 of 25)"*

**BULLETS** (right 45 %)
```
•  OpenDSS simulation, 1 ms sampling
•  825 scenarios = 25 buses × 11 types × 3 resistances
•  41,250 events → 1,650,000 windows
•  Exactly 50 % no-fault
•  Sample = 25 nodes × 20 steps × 6 features
•  Graph = 25 nodes, 24 edges
•  Event-level 70/15/15 split — zero leakage
```

**SAY:** "Here is the data. We simulate the IEEE 123-bus feeder in OpenDSS at one-millisecond
resolution. The design is fully factorial — 25 fault buses, 11 fault types, 3 fault resistances,
so 825 scenarios; 50 runs each with randomised load gives 41,000 events and 1.65 million windows,
balanced exactly fifty-fifty.

Now look at the figure, because this is the graph the model actually sees. Not 123 buses — just
the 25 measured ones, connected as a substation-rooted spanning tree. The colours are phase
instrumentation. The important thing is the red rings: there are exactly six. Those are the only
fault buses we observe directly. Every other fault location has to be inferred from what its
neighbours see.

One methodological point: our split is at the *event* level, not the window level, and we verified
zero leakage. All 40 windows of an event stay together. Splitting by window would leak and inflate
every number in this talk."

**CLICKS** — figure, then bullets. **Point at the red rings when you mention them.**
**TIME** — 1:30

---

## SLIDE 4 — Model

**TITLE**
```
One Template, Five Architectures
```

**VISUAL** — pipeline block, centred, **Consolas 18 pt** so the arrows align:
```
[25 nodes × 20 steps × 6 features]
        ↓   shared GRU-128 per node          (temporal)
        ↓   4 × GNN layers  ← only variable  (spatial)
        ↓   BatchNorm → ReLU → Linear(26)
        ↓   soft vote over 25 nodes → prediction
```

**BULLETS** (below the block)
```
•  Baselines: GRU (no graph) · RGCN · RGSAGE-Mean · RGSAGE-Max · RGATv2
•  30 epochs · seeds 42 / 43 / 44 · single T1000 8 GB
•  Declared deviations: depth 4 (paper used 1) · 1.65 M windows (paper reports 2.5 M)
```

**SAY:** "All five models share one template. A GRU encodes each node's 20-step voltage window in
time. Four graph layers then mix information across the feeder. A per-node classifier follows, and
finally soft voting — we sum the softmax outputs of all 25 nodes and take the argmax as the
event's prediction.

The only thing that differs between the five baselines is the spatial block. That is deliberate:
it lets us attribute any difference to the aggregation operator and nothing else. Everything runs
at three seeds, so every number I show has error bars.

Two deviations I want to declare up front. We use four graph layers where the base paper used one,
because depth one was unstable for us. And our corpus is 1.65 million windows against their
reported 2.5 million, which is not reconstructable from the 825-scenario grid. So this is not a
numeric reproduction — we adopt their methodology and report our own numbers. Both deviations are
tabulated in the paper."

**CLICKS** — none. **TIME** — 1:15

---

## SLIDE 5 — Baseline Accuracy

**TITLE**
```
With All Sensors Available
```

**VISUAL** — full-width table. **Bold the RGSAGE-Max row.**

| Model | macro-F1 | 0.1 Ω | 1 Ω | 10 Ω |
|---|---|---|---|---|
| GRU (no graph) | 0.286 ± 0.009 | 0.410 | 0.292 | 0.105 |
| RGATv2 | 0.522 ± 0.087 | 0.630 | 0.574 | 0.312 |
| RGCN | 0.712 ± 0.017 | 0.874 | 0.792 | 0.428 |
| RGSAGE-Mean | 0.786 ± 0.023 | 0.906 | 0.843 | 0.587 |
| **RGSAGE-Max** | **0.846 ± 0.007** | **0.949** | **0.893** | **0.676** |

**BULLETS** (below table)
```
•  Topology does real work: +0.56 F1 from GRU to RGSAGE-Max
•  Max aggregation wins — a fault is a localized extremum
•  High-resistance faults are the hard case: 0.949 → 0.676
```

**SAY:** "This is the clean benchmark — all sensors available, mean and standard deviation over
three seeds.

Two takeaways. First, the graph matters enormously: the GRU baseline with no spatial layer reaches
0.29, and adding graph layers takes us to 0.85. That is half a point of F1 from topology alone.
Second, max aggregation is the clear winner, and there is a physical reason — a fault is a
localised extremum in the voltage field, and mean-pooling averages it away while max-pooling
preserves it.

The last three columns split by fault resistance and show where the difficulty lies. A bolted
0.1-ohm fault is easy at 0.95; a 10-ohm fault drops us to 0.68, because a high-impedance fault
barely moves the voltage at all.

One honest caveat: our best validation epoch was around 28 of 30, so these models were still
improving when the budget ran out. Treat these numbers as a floor."

**CLICKS** — table, then bullets. **TIME** — 1:15

---

## SLIDE 6 — The Collapse ⭐ *most important slide*

**TITLE**
```
Removing Sensors Destroys Every Model
```

**VISUAL** — two-column layout.
**Left 50 %:** `paper/figures/sdr_heatmap.png`
> ⚠️ **Mask the bottom row before showing this** (Setup Step 7). The bottom row is SDR-STGNN —
> your own fix — and revealing it here spoils slides 8 and 9.

Caption: *"Macro-F1 as μPMUs are removed — green good, pale is failure"*

**Right 50 %:** table. Colour the RGSAGE-Max `0.048` cell **amber**.

| Model | k=0 | k=2 | k=5 | k=10 | Retained |
|---|---|---|---|---|---|
| **RGSAGE-Max** | 0.846 | 0.590 | 0.236 | **0.048** | **5.7 %** |
| RGSAGE-Mean | 0.786 | 0.647 | 0.393 | 0.108 | 13.8 % |
| RGCN | 0.712 | 0.610 | 0.418 | **0.164** | **23.1 %** |
| RGATv2 | 0.522 | 0.444 | 0.302 | 0.157 | 30.1 % |
| GRU | 0.286 | 0.252 | 0.111 | 0.051 | 17.9 % |

**BULLETS** (below the table, right column)
```
•  Zero k μPMUs at inference only — training untouched
•  k = 10 of 25 means 40 % of the fleet offline
•  The most accurate model is the most fragile
```

**SAY:** "Now the core experiment. At inference time only — training is untouched — we zero out k
randomly chosen μPMUs and re-evaluate. k goes up to 10 of 25, so 40 % of the fleet offline. Three
mask realisations and three training seeds, so nine runs per point.

Everything collapses. Look at the top row: our best model falls from 0.85 to 0.048. It retains
under six percent of its accuracy. It is effectively destroyed.

But here is the finding that surprised us. Look at RGCN. It is *less* accurate with all sensors up
— 0.71 against 0.85 — yet under sensor loss it retains three to four times more. The most accurate
architecture is the most fragile. The reason is the same max-pooling that won the previous slide:
it concentrates the decision on a few extremal nodes, so silencing those nodes is catastrophic.
Mean-pooling and RGCN spread the decision out, which costs accuracy but buys robustness.

I will be careful here — this inversion has the same sign in all three seeds, but at three seeds it
does not reach significance, p is 0.076. So I present it as a strong observed trend, not a law."

**CLICKS** — figure → table → bullets → amber highlight on `0.048`.
**TIME** — 1:45 · **do not rush this slide**

---

## SLIDE 7 — The Fix

**TITLE**
```
Dropout-Aware Training
```

**VISUAL** — pseudocode box, top half, Consolas 18 pt:
```
for each training batch:
    k ~ U{0 .. k_max}
    S ~ random subset of k nodes
    zero all 6 channels of nodes in S
    train as usual
```

**BULLETS** (below the box)
```
•  A data augmentation — not an architecture change
•  Mask channels are static, and only 14 of 25 nodes are 3-phase
•  Dropping a node forces its mask to (0,0,0) — never seen in training
•  Dropout-aware training puts that pattern into the distribution
•  Cost: identical parameters, identical latency, +2.7 % training time
```

**SAY:** "Our fix is deliberately simple: apply the same random sensor masking *during training*.
Sample a k for each batch, zero those nodes, train as normal. It is a data augmentation — no
architecture change, no loss change, no new hyper-parameters.

Why does something this simple work? This is the part worth a moment. Each node has six features:
three voltages and three availability-mask channels. Those mask channels are static — they encode
which phases each μPMU instruments — and only 14 of our 25 sensors are three-phase; the rest show
patterns like one-zero-zero. Here is the key. When we drop a node we zero all six channels, so its
mask becomes zero-zero-zero, and that is a pattern the node has never once shown during training.
The network is off-distribution, and it fails in the uncontrolled way you saw. Dropout-aware
training simply puts that pattern *into* the training distribution, and the model learns to
recognise a dead sensor and route around it.

The cost is essentially nothing — identical parameter count, identical inference latency, both
measured, and 2.7 % more training time."

**CLICKS** — pseudocode → bullets. **TIME** — 1:30

---

## SLIDE 8 — It Works ⭐ *the payoff*

**TITLE**
```
Retention at 40 % Sensor Loss: 5.7 % → 80.3 %
```

**VISUAL** — `paper/figures/sdr_degradation_all.png`, **full slide width, top 55 %**.
Caption: *"Baselines dive; both mitigations stay flat"*

Table below:

| RGSAGE-Max variant | k=0 | k=10 | Retained | AUDC |
|---|---|---|---|---|
| Standard | 0.846 | 0.048 | 5.7 % | 0.402 |
| + neighbour-mean imputation | 0.846 | 0.099 | 11.8 % | 0.476 |
| **+ dropout-aware training** | 0.839 | **0.674** | **80.3 %** | **0.913** |

**BULLETS**
```
•  Clean accuracy cost: −0.008 F1, not significant
•  Imputation barely helps — the model never learned to distrust it
•  Works on all five architectures (p ≤ 0.027)
```

**SAY:** "Here is the result. The diving curves are the five standard baselines. The two flat lines
along the top are the mitigations. Retention at 40 % sensor loss goes from 5.7 % to 80.3 %, and the
clean-accuracy cost is 0.008 F1, which is not statistically significant. The robustness is
essentially free.

The middle row answers a question every reviewer asks: why not just *impute* the missing readings?
We tried it — neighbour-mean imputation, and we were generous, marking imputed nodes as fully
observed. It gains a little, 5.7 to 11.8 percent, but that gain is not significant either. The
lesson matters: reconstructing the missing *signal* is not the bottleneck, because the
reconstructed values still reach a model that was never trained to distrust them. Robustness has to
come from training, not from patching the input at test time.

And this is what makes it a general result rather than a trick: it works on all five architectures,
every one significant after correction. I will note honestly that RGCN and RGSAGE-Mean pay about
0.1 of clean F1 for it; RGSAGE-Max does not."

**CLICKS** — figure → table → bullets. **TIME** — 1:40

---

## SLIDE 9 — The Ablation

**TITLE**
```
Our Gated Architecture Matches — But Does Not Win
```

**VISUAL** — `paper/figures/sdr_mechanism.png`, **full width, top half** (it is wide and short).
Caption: *"Left: dead sensor still sends a message. Right: the gate blocks it."*

Table below:

| | AUDC | k=10 | Retained | Latency |
|---|---|---|---|---|
| **Dropout-aware (ungated)** | **0.913** | **0.674** | **80.3 %** | 0.639 ms |
| SDR-STGNN (gated, ours) | 0.892 | 0.632 | 75.2 % | 0.755 ms **+18 %** |

**BULLETS**
```
•  Difference p = 0.098 — a statistical tie, at +18 % latency
•  Gating a dead node also blocks routing THROUGH it
•  A healthy 2-hop neighbour can no longer relay
```

Bottom box, accent teal border:
```
We report this faithfully. We do not claim the gate wins.
```

**SAY:** "We also asked the obvious architectural question: instead of fixing this in training,
could we fix it in the model? So we built SDR-STGNN, an availability-gated version of RGSAGE-Max
where a dropped node emits no messages at all, gated at every hop. At zero dropout it is identical
to RGSAGE-Max.

The honest answer is that it matches but does not beat the training-side fix. AUDC 0.892 against
0.913 — a p-value of 0.098, so statistically a tie — and it costs 18 % more inference latency for
that tie.

We think we know why. Silencing a dead node at every hop also blocks multi-hop routing *through*
that node. Information a healthy two-hop neighbour could have relayed is destroyed, while the
ungated model can still reconstruct along longer paths.

I present this as a negative result deliberately. The architectural idea was ours, it is intuitive,
and it did not win. The honest conclusion is more useful than a claimed one: on this task the
aggregation operator is a weak lever, and robustness is better bought during training."

**CLICKS** — figure → table → bullets → the box last. **TIME** — 1:30

---

## SLIDE 10 — Stress Tests

**TITLE**
```
Worst-Case and Realistic Sensor Failure
```

**VISUAL** — table, top half:

| RGSAGE-Max | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|
| Standard — targeted | **0.448** | 0.361 | 0.206 | 0.048 |
| Standard — random | 0.736 | 0.590 | 0.236 | 0.048 |
| **Dropout-aware — targeted** | 0.755 | 0.695 | 0.600 | **0.521** |

**BULLETS** — two labelled groups:
```
TARGETED — drop the sensors nearest the fault
•  At k=1 the standard model falls 3.6× further than random
•  Our fix is penalized too: AUDC 0.913 → 0.747
•  But from k=2 up, its worst case beats standard's average case

PERSISTENT — a failed sensor stays dark all event
•  Nothing moves: no point shifts more than 0.003 F1
•  Because our decision rule is per-window, not per-event
```

**SAY:** "Two stress tests, because uniform random dropout is the *average* case.

First, worst case. Instead of dropping sensors at random we drop the ones topologically nearest
the fault — the most informative ones. This is far more damaging where the model is still usable:
at k equals one the standard model falls to 0.448 instead of 0.736, a drop three and a half times
larger. And I will be straight: our mitigation is penalised too, its AUDC falling from 0.913 to
0.747. Worst-case loss is a distinct stressor, not one that dropout-aware training absorbs. What we
*can* claim is that from k equals two upward, the dropout-aware model's worst case still beats the
standard model's average case. At k equals one they are indistinguishable.

Second, realism. In the main sweep the mask is redrawn every window, but a real failed sensor stays
dark. So we repeated it holding the dropped set across all 40 windows of an event — and nothing
moved, under 0.003 F1. The reason is that our decision rule is per-window: soft voting pools the 25
nodes of one window, never the 40 windows of an event. I flag in the paper that this equivalence
would not hold for an event-level rule."

**CLICKS** — table → targeted group → persistent group. **TIME** — 1:30

---

## SLIDE 11 — What Is and Isn't Significant

**TITLE**
```
Honest Statistics at Three Seeds
```

**VISUAL** — full-width table. Colour ✅ rows **green**, ⚠️ rows **amber**.

| Claim | Δ | p | Verdict |
|---|---|---|---|
| Dropout-aware > standard @ k=10 | +0.626 | 0.003 | ✅ significant |
| Retention @ k=10 | +74.6 pp | 0.003 | ✅ significant |
| AUDC | +0.511 | 0.003 | ✅ significant |
| Dropout-aware > imputation @ k=10 | +0.574 | 0.008 | ✅ significant |
| Clean-F1 cost | −0.008 | 0.196 | ✅ null, as claimed |
| Dropout-aware over SDR-STGNN | +0.021 | 0.196 | ⟺ tie |
| Accuracy/robustness inversion | +0.116 | 0.076 | ⚠️ trend only |
| Imputation over zeroing @ k=10 | +0.051 | 0.102 | ⚠️ not significant |

**BULLETS**
```
•  Paired t-tests over seeds 42/43/44, Holm-corrected, 95 % CIs
•  Plus a cluster bootstrap over 6,175 test events
•  Measured cost: dropout-aware free at inference, +2.7 % training
```

**SAY:** "I want to be explicit about what is and is not significant, because with three seeds you
have to be. The green rows are our mitigation claims — all significant after Holm correction. The
fifth row is a null we *want*: the clean-accuracy cost is indistinguishable from zero, which is
exactly what we claim.

The bottom three are the ones I will not overstate. Dropout-aware versus our own gated architecture
is a tie. The accuracy-robustness inversion I showed on slide six is a trend at p equals 0.076, not
a significant result. And imputation's advantage over plain zeroing is not significant either. I
would rather show you this table than have you find it in the paper."

**CLICKS** — table → bullets. **TIME** — 1:10

---

## SLIDE 12 — Conclusion

**TITLE**
```
Conclusion
```

**BULLETS** — three labelled groups:
```
FINDINGS
•  Benchmarked 5 STGNN architectures — RGSAGE-Max reaches 0.846
•  Sensor loss collapses them: 0.846 → 0.048
•  The most accurate architecture is the most fragile
•  Dropout-aware training: 5.7 % → 80.3 % retention, free at inference
•  Our gated architecture matches but does not beat it

LIMITATIONS
•  One feeder · simulation only · three seeds
•  Stress tests on a single backbone · accuracies are a floor

NEXT
•  Spatially correlated failure — a comms hub taking out a region
•  100-run corpus · real μPMU field data
```

**SAY:** "To conclude. We benchmarked five spatio-temporal GNN architectures on the IEEE-123 feeder
under partial observability and report our own three-seed numbers, topping out at 0.846.

We showed these models collapse under sensor dropout, and that clean accuracy is an unreliable
guide to robustness, because the best model was the most fragile.

Our contribution, dropout-aware training, takes retention at 40 % sensor loss from under six
percent to over eighty, at zero inference cost, and works on every architecture we tried.

And we report a negative result honestly: the gated architecture we designed matches but does not
beat that simple training-time fix.

The limitations are one feeder, in simulation, with three seeds, and stress tests on a single
backbone. The immediate next step is spatially correlated failure — a communications hub taking out
a whole region, which is the one dropout geometry we have not tested. Thank you."

**CLICKS** — one group per click. **TIME** — 1:15

---
---

# PART 3 — BACKUP SLIDES

Build these after slide 12, then `right-click → Hide Slide`. Jump to them only if asked.

| # | Title | Figure | Say if asked |
|---|---|---|---|
| B1 | Why is a 10 Ω fault hard? | `fault_signature.png` | "A bolted fault sags 0.83 p.u.; a 10-ohm fault sags 0.01. There is almost no signal." |
| B2 | How the dataset was built | `stgnn_dataset_size_breakdown.png` | "825 scenarios, 50 runs each, 40 windows per event — 1.65 million windows." |
| B3 | SDR-STGNN architecture | `sdr_architecture.png` | "The availability vector gates message passing at every hop." |

---
---

# PART 4 — Q&A PREPARATION

Rehearse these. Do not build slides for them.

**Is this a reproduction of the base paper?**
No, and we say so explicitly. We adopt the methodology and report our own numbers. Their 2.5 M
windows is not reconstructable from the 825-scenario grid; we use 1.65 M and depth 4 instead of 1.
All deviations are tabulated. We make no numeric-replication claim.

**Isn't dropout-aware training just standard dropout?**
No. Standard dropout zeroes random *activations inside* the network. This zeroes *entire sensor
nodes' input channels*, including the mask channels — matching the real failure mode. That is what
makes the (0,0,0) pattern learnable.

**Why not impute the missing readings?**
We did, generously — neighbour-mean, with imputed nodes marked as observed. AUDC only moves 0.402
to 0.476, and the gain is not significant. The values still reach a model never trained to
distrust them.

**What if the failed sensors are the ones near the fault?**
Measured — slide 10. Much worse at low k. Our fix is penalised too, but from k=2 up its worst case
beats the standard model's average case.

**Real sensors stay dead — doesn't per-window remasking flatter you?**
Also measured. Holding the mask for a whole event changes nothing, because the decision rule is
per-window. We flag that it would not hold for an event-level rule.

**What about a whole substation region going down?**
That is spatially correlated dropout — the one geometry we have *not* tested. It is our top
future-work item.

**Why only three seeds?**
Between 4.7 and 12.3 GPU-hours per run on a single T1000, times five architectures times three
variants. We mark exactly which claims survive n=3 and which do not.

**Is 40 % sensor loss realistic?**
It is a stress test, not an operating point. We report the whole curve so a utility can read off
its own risk level — and even at k=1 or 2 the standard model already loses 0.11 to 0.26 F1.

---
---

# PART 5 — FINAL CHECKLIST

### Figures used

| Slide | File | Note |
|---|---|---|
| 3 | `paper/figures/measured_only_graph.png` | Do not crop — legend must stay |
| 6 | `paper/figures/sdr_heatmap.png` | ⚠️ **Mask the bottom row** |
| 8 | `paper/figures/sdr_degradation_all.png` | Full width |
| 9 | `paper/figures/sdr_mechanism.png` | Wide and short — fits the top half |
| B1–B3 | `fault_signature` · `stgnn_dataset_size_breakdown` · `sdr_architecture` | Backup only |

**Not used:** `sdr_retention.png`, `sdr_audc_bar.png` (same story as slide 8 in other units);
`sdr_ablation_backbone.png` — **its title is clipped at both edges; regenerate with
`bbox_inches="tight"` before using it anywhere.**
**Never use** any `*.prebugfix.png` file. Ignore `Screenshot 2026-08-03 112448.png`.

### Before you present

- [ ] 16:9 set; fonts and colours defined in the Slide Master
- [ ] Slides 1–12 built using only the BULLETS text
- [ ] All SAY text pasted into the Notes pane
- [ ] 4 figures inserted, aspect ratios undistorted
- [ ] **Slide 6 bottom row masked**
- [ ] `0.048` cell amber; significant rows green
- [ ] 3 backup slides built and hidden
- [ ] Rehearsed once with a timer — target 16 min
- [ ] Exported to PDF as a projector fallback

### Timing

| Slides | Time |
|---|---|
| 1–4 (setup of the problem) | 4:05 |
| 5–6 (the finding) | 3:00 |
| 7–8 (the contribution) | 3:10 |
| 9–11 (honesty and rigour) | 4:10 |
| 12 (close) | 1:15 |
| **Total** | **≈ 15:40** |

**If cut to 12 minutes:** drop slide 10, compress slide 4. **Never cut 6, 8, or 9** — they are the
argument.

### Six things not to do

1. **Don't** put the SAY script on the slides.
2. **Don't** show slide 6's heatmap unmasked — it spoils slides 8 and 9.
3. **Don't** call SDR-STGNN a win. It is a tie (p = 0.098) — saying so is what makes the rest of
   the talk credible.
4. **Don't** call the accuracy/robustness inversion significant. p = 0.076 — it is a trend.
5. **Don't** say "reproduce" or "reproduction". We adopt the methodology and report our own numbers.
6. **Don't** promise the green-feeder experiment. It was cut from scope.
