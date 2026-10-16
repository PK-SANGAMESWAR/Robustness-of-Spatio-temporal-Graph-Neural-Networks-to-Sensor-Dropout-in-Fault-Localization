# REVIEW 1 — Speaking Script for `REVIEW1.pdf` (22 slides)

Built against the **actual deck you exported**, not the older `PPT-GUIDE-REVIEW.md` plan.
Every number quoted here was re-verified against `results/` on 2026-08-17.

---

## PART 0 — READ THIS FIRST

### 0.1 The one-paragraph version of your talk

> "Fault localization on distribution feeders is done with spatio-temporal GNNs, and every
> published result assumes all sensors are online. I measured what happens when they aren't:
> drop 10 of 25 μPMUs and the best model keeps **5.7%** of its accuracy — it doesn't degrade,
> it dies. Then I fixed it: randomly masking sensor nodes *during training* restores retention
> to **80.3%**, at identical parameter count and identical inference latency. It works on all
> five architectures. I also built a gated architecture to do the same thing structurally — it
> only matches the training fix and costs 18% more latency, so I report it as a negative result."

If you can say that from memory, you can survive the whole review. Everything else is detail.

### 0.2 What the examiners are actually scoring (from `REVIEW.md`)

| Rubric item | Guide (8) | Panel (12) | Covered by slide |
|---|---|---|---|
| Problem statement & research gap | ✅ | ✅ | 2, 3, 4, 5 |
| Proposed methodology | ✅ | ✅ | 6, 7, 8, 9, 10, 12, 14 |
| Implementation | ✅ | ✅ | **weakly — see 0.3** |
| Regular meetings with guide | ✅ | — | **❌ MISSING** |
| SDG mapping + justification | — | ✅ | **❌ MISSING** |
| Presentation | — | ✅ | delivery, see Part 5 |
| Sem 3 project plan | — | ✅ | **❌ MISSING** |

### 0.3 ⚠️ THREE GAPS TO CLOSE BEFORE YOU PRESENT

This deck is a **conference talk**, not a review deck. It's scientifically stronger than it needs
to be and administratively incomplete. Three slides are missing that carry marks:

**Gap 1 — "Regular meetings with guide" (guide review, marks).**
Add one slide after the Conclusion. A five-row table: Date | What we discussed | What changed
because of it. Only you have this data — fill it from memory/email. Include at least one row where
the guide's input **changed the work** (e.g. "scope of the dropout contribution", "cut the
green-feeder experiment", "add an imputation baseline"). Examiners notice when the outcome column
is empty.

**Gap 2 — "SDG mapping + justification" (panel, marks).**
Add one slide. Use this content:

| SDG | Link | Justification |
|---|---|---|
| **7 — Affordable & Clean Energy** | Direct | Faster localization → shorter outages → less energy not supplied; robust monitoring supports DER-heavy unbalanced feeders |
| **9 — Industry, Innovation & Infrastructure** | Direct | Resilient infrastructure: the method keeps working when 40% of monitoring hardware is offline, instead of demanding perfect instrumentation |
| **11 — Sustainable Cities & Communities** | Direct | Reliable distribution supply is basic urban infrastructure; faster restoration reduces household/service disruption |
| 13 — Climate Action | Indirect | Makes DER-heavy feeders observable, enabling higher renewable penetration |

Your strongest sentence, say it out loud: *"The contribution reduces the sensor coverage needed
for reliable localization — it lowers the cost barrier for utilities that cannot afford dense
μPMU deployment."*

**Gap 3 — "Sem 3 project plan" (panel, marks).**
Add one slide, six phases, with **month ranges in the first column** (examiners specifically look
for dates):

| Phase | Work | Output |
|---|---|---|
| 1 | Conference paper submission & revision | ICMLDE / Procedia paper |
| 2 | Spatially correlated failures — a comms hub taking out a whole region | New degradation study |
| 3 | Extend worst-case + persistent stress tests to all five architectures | Complete robustness matrix |
| 4 | Full-scale run on the 3.30 M-window corpus | Scale validation |
| 5 | Cross-topology generalization — reconfigured feeder | Transfer results |
| 6 | Thesis writing + journal extension | Thesis + journal draft |

Say: *"Phase 2 is the immediate next step — spatially correlated dropout is the one failure
geometry I have not tested. The code, dataset and evaluation harness already exist, so every phase
reuses the same pipeline."*

**Also worth adding: an "Implementation status" slide.** Right now implementation is implied by
your results but never *shown*. Review 1's stated goal is "show that implementation has started."
Yours is finished — say so explicitly, on a slide, with the repo structure. One slide:
`fault_localization/` (simulation, processing, graph, models, dropout, experiments) ·
`scripts/` (30+ runnable sweeps and figure builders) · `config/` (YAML per model) · `tests/` ·
Git-versioned, every run writes metrics + checkpoint + config. Then the punch line:
*"Review 1's target was 'implementation has started'. Implementation is complete."*

### 0.4 Small fixes on existing slides

- Slide 12 title: **"Eval Stategy" → "Evaluation Strategy"** (typo, on screen, will be seen).
- Slide 13 title: **"Removing Sensors Destroy every model" → "Removing Sensors Destroys Every Model"**.
- Slide 6 (Dataset) and slide 7 (Graph Construction): bullet text runs **under/behind the figure**.
  Shrink the text box or cut bullets — a panel member reading a half-hidden line will stop
  listening to you.
- Slide 2 and slide 4: too dense. You do not need to fix them if time is short, but shrink the
  font by one step so nothing touches the slide edge.
- Slide 15 (the degradation curve) has **no title**. Add one: **"Sensor Dropout — Every Baseline
  Collapses, the Mitigation Doesn't"**. A titleless slide looks unfinished.
- Number consistency: slide 13 says RGCN retains **23.1%** and RGATv2 **30.1%**; slides 17 and 20
  say **23.2%** and **31.0%**. Both are correct — 23.1 is (mean F1 at k=10)/(mean F1 at k=0),
  23.2 is the mean of the three per-seed retention ratios. If anyone catches it, that is your
  one-sentence answer. If you have five minutes, make them all match (use 23.1 / 30.1 everywhere,
  the ratio-of-means convention).

---

## PART 1 — TIMING

**If you get 15 minutes** (target 12 min talk + 3 min Q&A):

| Block | Slides | Minutes |
|---|---|---|
| Setup — problem, gap, objectives | 1–4 | 2:30 |
| Related work | 5 | 0:45 |
| Method — data, graph, model, workflow | 6–9 | 3:00 |
| Eval protocol | 10, 12 | 1:15 |
| Result 1 — clean baselines | 11 | 1:00 |
| Result 2 — the collapse | 13, 15 | 1:30 |
| Result 3 — the fix | 14, 16, 17 | 2:00 |
| Honest negative + stress tests | 18, 19 | 1:15 |
| Conclusion + admin slides | 20 + new | 1:15 |

**If you get 8 minutes** — cut slides 5, 9, 10, 12, 17, 19 and merge 6+7 verbally. Never cut
13, 15, 16 — that trio *is* the contribution.

**The rhythm that matters:** slides 11 → 13 → 16 is your story arc. Good (0.846) → dead (0.048)
→ saved (0.674). Slow down on those three. Speed up everywhere else.

---

## PART 2 — SLIDE BY SLIDE

Format: **SAY** = what to speak. **POINT** = what to gesture at. **WATCH** = trap to avoid.

---

### SLIDE 1 — Title

**SAY (20s):** "Good morning. My work is on fault localization in electrical distribution grids —
identifying *which bus* has faulted — using spatio-temporal graph neural networks. The specific
contribution is making that work when the sensors themselves fail. I'm P K Sangameswar, guided by
Dr. M Anbazhagan."

**WATCH:** Don't read the title aloud word for word. Everyone can read.

---

### SLIDE 2 — Problem Statement

This slide has four bullets and each one is a separate idea. Do not read them; *narrate* them.

**SAY (1:15):**
"Start with the physical situation. A distribution feeder is a branching low-voltage network —
the IEEE 123-bus test feeder is the standard benchmark. When a fault occurs, a line touches
ground or two phases short, the protection trips and customers lose power. The utility then has
to find *where*. Crews cannot patrol the whole feeder, so restoration time — and therefore
outage cost — is dominated by the search.

Second: this is a **localization** problem, not a detection problem. Detection is easy — the
voltage collapses, everyone knows. The hard question is *which of the 25 candidate buses*. With
a no-fault class that's a 26-class classification problem.

Third, and this is the structural difficulty: μPMUs — micro phasor measurement units — cost
tens of thousands of dollars each, so utilities never instrument every bus. In our setup there
are 25 μPMUs on a 123-bus feeder, and critically, **only 6 of the 25 candidate fault buses
actually carry a sensor**. Nineteen fault locations are never directly measured. The model has to
infer them from what the neighbouring sensors see, through the topology. That is precisely why
this is a *graph* problem.

Fourth: real deployments are worse than that. μPMUs lose their communications link, lose GPS
sync, lose auxiliary power, or go down for maintenance. So the sensor set you have at inference
time is not the sensor set you trained on. And every published STGNN result assumes full
availability."

**POINT:** the "6 of 25 → 19 invisible" bullet. That is the sentence that makes the graph
necessary.

**WATCH:** don't say "the model must infer them from neighbours" without adding *"through the
topology"* — that's the phrase that motivates the whole architecture.

---

### SLIDE 3 — Research Gap

One sentence on the slide. This is deliberate. Let it sit.

**SAY (30s):** "So the gap is narrow and specific. There is no published evaluation of what
happens to STGNN fault localization when sensors go missing at inference time, and consequently
no mitigation for it. Not 'nobody has done fault localization' — plenty have. Not 'nobody has
done GNNs on grids' — that's an active field. The gap is that **robustness to sensor dropout has
never been measured for this task**, so nobody knows whether these models are deployable."

**WATCH:** This is the slide where a panel member decides whether your gap is real or manufactured.
Emphasise *measured*. You are claiming a missing measurement, which is a much more defensible
claim than "nobody has thought of this."

**IF ASKED "are you sure nobody has done this?"** — "Chen et al. 2020 use data-loss augmentation,
but on a static single-snapshot spectral GCN with no temporal axis, and they don't report a
degradation curve. Zhang et al. 2025 handle missing nodes by imputation, but they assume the
unmeasured set is fixed and known in advance — which is exactly the assumption that breaks when
a sensor fails at runtime. Karabulut et al. 2026, the STGNN work I build on, assumes full
availability. So the pieces exist separately; the measurement for this task does not."

---

### SLIDE 4 — Objectives

**SAY (45s):** "Five objectives, in dependency order.

One — adopt the STGNN methodology from Karabulut et al. and produce **our own** benchmark, five
architectures at three seeds each. I say *adopt*, not reproduce: I'm not trying to match their
number, I'm establishing a controlled baseline on my own data.

Two — quantify the degradation: how does accuracy fall as k = 0, 1, 2, 5, 10 μPMUs are removed.

Three — mitigate it, with something **architecture-agnostic and cheap**, because a mitigation
that only works on one model or costs 3× latency is not deployable.

Four — also test the obvious *architectural* alternative, and report the outcome honestly
whichever way it goes.

Five — validate everything statistically. With three seeds you cannot eyeball a curve and call it
a result."

**WATCH:** The word **"adopt"** is load-bearing. Never say "reproduce". If you say reproduce,
the next question is "did you match their 0.95?" and you're now defending a number you never
claimed.

---

### SLIDE 5 — Related Work

**SAY (45s):** "Four rows, and the last column is what matters.

Chen 2020 — spectral GCN on a static snapshot. They *do* use noise and data-loss augmentation,
so the idea isn't alien to the field. But there's no temporal axis at all; a fault is a
transient, and a single snapshot throws away the transient.

Zhang 2025 — imputation plus contrastive learning to fill in unmeasured nodes. Good method, but
the unmeasured set is assumed fixed and known ahead of time. A sensor failure is neither.

Karabulut 2026 — the STGNN I adopt: GRU for time, relational GNN for space. Strong results, no
loss mitigation, full availability assumed.

This work — same STGNN family, plus a systematic node-dropout study, plus dropout-aware training
as the mitigation."

**WATCH:** Don't skip Chen. If you skip it and a panel member has read it, "you missed prior work
on data-loss augmentation" is a bad moment. Naming it yourself and explaining why it doesn't
close the gap is a strong moment.

---

### SLIDE 6 — Dataset and Simulation Setup

Dense slide. Give the *shape* of the data, not every number.

**SAY (1:00):** "Everything is simulated in OpenDSS on the IEEE 123-bus feeder — 4.16 kV, 60 Hz,
one-millisecond time steps.

The design is fully factorial: 25 candidate fault buses × 11 fault types — three
line-to-ground, three line-to-line, three double-line-to-ground, two three-phase — × 3 fault
resistances, 0.1, 1 and 10 ohms. That's 825 distinct scenarios. Each is run 50 times with the
load multiplier drawn uniformly between 0.5 and 1.3, so the model can't memorise a single
operating point. 41,250 events.

Each event is a 60-millisecond record — 40 ms pre-fault, 20 ms faulted — cut into 20-step sliding
windows, 40 windows per event, 1.65 million windows total. The pre-fault windows are the no-fault
class, which is why the corpus is **exactly 50% no-fault by construction** — I don't need loss
weighting, the design balances it.

The split is the part I'd defend hardest: **event-level** 70/15/15, not window-level. If you split
windows randomly, 40 near-identical windows from one event get scattered across train and test
and every number you report is inflated. I verify zero leakage — no window of any event appears
in two splits. 6,175 test events."

**POINT:** the funnel diagram on the right, top to bottom, as you speak. It does the counting for
you.

**WATCH:** The bullets overlap the diagram. Physically stand so you're not also blocking it.

---

### SLIDE 7 — Graph Construction

**SAY (1:00):** "The graph is the modelling decision that makes this work.

Nodes are the **25 μPMU buses only** — not all 123. There is no point having nodes you have no
data for.

Edges come from Algorithm 1: a substation-rooted spanning tree over the measured set. Two μPMUs
are connected if the electrical path between them contains no third μPMU — so each edge is a
'stretch of feeder with nothing observed in between'. The result is a connected 25-node,
24-edge tree.

The important consequence: this graph is a **topological abstraction of the feeder**, so a fault
at an unmeasured bus doesn't vanish — it appears as a *signature pattern across its measured
neighbours*. That is exactly what a message-passing network is good at reading.

One detail that becomes essential later: each node has 6 features — three RMS phase voltages and
three **availability-mask channels**. Those masks are **static, and they are not all ones**. Only
14 of the 25 μPMUs are three-phase; the others are single- or two-phase and show patterns like
one-zero-zero. The masks encode per-node phase instrumentation and never change over time. Hold
that thought — it's the mechanism behind my main result."

**POINT:** the red-ringed nodes — those are the 6 buses that are both sensor and fault location.

**WATCH:** Do **not** say "the mask is always 1". It isn't, and slide 14's mechanism depends on
that being false.

---

### SLIDE 8 — Model Architecture

**SAY (50s):** "One template, five instantiations.

Input is 25 nodes × 20 timesteps × 6 features. A shared GRU with 128 hidden units encodes each
node's time series independently — same weights for every node, so it learns 'what a fault
transient looks like' once. Then four GNN layers do the spatial reasoning across the tree. Then
a head — BatchNorm, ReLU, dropout, linear to 26 classes — per node. Then the decision rule: a
**soft vote**, summing the softmax probabilities over all 25 nodes and taking the argmax.

The design point is that **the GNN block is the only thing that varies**. Same encoder, same head,
same decision rule, same data, same split, same seeds. So any accuracy gap between the five models
is attributable to spatial aggregation and nothing else. The five are: GRU with no graph at all —
that's my ablation, it tells me what the topology is worth — then RGCN, RGSAGE-Mean, RGSAGE-Max,
and RGATv2.

Two declared deviations from the base paper: I use depth 4 where they used 1, and 1.65 million
windows where they report 2.5 million. Both are stated in the paper as deviations, not hidden."

**POINT:** the blue "4 × GNN layers — the ONLY variable" box.

**IF ASKED "why depth 4?"** — "Depth-1 underperformed in my setting. The measured-only graph is a
24-edge tree with a diameter well above 1, so a single hop can't propagate a signature from an
unmeasured bus's neighbours to the rest of the network. I swept depth and froze 4 before running
any of the dropout experiments, so the choice isn't tuned on the result I'm reporting."

**IF ASKED "why declare the deviation instead of matching?"** — "The 2.5 million figure isn't
reconstructable from an 825-scenario factorial grid, so rather than reverse-engineer a number I
can't justify, I state my corpus size and report my own baseline."

---

### SLIDE 9 — Workflow

Fast slide. 30 seconds, no more.

**SAY:** "Four phases. Generate data and build the graph. Split at event level and train five
baselines at three seeds. Evaluate those trained models under sensor dropout to get degradation
curves. Then retrain dropout-aware, re-evaluate on the identical protocol, and run the statistics.

Two controls worth naming: one frozen dataset and one frozen split are reused by every single
model, and the dropout masks are **paired** — every model is evaluated on the identical dropped
node sets. So when two curves differ, it's the model, not the mask draw."

---

### SLIDE 10 — Evaluation Strategy

**SAY (45s):** "Metric is soft-voted **macro**-F1 over 26 classes. Macro, not accuracy — the
corpus is 50% no-fault, so a model that predicts 'no fault' always would score 50% accuracy and
be useless. Macro-F1 forces every one of the 25 fault buses to count equally. I also break results
down by fault resistance, because that's where the difficulty lives.

Robustness protocol: drop k of the 25 μPMUs at inference, k in 0, 1, 2, 5, 10. k=10 is 40% of the
fleet offline — deliberately extreme.

Two summary numbers so I'm not comparing five-point curves by eye: **AUDC**, the normalised area
under the degradation curve, where 1.0 means no degradation at all; and **retention at k=10**, the
F1 at k=10 as a fraction of F1 at k=0.

Every point on every curve is 3 training seeds × 3 mask seeds. Significance is paired two-sided
t-tests over training seeds with Holm–Bonferroni correction within each family, plus a cluster
bootstrap over the 6,175 test **events** — clustered by event, because 40 windows from one event
are not 40 independent samples."

**WATCH:** Fix the typo before you present. But if you're standing in front of it and notice it,
say nothing.

---

### SLIDE 11 — Results, All Sensors Available

Your first real result. Slow down.

**SAY (1:00):** "This is the clean benchmark, all 25 sensors up, mean ± standard deviation over
seeds 42, 43, 44.

Three readings.

First, **the graph is doing real work**. GRU with no graph gets 0.286. RGSAGE-Max gets 0.846.
That's a gain of 0.56 macro-F1 from message passing alone, on identical data with an identical
encoder. This is the ablation that justifies the entire approach.

Second, **max aggregation wins** — 0.846 versus 0.786 for mean. That's mechanistically sensible:
a fault is a *localized extremum* in the voltage field. Max-pooling preserves an extremum;
mean-pooling averages it against 23 healthy neighbours and washes it out.

Third, look at the resistance columns. 0.949 at 0.1 ohms, 0.893 at 1 ohm, 0.676 at 10 ohms.
**High-impedance faults are the hard case** — they produce a smaller voltage deviation, so there's
less signal to localize. That's the known hard problem in this field and my results reproduce it.

Two things I'll flag before anyone asks. No-fault precision is at or above 0.97 with recall
approximately 1.00 for every model — the system does not produce false alarms, which is the
property an operator cares about most. And best-validation epoch was 27.7 to 28.7 out of 30 for
the RGCN and RGSAGE models, meaning **they were still improving when the budget ended**. So treat
these accuracies as a floor, not a ceiling."

**POINT:** the GRU row, then the RGSAGE-Max row. The gap between them is your argument.

**IF ASKED "RGATv2 has std 0.087, that's huge"** — "Yes, attention is the least stable of the
five here. Three seeds spread from roughly 0.43 to 0.61. I report the standard deviation rather
than the best seed, and I don't make any claim that rests on RGATv2's clean accuracy."

---

### SLIDE 12 — Robustness Protocol

**SAY (40s):** "The dropout protocol precisely. At **inference only** — training is untouched for
these curves — I take a uniformly random subset of k μPMU nodes and zero **all six channels** of
each. That's the `zero_messages` policy: the node still exists in the graph, it just contributes
nothing.

k runs 0, 1, 2, 5, 10 out of 25. Three mask seeds by three training seeds at every point; the
error bars you'll see are the standard deviation across training seeds.

And the two summary metrics again — AUDC and retention at k=10."

**POINT:** the five little graph thumbnails; the red nodes are the dropped ones. This visual does
90% of the explaining.

**IF ASKED "why zero rather than delete the node from the graph?"** — "Zeroing is the honest model
of the physical failure: the utility's data pipeline still expects a record from that μPMU and
gets nothing. Deleting the node would require the model to handle a changing graph structure at
runtime, which is a different and stronger assumption about the deployment. I do test the
structural version — that's SDR-STGNN on slide 18."

---

### SLIDE 13 — Removing Sensors Destroys Every Model

**This is the money slide of the first half. Pause before you start.**

**SAY (1:00):** "Same trained models, same test set, nothing retrained. Only the sensors change.

RGSAGE-Max: 0.846 with all sensors, **0.048** with ten of twenty-five removed. It retains **5.7
percent**. That is not graceful degradation — that is total failure. It's barely above what you'd
get by guessing.

Now look down the retention column, because there's something counter-intuitive. RGSAGE-Max is
the **most accurate** model and the **most fragile**. RGATv2, the weakest model at k=0, retains
the most — 30 percent. RGCN retains 23 percent.

The interpretation: **clean accuracy does not predict robustness.** If you select a model the way
the field currently selects models — by the number in the k=0 column — you systematically pick
the most brittle one.

I want to be careful here: that inversion is a **trend, not a statistically significant result**.
It has the same sign in all three seeds and the ratio is between 2.4 and 6.7×, but the gap size
varies enough that at n=3 it comes out at p = 0.076. I'm reporting it as an observation worth
following up, not as a proven claim."

**WATCH:** Say the p = 0.076 caveat *yourself*. If a panel member extracts it from you, you look
like you were overclaiming. If you volunteer it, you look rigorous. This is the single highest-value
sentence in your talk for a research-panel audience.

**Why does max collapse hardest? (be ready, this is the likeliest technical question):**
"Max is a hard selector — it passes through exactly one neighbour's value per feature. If the
node carrying that extremum goes dark, the signal is gone, and there's no averaging to fall back
on. Mean aggregation degrades more gracefully because it's already pooling over survivors. So the
very mechanism that makes max the most accurate — sharp selection of the localized extremum —
is what makes it the most brittle."

---

### SLIDE 14 — Dropout-Aware Training

**SAY (1:15):** "The fix is not a new model. It's teaching the existing model what failure looks
like.

For every training batch: sample k uniformly from 0 up to k_max, pick a random subset of k nodes,
zero all six of their channels, and train exactly as before. Four lines of code. It is a **data
augmentation**, not an architecture change — every model keeps its exact structure and parameter
count.

Now, *why* does something this trivial work so well? This is the part I'd most like to explain,
because it isn't just 'augmentation helps'.

Remember the mask channels from the graph slide — they're static, and only 14 of the 25 nodes are
three-phase. So in normal training, every node presents a **fixed, node-specific mask pattern**:
one-one-one, or one-zero-zero, or one-one-zero. The network learns those as identity fingerprints.

When you drop a node at inference, you zero all six channels — so its mask becomes
**zero-zero-zero**. That is a pattern **no node ever exhibits in standard training**. The model is
being handed a genuinely out-of-distribution input, and it has no learned response to it. That is
the mechanism of the collapse.

Dropout-aware training simply puts that pattern into the training distribution. The model learns
that zero-zero-zero means 'this node is dark, route around it' rather than 'this is some node I've
never met'.

And the cost: **identical parameter count, identical inference latency, plus 2.7 percent training
time.** That's measured, not asserted."

**WATCH:** This slide has a formatting problem — the pseudocode is squeezed inline. If you have
time, break `k ~ U{0..k_max}` and `S ~ random subset` onto separate lines.

**IF ASKED "isn't this just standard dropout?"** — Best question you can get; have this ready:
"No, and the distinction matters. Standard dropout zeroes random *hidden units* to regularise.
This zeroes entire *input nodes*, structurally, in a pattern that matches a real physical failure
mode. And the key difference is the effect on the mask channels — standard dropout doesn't create
a coherent all-zero availability signature. It's closer to DropNode, but the purpose is inverted:
DropNode is a regulariser aimed at clean accuracy, this is a train-test distribution match aimed at
a specific deployment failure."

**IF ASKED "what is k_max?"** — "Training samples k from 0 to 5. I evaluate up to k=10. So k=10 is
**twice the maximum dropout level the model ever saw in training** — the retention result at k=10
is genuine generalization beyond the training regime, not memorisation of the test condition."
*(That is a strong point. Make it.)*

---

### SLIDE 15 — The Degradation Curve (full-bleed figure)

**Say almost nothing. Let them look.** Then trace with your hand.

**SAY (40s):** "One picture for the whole contribution. X-axis is sensors dropped, zero to ten.
Y-axis is soft-voted macro-F1.

Every solid line is a standard-trained baseline. All five fall off a cliff — and notice the orange
one, RGSAGE-Max, starts highest and ends lowest; it crosses *below* three other models on its way
down.

The two flat lines at the top are mine. The teal is RGSAGE-Max with dropout-aware training. The
black dashed is the gated architecture. They stay essentially flat out to 40% sensor loss.

The gap between the orange line and the teal line at k=10 is the result."

**POINT:** Physically trace orange from top-left down to bottom-right, then teal straight across.
The crossing pattern is the visual argument — use it.

**Add a title to this slide before presenting** (see 0.4).

---

### SLIDE 16 — Dropout-Aware Training Results

**SAY (1:00):** "Four variants of RGSAGE-Max, same everything else.

Standard training: AUDC 0.402, retention 5.7%.

Neighbour-mean imputation — reconstruct each dropped node's voltages from the per-timestep mean of
its available one-hop neighbours, and set its mask to 1, which is the *most favourable* setup for
imputation. Retention goes 5.7% to 11.8%. It roughly doubles, and it's still catastrophic. And
that difference isn't statistically significant at n=3 — p = 0.102.

Dropout-aware training: retention **80.3%**, AUDC 0.913. That's **+0.626 macro-F1 at k=10 over
standard training**, and it survives Holm correction at p = 0.003.

Clean accuracy: 0.839 versus 0.846. A cost of 0.008, p = 0.181 — that **tests as null**, which is
what I claim. I am not claiming it improves clean accuracy; I am claiming it doesn't hurt it, and
the statistics support 'doesn't hurt'.

And it's free at deployment: identical parameters — 95,258 in both — identical latency, 0.639
versus 0.640 milliseconds, about 6,000 windows per second either way.

The imputation result is worth a sentence of interpretation. **Why does imputation barely help?**
Because it fixes the wrong half of the problem. It reconstructs a plausible number, but it feeds
that number to a model that was never taught to distrust it — the model treats an inferred voltage
with exactly the same confidence as a measured one. Dropout-aware training fixes the model's
*expectations*, which is where the failure actually lives."

**POINT:** the 5.7% → 80.3% jump. That's the number they'll remember.

---

### SLIDE 17 — Across All Five Architectures

**SAY (45s):** "The mitigation is not a property of one model.

All four spatial GNNs jump to between 80 and 91 percent retention — AUDC goes from the 0.40–0.63
range to 0.91–0.96. And that AUDC gain survives Holm correction on **every one of the five
architectures**, p ≤ 0.027. That's what makes the claim 'architecture-agnostic' rather than
'worked on my favourite model'.

Two honest qualifications. GRU gains least — 17.9 to about 25 percent — and the reason is
structural: with no message passing there is nowhere to route around a dark node. Augmentation
can't create a path that the architecture doesn't have. That's actually confirmatory evidence for
the mechanism I described.

And the clean-accuracy cost is **not uniform**. It's essentially zero on RGSAGE-Max, but it costs
roughly 0.1 macro-F1 on RGCN and RGSAGE-Mean. So there is a robustness–accuracy trade-off for
some architectures; on the best one there isn't. I report both."

**WATCH:** Volunteering the non-uniform cost is worth a lot. It's the kind of caveat panels probe
for, and getting there first defuses it.

---

### SLIDE 18 — SDR-STGNN: An Honest Negative Result

**SAY (50s):** "I also built the architectural answer, because it's the obvious thing to try. SDR-STGNN
is RGSAGE-Max with **availability-gated aggregation**: a dropped node emits no messages at all,
and the gate is re-applied at every hop keyed on the node's original availability. At k=0 it's
mathematically identical to RGSAGE-Max.

It **matches** the training fix and does not beat it. AUDC 0.892 versus 0.913, retention 75.2
versus 80.3. The difference is 0.021, p = 0.098 — statistically that is a **tie**. And it costs
18 percent more inference latency, 0.755 against 0.639 milliseconds, for zero robustness gain.

The explanation is the interesting part. Silencing a dropped node at *every hop* also destroys the
**multi-hop paths that route information around the failure**. A node that is itself dark can still
be a useful relay — its neighbours' information passes through it. The gate is too aggressive: it
buys local correctness and pays with global connectivity.

The takeaway I draw: **dropout robustness is best bought in training, not baked into the
architecture.**"

**WATCH:** Under no circumstances describe SDR-STGNN as a win, an improvement, or "slightly worse
but promising". It is a tie that costs latency. Presenting a clean negative result deliberately is
a *strength* in a research review — say the words "negative result" out loud and own it.

**IF ASKED "then why keep it in the work at all?"** — "Because it's the natural first hypothesis,
and the negative result is informative. Without it, a reader would reasonably ask 'wouldn't a gated
architecture do better?' Now there's an answer, with a mechanism."

---

### SLIDE 19 — Harder Failure Modes

**SAY (1:00):** "Random dropout is the easy case. Two stress tests.

**Targeted — worst case.** Instead of dropping random sensors, drop the k μPMUs *topologically
nearest the fault*, by hop distance on the full 123-bus topology. Selection is a deterministic
function of the label, so there are no mask seeds here — error bars are over training seeds only.

This hurts everyone, and it hurts where it matters. Standard RGSAGE-Max falls to 0.448 at k=1,
versus 0.736 under random loss — a **3.6× larger** drop from clean, p = 0.010.

Dropout-aware training is **also penalized** — AUDC drops from 0.913 to 0.747. So worst-case loss
is a **distinct stressor, not one the mitigation absorbs**. I'd rather state that than pretend the
fix is universal. But it still retains between +0.31 and +0.47 macro-F1 over standard training
under identical targeting, p ≤ 0.008. And from k=2 upward, the dropout-aware model's *worst-case*
score beats the standard model's *average-case* score — 0.695 against 0.590.

One honesty note: no-fault windows have no fault bus, so they fall back to random selection. That
means my reported worst-case number is a **conservative bound** — the true worst case is somewhat
worse.

**Persistent — per-event.** Real μPMUs fail for minutes, not milliseconds. So instead of redrawing
the dropped set every window, I draw it once per event and hold it across all 40 windows of that
event. Result: a **clean null**. No point on any curve moves by more than 0.003 macro-F1, and
nothing survives Holm — minimum p is 0.46.

And I can explain *why* it's null, which is the point of running it: the decision rule is
per-window. Soft voting pools the 25 nodes of **one window**, never the 40 windows of an event. So
a window's prediction depends only on its own mask, and the marginal distribution of that mask is
identical under both schemes. Important caveat — this equivalence would **not** hold for an
event-level decision rule, where per-window resampling would flatter you by supplying 40
differently-masked views of a failure that only happens once."

**WATCH:** A null result explained by a mechanism is worth far more than a null result reported. The
last three sentences are what make this slide count. Don't rush them.

---

### SLIDE 20 — Conclusion

**SAY (45s):** "To close.

A measurement: clean accuracy does not predict robustness — the strongest model degraded fastest,
retaining 5.7% where RGCN retained 23%.

A fix: dropout-aware training takes retention from 5.7% to 80.3%, at zero parameter and zero
latency cost, on all five architectures.

A negative result: an availability-gated architecture only matches it, and costs latency.

The mitigation holds up against imputation, against worst-case targeting, and against realistic
per-event loss.

The practical statement: **an operator can lose about 10 of 25 μPMUs and still localize faults at
macro-F1 above 0.67 — with no new hardware, just a different training loop.**

Future work: spatially correlated failures — a comms hub taking out a whole region, which is the
one failure geometry I haven't tested — larger and reconfigured feeders, and adaptive gating.

This work is written up and was submitted to ICMLDE, Procedia Computer Science."

**WATCH:** Finish on the operator sentence, not on the paper submission. The deployment claim is
the memorable one.

---

### SLIDE 21 — References / SLIDE 22 — Thank You

**SAY:** "Thank you — happy to take questions." Don't read the references.

---

## PART 3 — NUMBERS CHEAT SHEET

Memorise the **bold** ones. Have the rest available if pressed.

**Clean baselines (macro-F1, mean ± std over seeds 42/43/44)**
GRU 0.286 ± 0.009 · RGATv2 0.522 ± 0.087 · RGCN 0.712 ± 0.017 · RGSAGE-Mean 0.786 ± 0.023 ·
**RGSAGE-Max 0.846 ± 0.007**

**Retention @ k=10, standard training**
**RGSAGE-Max 5.7%** · RGSAGE-Mean 13.8% · GRU 17.9% · RGCN 23.1% · RGATv2 30.1%

**Retention @ k=10, dropout-aware**
**RGSAGE-Max 80.3%** · RGSAGE-Mean 81.9% · RGCN 87.0% · RGATv2 91.4% · GRU 25.4%

**RGSAGE-Max mitigation ladder (AUDC → Ret@10)**
standard 0.402 → 5.7% · imputation 0.476 → 11.8% · **dropout-aware 0.913 → 80.3%** ·
SDR-STGNN 0.892 → 75.2%

**Significance — what IS significant**
DA vs standard @k=10: Δ **+0.626**, p_Holm = **0.003** · Ret@10 +74.6 pp · AUDC +0.511 ·
DA vs imputation @k=10: +0.574, p_Holm = 0.008 · AUDC gain on all 5 architectures p_Holm ≤ 0.027

**Significance — what is NOT (never call these significant)**
accuracy/robustness inversion p = 0.076 · k=0 architecture ranking (nothing survives Holm) ·
imputation over zeroing @k=10 p = 0.102 · clean-F1 cost −0.008, p = 0.181 (null, as claimed) ·
DA over SDR AUDC +0.021, p = 0.098 (a tie, as claimed)

**Cost**
params 95,258 both · latency 0.639 vs 0.640 ms @ batch 1 · ~6,038 windows/s both ·
training +2.7% sec/epoch · SDR-STGNN +18% latency (0.755 ms), 0 extra params ·
activation memory 145 MiB for every model (the shared GRU-128 dominates) · single T1000 8 GB

**Data**
825 scenarios (25 buses × 11 types × 3 resistances) · 50 runs each · 41,250 events ·
**1.65 M windows** · exactly 50% no-fault · event-level 70/15/15 · **0 leakage verified** ·
6,175 test events · one sample = 25 × 20 × 6

**Graph**
25 nodes, 24 edges, connected tree · **6 of 25 fault buses have a μPMU → 19 unmeasured** ·
**14 of 25 μPMUs are three-phase** · masks static, node-specific, not all ones

**Training**
depth 4 · 30 epochs · grad-clip 1.0 · 3 warm-up epochs · unweighted cross-entropy ·
best-val epoch 27.7–28.7 of 30 for RGCN/RGSAGE → **accuracies are a floor**

---

## PART 4 — QUESTION BANK

### On the problem and gap

**"Is this gap real, or has someone done it?"**
See slide 3 notes. Name Chen 2020, Zhang 2025, Karabulut 2026 and say precisely why each doesn't
close it.

**"Why is this a research contribution and not engineering?"**
"Three things. A measurement nobody had made — the degradation curve for STGNN fault localization.
A mechanism — why the collapse happens, which is the all-zero mask signature going out of
distribution. And a falsified hypothesis — the architectural gate, which was the obvious answer and
turns out to be the wrong one. The mitigation itself is simple; simplicity is a feature when you're
proposing something a utility could deploy."

### On the data

**"It's all simulation — does it transfer to real feeders?"**
"That's the honest limitation and it's stated in the paper. Everything is OpenDSS on IEEE-123 with
no measurement noise, no CT/PT error, no communication jitter. What I'd argue transfers is the
*mechanism*: the collapse is caused by a train–test distribution mismatch in the availability
signature, which exists in any deployment where availability is encoded in the input. The specific
numbers would need field μPMU data to confirm, and that's future work."

**"1.65 M windows but the base paper says 2.5 M — did you under-train?"**
"It's a declared deviation. 2.5 million isn't reconstructable from an 825-scenario factorial grid at
40 windows per event, so rather than invent a configuration to match a number, I state my corpus and
report my own baseline. And I never claim to match their accuracy — I adopt the methodology, not the
result."

**"Why only 25 μPMUs? Why those buses?"**
"That placement follows the base paper, and it reflects realistic partial observability — utilities
instrument a small fraction of buses. The important consequence for my work is that only 6 of the 25
candidate fault buses carry a sensor, so 19 must be inferred through the topology."

**"Is 50% no-fault realistic? Real feeders are 99.99% no-fault."**
"No, and it's by construction, not by accident. The 50/50 comes from the windowing — 40 ms pre-fault
and 20 ms faulted per event. It gives a class-balanced training corpus, which is why I use
unweighted cross-entropy. For a deployment prior you'd recalibrate the no-fault threshold; the
localization performance among the 25 fault classes is what I'm measuring, and that's unaffected."

### On the method

**"Why event-level split? Isn't that just being cautious?"**
"It's the difference between a valid and an invalid result. 40 windows from one event share the same
load multiplier, same fault resistance, same fault type — they're near-duplicates. A random window
split puts duplicates in train and test, and every number inflates. I verify zero leakage
programmatically, not by inspection."

**"Why macro-F1?"**
"26 classes with 50% no-fault. Accuracy would be dominated by the no-fault class; a degenerate
always-no-fault model would score 50%. Macro-F1 weights all 26 equally, so a model that's good at
detection and bad at localization scores badly — which is correct, because localization is the task."

**"Why soft voting rather than pooling node embeddings?"**
"Soft voting lets every node cast a probability vote and sums them, so a node with a strong local
signature can carry the decision even if most nodes are ambivalent — which is exactly the structure
of a localized fault. It also degrades gracefully: a dropped node contributes a diffuse vote rather
than corrupting a pooled representation."

**"Isn't dropout-aware training just standard dropout / DropNode?"**
See slide 14 notes. Key points: input-node-level not hidden-unit-level; structured to match a
physical failure; creates the all-zero mask signature; purpose is train–test distribution match, not
regularization; and it's evaluated at 2× the training k_max.

**"You train to k=5 but evaluate at k=10 — isn't that unfair to the baselines?"**
"It's the opposite of unfair — it's the harder condition for *my* method. The baselines are
evaluated at exactly the same k values. The point of going to k=10 is that it's beyond anything the
dropout-aware model saw in training, so the 80.3% retention there is genuine generalization rather
than the model having memorised the test-time corruption."

### On the results

**"n = 3 seeds. That's not enough for a t-test."**
"Agreed that it's the floor, and I treat it that way. Three things. I use paired tests over the same
seeds and Holm correction within each family, so I'm not p-hacking across a dozen comparisons. I
report Wilcoxon but don't rely on it — at n=3 its minimum achievable p is 0.25, so it can't reject
anything. And I explicitly do **not** upgrade borderline results: the accuracy/robustness inversion
at p = 0.076 is reported as a trend. The claims I do make are at p ≤ 0.008, with effect sizes above
+0.6 macro-F1 — that's a gap that doesn't need a large n to be visible. I also run a cluster
bootstrap over the 6,175 test events at k=0, clustered by event, for a sample-level check."

**"Why is the most accurate model the most fragile?"**
See slide 13 notes — max is a hard selector, no averaging fallback.

**"Why doesn't dropout-aware training cost clean accuracy?"**
"Because the task has spatial redundancy the standard model wasn't exploiting. A fault signature
appears across several measured neighbours, not just one. Standard training lets the model latch
onto the single most informative node; dropout-aware training forces it to spread the representation
across the neighbourhood. The redundant representation happens to be about as good on clean data
and far better under loss. That said, it's not free everywhere — on RGCN and RGSAGE-Mean it costs
about 0.1 macro-F1."

**"You said 80.3% retention — retention of a number that itself dropped. Isn't that misleading?"**
"Fair, which is why I report both. Absolute macro-F1 at k=10 is 0.674 for dropout-aware against
0.048 for standard. Clean accuracy is 0.839 versus 0.846. So the absolute numbers tell the same
story as the ratios."

**"Your gated model doesn't beat the simple fix. Isn't that a failed contribution?"**
"It's a reported negative result, and I'd argue that's a contribution rather than a failure. It
tests the natural hypothesis — that robustness should be structural — and falsifies it, with a
mechanism: gating at every hop cuts the multi-hop relay paths. The paper frames dropout-aware
training as the contribution and SDR-STGNN as an ablation, which matches what the data supports."

### On deployment

**"What would a utility actually do with this?"**
"Change the training loop, redeploy the same model file. Same architecture, same 95,258 parameters,
same 0.64 ms inference. There is no hardware implication and no change to the operational pipeline.
The claim is: you can lose 10 of your 25 μPMUs and still localize at macro-F1 above 0.67."

**"What about correlated failures — a substation comms outage takes out a whole region?"**
"That's the honest hole, and it's Phase 2 of the Semester 3 plan. My random dropout is spatially
independent; my targeted experiment is adversarial-by-proximity, which is a partial proxy since the
nearest-to-fault μPMUs are also topological neighbours of each other. But a genuine region-wide
correlated failure is untested and I'd expect it to be harder than random and closer to targeted."

**"How long does it take to train?"**
"30 epochs on a single T1000 8 GB. Dropout-aware adds 2.7% to sec/epoch. And best-validation epoch
was 27.7–28.7 out of 30 for RGCN and RGSAGE, so the models were still improving — the accuracies I
report are a floor under a fixed budget, not converged optima."

### If you genuinely don't know

Say: *"I haven't tested that. My instinct is X, but I'd want to run it before claiming it."*
That answer costs you nothing. Guessing and being wrong costs a lot.

---

## PART 5 — DELIVERY

**Structure your voice around three beats.** 0.846 (good) → 0.048 (dead) → 0.674 (saved). Every
slide serves one of those three. If you're lost mid-talk, ask yourself which beat you're on and
move toward the next one.

**Pause before slides 13, 15, 16.** Two full seconds of silence before a number is what makes it
land. Most presenters rush their best slide.

**Never read a bullet.** Slides 2, 4, 6, 7, 16 and 20 are text-heavy — if you read them the panel
reads ahead and stops listening. Say the *idea*, let them read the *detail*.

**Say your caveats before they're extracted from you.** The p = 0.076 on slide 13, the non-uniform
clean cost on slide 17, the "tie" on slide 18, the conservative-bound on slide 19. Volunteering
these is what separates a student presenting results from a researcher presenting evidence.

**Use the figure on slide 15 physically.** Trace orange down, teal across. That gesture is worth
more than the paragraph you'd otherwise say.

**When you're asked something you've already answered on a slide:** don't say "as I mentioned".
Just answer it again, shorter.

**Don't apologise for the negative result, the simulation-only data, or the n=3.** State each as a
scoped limitation with a reason. Apologising invites the panel to treat it as a flaw; scoping
invites them to treat it as a design decision.

---

## PART 6 — DO NOT SAY

1. **"Reproduce."** You *adopt* Karabulut's methodology and report your own numbers. Saying
   "reproduce" invites "did you match their 0.95?" — a question about a claim you never made.
2. **"SDR-STGNN is better / an improvement / promising."** It is a **tie** (Δ = 0.021, p = 0.098)
   that costs 18% latency. Say "matches".
3. **"The inversion is significant."** p = 0.076. It's a trend with a consistent sign.
4. **"Imputation is significantly worse than dropout-aware at every level."** DA vs imputation at
   k=10 *is* significant (p_Holm = 0.008). Imputation vs plain zeroing is **not** (p = 0.102).
   Don't merge those two claims.
5. **"The mask is always 1."** It isn't — masks are static and node-specific, only 14 of 25 nodes
   are three-phase. Your whole mechanism depends on this being false.
6. **The green-feeder / reconfigured-topology experiment as something you did.** It was cut from
   scope. It exists only as a future-work clause.
7. **"Dropout-aware training solves worst-case loss."** It doesn't — targeted dropout penalizes it
   too (AUDC 0.913 → 0.747, p = 0.003). It *reduces* the damage substantially; it does not absorb it.
8. **Any number you haven't checked.** If you're unsure mid-answer, say "I'd have to check the
   exact figure" rather than producing one.
