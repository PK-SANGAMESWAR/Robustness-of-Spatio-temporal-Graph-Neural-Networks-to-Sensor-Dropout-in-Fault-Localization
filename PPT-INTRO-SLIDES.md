# Intro slides — Fault Localization on Power Grids

Background/motivation slides to sit **before SLIDE 2** of `PPT-GUIDE-REVIEW.md`
(which jumps straight into "What problem am I solving"). These give the examiner
the domain context first, then hand off to your existing Slide 2.

Every external statistic below is sourced. Every project number is from our own
`results/` and is already in the paper — do not change one without changing the other.

---

## SLIDE A

**Title:** Why Fault Localization Matters

```
WHERE OUTAGES ACTUALLY COME FROM
•  ~92% of all customer outages originate in the DISTRIBUTION system —
   not generation, not transmission
•  US customers averaged 11 hours without power in 2024, nearly 2x the
   previous decade's average (hurricanes = 80% of those hours)
•  Excluding major weather events, interruptions run ~2 hours/customer/year

WHAT IT COSTS
•  Sustained power interruptions cost the US ~$44 billion/year (LBNL, 2015$)
•  Earlier LBNL estimate ~$80B/yr; range $30B-$130B
•  EPRI, including power-quality events: $119B-$188B/yr

THE PART THAT TAKES TIME
•  A protection relay DETECTS a fault in milliseconds
•  Crews then have to FIND it — that is the restoration clock
•  Faster localization = smaller patrol area = shorter outage
```

**Speaker note:** The headline is the first bullet. Reliability investment tends
to be discussed at the transmission level, but nine out of ten customer-minutes
lost happen on the distribution feeder — which is exactly the part that is
barely instrumented. That mismatch is the opening for this work.

**Sources:** Eaton, *Distribution System Reliability and Outage Rate Analysis*;
RMI, *Reliability Explored*; EIA, *Today in Energy* (2024 reliability data);
LBNL, *Updated Value of Service Reliability Estimates*.

---

## SLIDE B

**Title:** Detection is Easy. Localization is the Hard Part.

```
THREE DIFFERENT QUESTIONS

1. DETECTION      Did a fault occur?          -> solved; relays, milliseconds
2. CLASSIFICATION What kind of fault?          -> LG / LL / LLG / 3-phase
3. LOCALIZATION   WHERE on the feeder?         -> the expensive one

WHY (3) IS DIFFERENT
•  A feeder is kilometres of branching line with many laterals
•  "A fault occurred somewhere downstream" can mean hours of patrolling
•  Automated restoration schemes cannot re-energise what they cannot isolate

HOW WE POSE IT
•  26-class classification: 25 candidate fault buses + 1 no-fault class
•  IEEE 123-bus test feeder, the standard benchmark for this problem
```

**Speaker note:** Worth saying out loud that our own results confirm this
ordering — at full observability the models reach 0.990 F1 on detection but
0.841 on localization. The difficulty is entirely in the "where".

---

## SLIDE C

**Title:** How Fault Localization is Usually Done

**Table:**

| Family | How it works | Strengths | Why it struggles here |
|---|---|---|---|
| **Impedance-based** | Measure V and I at the substation, compute apparent impedance, convert to distance | Cheap; only substation instrumentation; long-established | *Multiple-estimation problem*: one impedance maps to several points on a branching feeder. Needs accurate line parameters and is confounded by fault resistance |
| **Traveling-wave** | Time the arrival of the fault-induced wavefront and triangulate | Very accurate; largely independent of fault resistance | Needs MHz sampling and GPS time-sync; costly hardware; reflections off many laterals make distribution feeders hard |
| **Data-driven / ML** | Learn the mapping from measured signatures to fault location | No exact circuit model needed; handles nonlinearity | Needs training data; classical feature-vector models discard which buses are electrically adjacent |
| **Graph neural networks** | Encode the feeder as a graph so message passing shares information between connected buses | Can infer faults at buses carrying **no sensor** | Assumes the sensors it does have are always reporting — the gap this project attacks |

```
THE PROGRESSION IN THE LITERATURE
   Krishnathevar & Ngu (2012)   generalised impedance formulation
   Liu et al. (2023)            traveling wave + improved EEMD
   Livani et al. (2013)         SVM on post-fault RMS voltages
   Chen et al. (2020)           spectral GCN on the feeder graph
   Karabulut et al. (2026)      spatio-temporal GNN  <- our base paper
```

**Speaker note:** The arc to emphasise: each family relaxes an assumption of the
previous one. Impedance methods need a circuit model; ML relaxes that; GNNs add
back the topology that feature-vector ML threw away; STGNNs add the time axis.
Our contribution relaxes the next assumption in line — that every sensor reports.

---

## SLIDE D

**Title:** Why Distribution Grids Make This Hard

```
PARTIAL OBSERVABILITY IS THE DEFINING CONSTRAINT
•  Transmission networks are densely instrumented. Distribution networks are not
•  micro-PMUs are expensive AND need communication infrastructure
•  A feeder with hundreds of buses may carry only a handful

ON OUR BENCHMARK (IEEE 123-bus)
•  25 buses instrumented with micro-PMUs
•  25 candidate fault locations
•  Only 6 of those 25 carry a sensor  ->  19 fault locations are NEVER
   directly measured. The model must infer them through the topology

AND THE SIGNAL ITSELF IS WEAK
•  Three-phase-to-ground fault at bus 101, deepest voltage sag observed:
      0.1 ohm  ->  0.83 p.u. below pre-fault   (obvious)
      1 ohm    ->  0.22 p.u.                   (visible)
      10 ohm   ->  0.01 p.u.                   (nearly invisible)
•  High-impedance faults are the hardest class throughout this study
```

**Speaker note:** These two facts together are the justification for a graph
model: 19 of 25 fault locations are invisible to any direct measurement, so the
only way to reach them is to propagate information across the topology from
buses that *are* measured. The resistance figures are measured from our own
generated corpus, not quoted from the literature.

---

## HOW THIS HANDS OFF

Slide D ends on partial observability. Your existing **SLIDE 2** then opens with
"Sensors are sparse: only 25 micro-PMUs on a 123-bus feeder" — so the two join
without repetition. If you are tight on time, **A and D are the two to keep**:
A establishes that the problem is worth solving, D establishes that it is hard.
B and C can be compressed into one slide of spoken context.
