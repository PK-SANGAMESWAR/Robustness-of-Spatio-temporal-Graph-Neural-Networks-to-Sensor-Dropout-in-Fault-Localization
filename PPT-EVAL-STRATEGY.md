# Evaluation strategy — slide content

Sits after the methodology slides in `PPT-GUIDE-REVIEW.md`. This is the part of
the talk that earns marks for rigour, so it is written to be defensible under
questioning rather than impressive at a glance.

Figure to use: `paper/figures/ppt_eval_strategy.png` (built by
`scripts/build_ppt_architecture.py`).

Every number below is from `results/` and is already in the paper.

---

## SLIDE E1

**Title:** The Core Idea — Break It at Inference, Not at Training

```
THE SETUP
•  Train the model ONCE, with the sensor set it was designed for
•  FREEZE the weights
•  At test time, remove k micro-PMUs and re-evaluate

WHY FREEZE
•  A model that was retrained for each k would be answering a different
   question — "how good is a 15-sensor model?"
•  We are asking the deployment question: the model is already in the
   field when a sensor goes dark

THE SWEEP
   k =  0    1     2      5      10        of 25 sensors
        0%   4%    8%     20%    40%       sensor loss
```

**Speaker note:** k = 0 is the control. It must reproduce the checkpoint's
recorded clean score exactly — we assert this in code, and a mismatch aborts the
run. That one check is what makes every other number in the sweep trustworthy.

---

## SLIDE E2

**Title:** What "Dropping a Sensor" Means, Precisely

```
FOR EACH DROPPED NODE, ZERO ALL SIX CHANNELS
•  3 RMS voltage channels  -> 0
•  3 availability-mask bits -> 0

WHY THE MASK MUST ALSO GO TO ZERO
•  A zero VOLTAGE is a legitimate reading on some buses
•  The mask is the only unambiguous signal that a node is unobserved
•  Zeroing voltages alone would be indistinguishable from a real sag

WHAT WE DO NOT DO
•  We do NOT remove the node from the graph
•  We do NOT remove its edges
•  It stays in the topology and keeps passing messages —
   its zeroed embedding propagates to its neighbours

   ^ that propagation IS the phenomenon under study
```

**Speaker note:** An edge-removal policy is implemented in the codebase but
deliberately unused: removing edges would change the topology at the same time
as the measurement, and we would no longer know which of the two caused the
degradation. One variable at a time.

---

## SLIDE E3

**Title:** Fairness — Every Model Sees the Same Failures

```
IDENTICAL DATA
•  One frozen test split: 247,000 windows / 6,175 fault events
•  Event-level partition — no window of a test event is ever trained on

IDENTICAL MASKS
•  mask_seed = 10000 x k + offset, then combined with the batch index
•  Deterministic and model-independent
•  => at a given (k, offset) the SAME sensors are dark for every
     architecture, on the same batch, in the same order

THIS IS A PAIRED STUDY, NOT SIX INDEPENDENT ONES
•  3 mask realizations   x   3 training seeds (42, 43, 44)
•  Error bars are over TRAINING seeds — the dominant source of variance
```

**Speaker note:** This mattered in practice. An earlier version of the evaluator
passed a constant mask seed, so the same subsets repeated on every batch and the
number of distinct realizations collapsed to the batch size. Fixing it changed
results in the 2nd-3rd decimal and no conclusion, but every number was
regenerated afterwards rather than patched.

---

## SLIDE E4

**Title:** Metrics — and Why One Number Was Not Enough

```
PRIMARY
   Soft-voted macro-F1 over 26 classes
   (sum the softmax of all 25 nodes, take the argmax)

DEGRADATION
   Retention        R(k) = F1(k) / F1(0)
   AUDC             normalized area under R(k), in [0, 1]
                    higher = degrades more slowly

THE PROBLEM WITH RETENTION, AND THE FIX
•  It divides by the model's OWN clean score, so a weak model can post
   a flattering number
•  RGATv2: 1st on relative AUDC (0.626), 4th of 5 on ABSOLUTE (0.324)
•  => we report AUDC_abs alongside it — same integral, no normalization

WHAT DOES "COLLAPSED" ACTUALLY MEAN?
•  A classifier that always answers "no fault" scores macro-F1 = 0.026
•  RGSAGE-Max at k=10 scores 0.048  ->  only 1.9x that floor
```

**Speaker note:** The last box is the one to dwell on. "5.7% retention" sounds
like something survived; "1.9 times a classifier that always says no-fault"
tells you it did not. Reference points matter more than ratios.

---

## SLIDE E5

**Title:** We Test Five Different Ways a Sensor Fleet Can Fail

**Table:**

| Policy | What it models | Finding |
|---|---|---|
| **Random / scattered** | Independent, uniform failures — the average case | Baselines collapse: AUDC 0.402 for the strongest model |
| **Targeted** | Worst case — the k sensors *nearest the fault* go dark | Far more damaging: 0.448 vs 0.736 at k=1. Mitigation is penalised too (AUDC 0.913 → 0.747) |
| **Persistent** | A failed unit stays dark for the whole event, not redrawn per window | A clean null — nothing moves by more than 0.003 |
| **Correlated** | A contiguous block fails — shared pole, comms path, feeder segment | Milder than scattered, not harsher. Does not survive Holm at n=3 |
| **Measurement noise** | Sensors report, but inaccurately (IEEE C37.118 allows 1% TVE) | 1% noise ≈ losing 4.1 sensors. Dropout-aware training gives **no** protection |

```
PLUS A TEST-TIME ALTERNATIVE
   Imputation: fill dropped nodes with the neighbour mean before inference
   -> AUDC 0.476 vs 0.402 for plain zeroing — not significant (p = 0.102)
```

**Speaker note:** The point of five policies is that a single missingness model
would have let us claim the mitigation is general when it is not. Targeted loss
penalises it, noise defeats it entirely. Reporting that is the difference
between an evaluation and a demonstration.

---

## SLIDE E6

**Title:** Statistics — Being Honest About n = 3

```
THE DESIGN
•  3 training seeds (42, 43, 44), every variant trained from all three
•  The SEED is the paired unit — same data, same splits, so pairing
   removes seed-to-seed training difficulty
•  Paired two-sided t-test, n = 3, df = 2, with 95% CIs
•  Holm-Bonferroni correction applied PER FAMILY of tests

WHAT WE DELIBERATELY DO NOT USE
•  Wilcoxon signed-rank: its smallest possible two-sided p at n = 3
   is 0.25, so it can never show anything. Computed, never relied on

A SECOND, INDEPENDENT CHECK
•  Cluster bootstrap over the 6,175 test EVENTS — not windows, because
   one fault contributes 40 correlated windows
•  It shows the limit is the number of SEEDS, not the size of the test set
```

---

## SLIDE E7

**Title:** What We Are *Not* Claiming

```
SURVIVES CORRECTION — we claim these
•  Dropout-aware > standard at k=10:  +0.626 macro-F1,  p_Holm = 0.003
•  Retention at k=10:                 +74.6 points,     p_Holm = 0.003
•  AUDC:                              +0.511,           p_Holm = 0.003
•  Holds for ALL FIVE architectures:                    p_Holm <= 0.028

DOES NOT SURVIVE — we report these as observations only
•  Accuracy/robustness inversion:     p = 0.076
   (same sign in all 3 seeds, but the SIZE of the gap varies)
•  Contiguous vs scattered:           min p_Holm = 0.054
•  Imputation over zeroing:           p = 0.102

TESTS AS NULL — and we claim it AS a null
•  Clean-accuracy cost of the augmentation:  -0.008,  p = 0.181
•  SDR-STGNN vs dropout-aware training:      +0.021,  p = 0.098
   => the architecture MATCHES the training fix. It does not beat it
```

**Speaker note:** This is the slide to leave up if the panel asks about rigour.
Anyone can report the wins. Being able to say precisely which of your own
results did not reach significance — and reporting the gate as a tie when it
would be easy to call it a win — is the harder and more convincing thing.

---

## IF YOU ONLY HAVE THREE SLIDES

Keep **E1** (freeze and break at inference), **E5** (five failure modes), and
**E7** (what we are not claiming). E1 establishes the design, E5 shows breadth,
E7 shows you can be trusted with the numbers.
