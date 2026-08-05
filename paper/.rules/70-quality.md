# 70 — Quality Bar

Goal: a paper that survives a critical reviewer, not one that merely reads smoothly. Fluent
prose around a weak contribution gets rejected. Plain prose around a clear, well-evidenced
contribution gets accepted.

## The contribution test

The paper must survive this, stated in **one** sentence, before any polishing effort:

> We show that **[specific claim]**, evidenced by **[specific experiment]**, which was not
> established by **[specific prior work]**.

For this paper the shape is roughly: STGNN fault localization on the IEEE 123-bus feeder
degrades in a characterized way under sensor dropout — a regime the Karabulut et al.
benchmark does not evaluate.

If that sentence cannot be filled in with specifics, **stop and tell the author.** No
amount of writing quality fixes an unstated contribution. Do not paper over it with
stronger verbs.

## Desk-reject and weak-review causes — check each

Run this as a checklist against the full draft and report pass/fail per line:

- [ ] Contribution stated explicitly in the abstract and the introduction, not implied.
- [ ] Related work explains **what gap remains**, not just what exists. Every cited paper
      earns its sentence.
- [ ] The baseline is reproduced faithfully and described well enough to confirm that. A
      baseline that underperforms its published numbers is a red flag reviewers hunt for —
      if ours differs, explain why.
- [ ] Experimental setup is reproducible: hyperparameters, splits, seeds, hardware,
      software versions.
- [ ] Results report **variance**, not single runs. Seeds stated, error bands defined.
- [ ] Dropout is defined precisely — which sensors, what rate, random or structured,
      applied at training or inference or both. Ambiguity here sinks the whole paper.
- [ ] Ablations isolate one variable each.
- [ ] Negative and unfavorable results are reported, not omitted.
- [ ] Limitations section exists and names real limitations, not humble-brags.
- [ ] Every figure and table is referenced and discussed in prose.
- [ ] Abstract, intro, results, and conclusion state the **same** numbers.

## Adversarial review pass

Before declaring the draft ready, write the **three harshest objections** a reviewer could
raise, in their voice, and check whether the paper answers each in the text.

Typical ones for this paper — verify these specifically:

1. *"Sensor dropout robustness is an incremental variation on an existing benchmark. What
   is novel beyond running the same model with missing inputs?"*
2. *"Is the degradation attributable to dropout, or to distribution shift between training
   and evaluation conditions? The two are confounded as set up."*
3. *"A simple imputation baseline plus the original model may match this. Was that
   compared?"*

Report each as **answered in Section X** / **partially answered** / **not answered**. An
unanswered objection is a finding to hand the author, not something to argue away in
prose.

## Precision beats polish

When a sentence is weak, the fix is almost always a specific fact, not a better adjective.

| Weak | Strong |
|---|---|
| significantly improves robustness | retains 0.87 F1 at 30% dropout, vs. 0.61 for the baseline |
| a large-scale evaluation | 1,500 scenarios across four fault classes |
| the model is robust | performance degrades monotonically, with a knee at ~40% dropout |

If the specific number is unavailable, flag it per `00-core.md`. Do not substitute an
adjective for a missing measurement — that is exactly the pattern reviewers read as
hand-waving.

## Section-level bar

- **Abstract** — problem, gap, what we did, headline quantitative result, why it matters.
  The number goes in the abstract.
- **Introduction** — ends with an explicit contributions list. Each item is a thing the
  paper demonstrates, not a thing it discusses.
- **Related work** — organized by theme, closing on the gap this paper fills.
- **Method** — reproducible from the text alone.
- **Results** — every claim traceable to a table or figure.
- **Conclusion** — no new claims, no result that does not appear in Results.

## Readiness report

```
Contribution sentence:  <filled in, or FAILS>
Checklist:              <n> pass, <n> fail — <failures listed>
Adversarial objections: <answered / partial / unanswered, each>
Blocking issues:        <items requiring author decisions>
```

Do not describe a draft as ready while any checklist item fails or any objection is
unanswered.