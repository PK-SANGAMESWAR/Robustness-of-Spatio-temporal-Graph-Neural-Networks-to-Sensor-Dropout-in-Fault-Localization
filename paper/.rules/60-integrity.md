# 60 — Research Integrity: Plagiarism and AI Provenance

This file governs two distinct risks. Do not conflate them.

- **Content plagiarism** — text, structure, or ideas taken from a source without proper
  attribution. This is misconduct. It is grounds for retraction and institutional action.
- **AI provenance** — which parts of this manuscript were drafted or edited by a model.
  This is not misconduct. **Undisclosed** AI authorship is.

The remedy for the first is not copying. The remedy for the second is disclosure.
Neither remedy is disguise.

---

## Part A — Provenance: report everything

### The ledger

Maintain `PROVENANCE.md` at the repo root. Every session that touches the manuscript
appends to it. Nothing is exempt — not a one-word fix, not a citation reorder, not a
caption tweak.

Per entry:

```
Date:        <ISO date>
File:        <path> — <section>
Class:       AI-DRAFTED | AI-EDITED | AI-SUGGESTED-AUTHOR-ACCEPTED | AI-VERIFIED-ONLY
Scope:       <what specifically changed>
Prior class: <what this text's class was before this edit>
Author input: <what the author specified vs. what the model chose>
```

### Provenance does not decay

**Re-editing text does not reclassify it.** If a passage was AI-drafted and is later
AI-edited, it remains AI-drafted with an additional edit logged. It does not become
author-written. It does not become "revised." Passing text through more edits to make its
origin ambiguous is laundering, and the ledger exists specifically to prevent it.

The cumulative history of a passage is the record. Never overwrite an entry; only append.

### Never silently touch author prose

If the author wrote it, changing it requires:

1. Showing the original and the proposed replacement as a diff.
2. One line of rationale.
3. The author's explicit acceptance before it lands.

Text the author wrote and then accepted a suggestion on is
`AI-SUGGESTED-AUTHOR-ACCEPTED`, not `AI-DRAFTED`. That distinction matters and must be
recorded accurately.

### Per-session report

Every response that touches the manuscript ends with:

```
Provenance this session:
  AI-drafted:   <sections>
  AI-edited:    <sections, and their prior class>
  Untouched:    <sections in scope that were not modified>
Ledger updated: yes / no
```

If you did not update the ledger, say so explicitly. A missing entry is a defect.

---

## Part B — Content plagiarism: hard prohibitions

### Never copy

No sentence, clause, or distinctive phrase moves from a source into this manuscript.
Not from papers, not from web pages, not from documentation, not from an LLM's recollection
of a paper's abstract.

### Paraphrase means rewrite

These all count as plagiarism even with a citation attached:

- **Patchwriting** — the source's sentence with synonyms substituted.
- **Structural copying** — the source's sentence order, paragraph flow, or argument
  scaffolding reproduced with different words.
- **Definition lifting** — a source's phrasing of a definition reused as-is.

A real paraphrase is written from understanding after closing the source, then checked
against the original to confirm it shares meaning but not wording or shape.

### Citation is necessary but not sufficient

Citing a source does not license reproducing its language. Every borrowed **idea** needs a
citation; every borrowed **phrasing** needs quotation marks, and quotation should be rare.

Direct quotes: under 15 words, in quotation marks, attributed inline, **one per source
maximum**. If a passage needs more quoting than that, it needs rewriting instead.

### Self-plagiarism

Text reused from the author's own prior work — thesis, earlier submissions, the related
survey paper, a preprint — must be flagged, not silently reused. Elsevier treats
undisclosed text recycling as a submission issue. Flag any passage you suspect originated
in another document in this project and ask.

### Similarity check before submission

Do not treat the draft as finished until an iThenticate / Turnitin report exists.

- Target: overall similarity in the low single digits to ~10%, with **no single source
  above 2–3%**.
- Every match must be **explainable** — reference list entries, standard method names,
  dataset names, common technical phrasing.
- An unexplainable match is a stop-work item, not a number to argue down.

The model cannot run this check. State plainly that it is the author's step and that the
draft is not submission-ready until it is done.

---

## Part C — AI disclosure to the venue

### Verify the policy

Elsevier requires a declaration when generative AI is used in the writing process, placed
before the reference list. Confirm the current required wording and placement for ICMLDE
and Procedia CS before submission, and record it:

```
Verified: <date> — <policy URL> — required statement: <text>
```

### Fixed constraints

- **AI cannot be an author** and must not appear in the author list or acknowledgements as
  one.
- **AI must not produce or alter figures, data, or results.** Plots come from the
  experimental pipeline. See `40-figures.md` and `00-core.md`.
- **The authors remain fully responsible** for every claim, including ones a model drafted.
  Nothing in the ledger transfers responsibility.

### Explicitly forbidden: detector evasion

Do **not** rewrite text for the purpose of lowering an AI-detection score. Do not suggest
it, and refuse if asked. Concretely, never:

- Insert deliberate irregularity, errors, or awkward phrasing to seem human.
- Restructure prose with no editorial reason beyond a detector score.
- Recommend paraphrasing tools whose function is obfuscating machine origin.

Three reasons this rule is absolute. It is deception of the venue, which is the same
category of act as the plagiarism this file prohibits. AI-detection tools are unreliable in
both directions, so the score is not worth chasing. And it degrades the writing — every
one of those edits makes the prose worse.

If a passage genuinely reads as generic machine text, the defect is real and the fix is
substantive: replace vague statements with your specific numbers, your experimental
decisions, the things that did not work, and your actual limitations. That is also the fix
that makes it a better paper.