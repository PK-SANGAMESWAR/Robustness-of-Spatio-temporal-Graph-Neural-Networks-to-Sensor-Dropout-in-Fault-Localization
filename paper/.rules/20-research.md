# 20 — Web Search and Citation Verification

## When search is mandatory

Search — do not answer from memory — for:

- Any reference before it is added to `.bib` or cited in prose.
- Any claim about what prior work did, achieved, or reported.
- Any "recent," "state of the art," "to the best of our knowledge," or "first to" claim.
- Any venue detail: ICMLDE dates, page limits, Procedia CS formatting requirements.
- Any dataset or benchmark description (IEEE 123-bus specs, Karabulut et al. setup).
- Anything you half-remember. Partial recognition is not knowledge.

Your training data has a cutoff. The literature does not.

## When search is not needed

Textbook material — what a GCN layer computes, what an LSTM gate does, the definition of
F1. Just write it correctly.

## Verification protocol per reference

For every entry that goes into the `.bib`:

1. Search for the exact title.
2. Open the authoritative source — publisher page, IEEE Xplore, ACM DL, arXiv abstract
   page. **Not** Google Scholar snippets, ResearchGate mirrors, or citation aggregators.
3. Confirm from that page: full author list, exact title, venue, year, volume, pages, DOI.
4. Record the URL you verified against.

If step 2 fails to find the paper, the reference is **suspect**. Flag it. Do not add it
with guessed fields. A plausible-looking BibTeX entry for a paper that does not exist is
the single most damaging thing you can add to this repository.

Apply this to every existing entry too when asked to audit the bibliography (~22 entries).

## Citing content, not just existence

Confirming a paper exists is not the same as confirming it says what you claim. Before
writing "Zhang et al. [7] showed that X," read the abstract at minimum. If the specific
result is in the body and you cannot access it, write the weaker claim you can support and
flag it.

## Copyright

Paraphrase. Never copy sentences from a source into the manuscript. If exact wording is
genuinely necessary, use quotation marks, keep it under 15 words, and attribute it
inline. One quote per source, maximum. Reproducing another paper's phrasing without
quotation marks is plagiarism regardless of whether the citation is present.

## Reporting research

For each search-backed claim, give a one-line audit trail:

```
Claim: STGNNs outperform static GNNs on feeder fault localization.
Source: <verified URL>
Confidence: verified in abstract / verified in full text / inferred — flagged
```