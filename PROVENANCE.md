# PROVENANCE.md — AI provenance ledger for the manuscript

Per `paper/.rules/60-integrity.md` Part A. Every session that touches
`paper/overleaf.tex` (or other manuscript assets) appends here. Append only;
never overwrite. AI provenance is disclosed, not disguised. The authors remain
fully responsible for every claim.

Classes: AI-DRAFTED | AI-EDITED | AI-SUGGESTED-AUTHOR-ACCEPTED | AI-VERIFIED-ONLY

---

Date:        2026-08-12
File:        paper/overleaf.tex — retroactive note for earlier 2026-08-07..12 sessions
Class:       AI-EDITED / AI-DRAFTED (see Scope)
Scope:       This ledger was created on 2026-08-12. Earlier edits in this project
             were made before the ledger existed and are recorded here in summary
             (honest retroactive disclosure, not a substitute for per-edit logging
             going forward):
               - Reframing away from "reproduction" toward "adopt methodology,
                 report our own numbers" (abstract, intro, methods, renamed
                 "Reproduction Notes" -> "Differences from the Base Paper's Setup",
                 limitations, conclusion). AI-EDITED on top of prior AI-DRAFTED text.
               - Fix of a self-contradicting sentence in the dropout-aware results
                 paragraph; full grammar/typo proofreading pass. AI-EDITED.
               - Table 4 / Fig. 4 degradation numbers updated to 3-seed values;
                 seed-42 caveats removed; discussion ranking narrative revised.
                 AI-EDITED; numbers come from the experimental pipeline, not the model.
               - Acknowledgements + Generative-AI disclosure statements drafted
                 (AI-DRAFTED, marked AUTHOR ACTION for author confirmation).
Prior class: Most manuscript prose was AI-DRAFTED in earlier sessions (per PROGRESS.md).
Author input: Author directed each change (reframing, proofread, fold-in of results,
             this revision pass) but did not hand-write the replacement prose.

Date:        2026-08-12
File:        paper/overleaf.tex — full paper-issue.md revision pass
Class:       AI-EDITED / AI-DRAFTED (see Scope)
Scope:       Executed the paper/paper-issue.md checklist under author's blanket
             authorization ("make edits and inform after everything finishes"):
             abstract dedup + clean-accuracy number + acronym fix; introduction
             trims + roadmap collapse; related-work interpretive paragraph;
             methods condensation (baselines, voltage-sag, Table 2 subtable, AUDC
             sentence placement, eqref); setup version trim + duplicate-paragraph
             removal; folded "Differences from Base Paper" section into a compact
             table; results interpretive rewrite + Table 4 demotion + RGATv2
             future-work sentence + per-resistance robustness statement; discussion
             restructure + deployment guideline; conclusion citation fix; reference
             cite/keep decision. Added a NEW architecture-agnostic result
             (dropout-aware across all 5 architectures) as a table, numbers from the
             experimental pipeline (results/dropout_study_aware/degradation_curves.csv).
Prior class: AI-DRAFTED (prior sessions).
Author input: Author authorized the full checklist pass and the numbers to report;
             the model performed the edits. Data/results are pipeline-generated, never
             model-generated (per 60-integrity.md Part C). No detector-evasion edits
             were made or suggested.
Author-only items NOT done by the model (stated, not performed):
             - iThenticate/Turnitin similarity check (author's step; draft is not
               submission-ready until done).
             - Final manual novelty-claim literature check (web check done; author confirms).
             - Corresponding-author contact details; funding/AI-disclosure final wording.
