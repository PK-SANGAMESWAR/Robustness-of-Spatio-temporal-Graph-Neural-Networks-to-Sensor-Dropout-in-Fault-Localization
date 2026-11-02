---
document_id: note_27_retrieval_scores
title: What a retrieval score means
document_type: technical_note
status: reviewed
approval_status: curated_for_research
version: 1
source_ref: implementation-plan.md section 4; implementation-plan.md section 10.5
verification: author review O34 2026-11-02; source_ref names project files
merged_from:
- B:note_27_retrieval_scores
- A:N30
created: '2026-10-31'
created_by: agent:claude-sonnet-5-5 (B) + agent:claude-opus-5-5 (A, merge)
reviewed: '2026-11-02'
review: accepted by the author in the O34 decision (2026-11-02)
---
# What a retrieval score means

A similarity score between a question and a passage is not a probability that the passage is relevant or correct, and a score of 0.8 does not mean 80 percent relevance. If no passage passes the sufficiency check the system must say that the evidence is insufficient and must not fill the gap with general knowledge.

It does not invent a procedure, a document section or a document revision.
