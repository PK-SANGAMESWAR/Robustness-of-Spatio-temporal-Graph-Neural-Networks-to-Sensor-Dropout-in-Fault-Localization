---
prompt: evidence_check
version: 0
status: draft
model: gpt-6-luna
source: author decision 2026-11-02 (the evidence decision uses tau as one input and the generator output "insufficient evidence" as another)
---
## System
You check whether a set of passages contains what is needed to answer a question. You do not answer the question. Return only JSON: {{"evidence_status": "sufficient" or "insufficient", "reason": "one short sentence"}}.
Use "insufficient" if the passages are about the topic but do not contain the specific fact, number or procedure that the question asks for.

## User
QUESTION
{question}

PASSAGES
{passages}
