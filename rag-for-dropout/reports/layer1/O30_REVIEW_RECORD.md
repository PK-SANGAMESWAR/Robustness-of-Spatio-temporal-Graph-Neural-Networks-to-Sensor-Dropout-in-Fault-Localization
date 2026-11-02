# O30 review record (2026-11-02)

Written for: the author. This file records how your O30 review was applied. Scripts: `corpus/notes_v1/_review_2026-11-02.py` and `data/eval/apply_o30_review.py` (both assert the facts below before they write).
Golden set: v0 (54 drafts, in Git `692ef94`) → **v1 (56 reviewed items)**. Notes: 19 reviewed, 11 still draft, SOP draft. Feeder documents: accepted, not edited.

## Queries (accept all except the listed ones)
| item | your instruction | verdict | what I did and what I checked |
|---|---|---|---|
| R-0005 | remove the results_md span if it does not show accuracy by fault resistance | **edited** | The span is a bare table row; the column labels (0.1 / 1 / 10 ohm) are in the header row and not in the span, so it does not show accuracy by resistance. Removed. The note_05 span stays |
| R-0036 | use a span that contains the 6 bus IDs, else reject | **accepted unchanged** | Both golden spans (feeder_overview, note_08) already contain 13, 18, 25, 35, 47, 97 (asserted) |
| R-0034 | accept or edit | **edited** | Span unchanged (the row is the only text with the phase count of bus 26). The reference answer was the bare row; now "Two phases: the μPMU at bus 26 (graph node 5) measures two phases" (checked in `pmu_phase_metadata.yaml`) |
| R-0044 | accept or edit | **edited** | Spans unchanged. The reference answer still held a "(ref)" parser artifact; now equal to the paper span |
| R-0047, R-0048 | accept or edit | **accepted unchanged** | "breaker" and "recloser" do not occur anywhere in the corpus (asserted) |
| R-0055 (new, test) | nearest μPMU to bus 51 on the operated feeder → bus 47 | added | Span: the row `\| 51 \| bus 47 (0.229 km, 3) \| bus 45 … \|` of `feeder_fault_buses` |
| R-0056 (new, dev) | what ABCG means → feeder_fault_taxonomy | added | Spans: "a trailing G means the fault includes ground … ABC is a three-phase fault without ground" (grade 3) and the bullet "three-phase: ABC, ABCG" (grade 2) |
The agent chose the golden spans of R-0055 and R-0056; the query, split, document and answer are yours. No query was changed or removed because of its rank.

## Notes
| notes | your instruction | applied |
|---|---|---|
| 01–07, 11 | `source_ref` = "general background, not checked against a specific page"; book to `further_reading` | done. Notes 05 and 07 keep their project facts in a new field `project_sources` (RESULTS.md § 1; constants.py) so that the project numbers stay sourced |
| 19–22 | write "checked" or apply the same rule | **both:** the project numbers were checked against the Layer 0 reports (listed in `project_sources`); the external papers were not checked against a page, so the same `further_reading` rule applies |
| 09, 25, 26, 28 | `source_ref` = the project design documents; remove external citations | done (plan sections and `gate_v1`; Phadke, Thorp removed). Note_09: the sentence "Ambiguity grows when sensors are missing" had no support in the plan; replaced by "Which candidates are indistinguishable depends on which sensors are available" (plan § 11.1) |
| 14, 15, 16 vs RESULTS.md | no note may call 0.846 the dropout-aware value | all numbers found in RESULTS.md (asserted). 0.846 appears in notes 14 and 17 only, both as the standard model (a test checks every note). **Note_16 had a real error:** "With one sensor lost the two models are not distinguishable" followed the dropout-aware-minus-standard comparison under the same targeting. In RESULTS.md § 5 the indistinguishable pair is dropout-aware under targeting against standard under random loss (+0.018, p = 0.33). Corrected |
| 07 | confirm 25 μPMUs against the dataset config | confirmed: `NUM_PMU_NODES` = 25 = `len(PMU_BUSES)`, `node_mapping.json` has 25 buses, an event file has windows of shape (25, 20, 6) with 3 voltage + 3 mask channels |
| (found while checking) note_20 | – | said the margin is "the only threshold used" at the 95 % point; `gate_v1.json` also has a probability threshold of 0.5424 at k = 2, and the margin threshold is not monotone (0.1459, 0.1459, 0.1368, 0.1633, 0.2518). Corrected to the exact values |
Not mentioned in your review, **left `draft`**: note_08, 10, 12, 13, 17, 18, 23, 24, 27, 29, 30 and the SOP (O34). Say "accept" and I set them to reviewed.

## Feeder documents (accepted; the files are generated and were not edited; sha256 in the manifest match)
`feeder_fault_buses` gives **bus 47 as the nearest μPMU to bus 51: 0.229 km, 3 hops**, with distances in km (kft × 0.3048; the normally open switches Sw7 and Sw8 open). Independent check from the raw DSS lines:
L48, L49 and L50 are 0.25 kft each, 0.75 kft = 0.229 km.
