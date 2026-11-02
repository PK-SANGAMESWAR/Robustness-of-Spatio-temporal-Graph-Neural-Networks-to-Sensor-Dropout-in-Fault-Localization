# Near-miss queries checked against the corpus (O39, 2026-11-02)

Written for: the author. Checked by `grep` on the 37 corpus documents (`corpus/feeder_v1`, `corpus/notes_v1`, `corpus/sop_v1`, `corpus/v1`), no model run, no API call. The queries stay `draft`.

## The three queries you named
| query | where I looked | what the corpus holds | verdict |
|---|---|---|---|
| R-0059 "How many windows of fault bus 57 are in the test split?" | `feeder_overview` § Scenarios, § Splits | Only totals: 825 scenarios, 50 runs per scenario, test 6,175 runs and 247,000 windows. No count per fault bus or per split and bus. A reader could only estimate it | unanswerable from a document; **a model that computes an estimate from the totals is a possible wrong answer. I keep the query as a trap for that, and you decide** |
| R-0064 "What was the validation macro-F1 of the dropout-aware model at its best epoch?" | `RESULTS.md`, `1170_revised.tex` | `RESULTS.md` line 18 gives the best-validation epoch (27.7–28.7 of 30) and no validation F1 value. The manuscript says that the best validation graph macro-F1 selects the checkpoint (line 392) and gives the validation size (line 254), but no value | unanswerable; a good near-miss (epoch and selection rule are present, the value is not) |
| R-0067 "How long can a μPMU keep its time reference after the clock signal is lost?" | `note_11`, `note_07` | `note_11` names "loss of its time reference" as one cause of dropout. `note_07` says the μPMU uses a common time reference. Neither gives a duration. The words holdover, GPS, clock (as a signal) and synchroni(s/z)ation do not occur in the notes or in RESULTS.md | unanswerable; a good near-miss |

## R-0060 (rephrased)
Old text: "Which sensors were dark in the worst-case run of the targeted study?" It had two meanings: "worst-case" is the heading of RESULTS.md § 5 (the targeted rule), but "run" could be one simulated event or one experiment run, and "dark" does not occur in the corpus.
New text: **"Which five μPMUs does the targeted study remove when the fault is at bus 57?"** One meaning: the 5 sensors that the targeted rule removes for one fault bus.
The corpus gives the rule (note_16, RESULTS.md § 5) and the 3 nearest μPMUs of bus 57 (buses 54, 60, 13 in `feeder_fault_buses`). It does not give a 4th or 5th. The absence patterns (`4th nearest`, `5th nearest`, `fifth nearest`, `worst-case run`, `dark in the worst`) find nothing. The item stays `draft`.

## What changed in the golden set (v3)
- R-0055 and R-0056: spans accepted, status `reviewed`. Reviewed queries: 56 (dev 25 answerable + 4 unanswerable, test 23 + 4).
- The frozen test runs and all sweep numbers use the 54 reviewed queries of v2 (R-0055 was in the test run only as a draft and is not counted). I did not rerun anything. A future run of the frozen configuration would see 56 queries and needs your approval (test split: one use per configuration).
- Script: `data/eval/apply_o39_golden.py`. Test: `test_golden_retrieval_v3_is_valid_and_statuses_follow_the_author`, `test_near_miss_unanswerable_queries_stay_unanswerable_by_the_corpus`.

## Left for you (O39)
Accept, edit or reject the 12 near-miss queries R-0057 to R-0068. Say if R-0059 should stay (see the table).

## Decision (author, 2026-11-02)
R-0059 stays as a trap: the correct output is "insufficient evidence" (`expected_output`); R-0060 wording accepted. Both are `reviewed` (golden v4). The other 10 near-miss queries stay draft.
