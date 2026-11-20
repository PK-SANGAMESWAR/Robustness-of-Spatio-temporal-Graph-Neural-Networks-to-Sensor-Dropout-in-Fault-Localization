# Layer 4: location and uncertainty correctness (deterministic, with ground truth)

Written for the author. Source: the rescored S4 live items (20261120T044827Z_s4_main_live_rescored.jsonl, 20261120T044828Z_s4_noise_live_rescored.jsonl); 60 reports from 30 distinct events (the noise events are a subset of the main events). Truth comes from the golden events. Draft events. Heuristic doubt regex: `\b(disagree\w*|does not (?:support|agree)|do not (?:support|agree)|contradict\w*|conflict\w*|mismatch\w*|inconsisten\w*|but all|differ\w* from|other than the predicted)\b`. No pooled score: the S4 events over-sample wrong predictions.

## Location correctness: the report names the predicted bus

| stratum | reports | distinct events | names the predicted bus |
|---|---|---|---|
| high_conf_error | 16 | 10 | 1.00 (n=16) |
| confident_correct | 22 | 10 | 1.00 (n=22) |
| low_confidence | 22 | 10 | 1.00 (n=22) |

Repeats of one event are not independent: the rates below are over reports, the distinct-event counts show how few events stand behind them.

## Wrong predictions: does the report state doubt, and was any signal against the prediction available?

| stratum | wrong n | doubt stated | by gate | by insufficient evidence | by text | **no signal against the wrong prediction** |
|---|---|---|---|---|---|---|
| high_conf_error | 16 | 0.81 (n=16) | 0.00 (n=16) | 0.81 (n=16) | 0.12 (n=16) | 0.56 (n=16) |
| confident_correct | 0 | – | – | – | – | – |
| low_confidence | 16 | 1.00 (n=16) | 1.00 (n=16) | 0.31 (n=16) | 0.00 (n=16) | 0.00 (n=16) |

'No signal' = gate CONFIDENT and all 10 neighbours have the predicted (wrong) bus. No generator can catch such an error from this evidence.
'By insufficient evidence' follows the retrieved passages, not the risk (E27, E28): read it next to the right-prediction table below before you take it as a risk-aware doubt.

## Right predictions: doubt without a cause

| stratum | right n | doubt stated | doubt without a cause |
|---|---|---|---|
| high_conf_error | 0 | – | – |
| confident_correct | 22 | 0.68 (n=22) | 0.68 (n=22) |
| low_confidence | 6 | 1.00 (n=6) | 0.00 (n=6) |
