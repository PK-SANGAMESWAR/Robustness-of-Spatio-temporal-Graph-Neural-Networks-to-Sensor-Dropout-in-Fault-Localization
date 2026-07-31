# Reproduction log

## Window and label convention

Paper evidence: BASE §4.2 states that a 60 ms stream, beginning 40 ms before fault onset, produces 40 overlapping windows of length 20 ms at a 1 ms sampling resolution. It also states that half are fault-free and half capture the fault. It does **not** say which endpoint is omitted or give window start indices.

Ambiguity: ordinary inclusive sliding-window arithmetic yields `(60 - 20) / 1 + 1 = 41` valid starts, `0..40`. The reported 40 is therefore not derivable from the text alone.

Adopted protocol: preserve the paper's reported count and exact class balance by retaining starts `0..39` and dropping the valid endpoint at start `40`. Starts `0..19` are wholly pre-fault and receive class 0; starts `20..39` include at least one post-onset sample and receive the fault-bus class. This produces 20 healthy and 20 fault windows per event. The endpoint is deliberately not labelled or stored.

Verification: `extract_windows` and `count_windows` enforce this special 60/20/1 protocol; strict validation checks 40 windows and a 20/20 balance for every event.

Status: Source ambiguity documented; reproduction convention fixed.

## Corpus-size reconciliation

Paper evidence: BASE §4.2 states 25 locations, 11 fault types, three resistances, 100 simulations per fault case, 40 windows per event, and a total of 2.5 million windows.

Ambiguity: interpreting a scenario as the stated `(location, type, resistance)` tuple gives `25 × 11 × 3 = 825` scenarios. The stated 100 runs then yields `825 × 100 × 40 = 3,300,000` windows, not 2,500,000. Conversely, 2.5M windows at 40/event requires 62,500 events, or 75.76 runs/scenario, which cannot be an equal integer run count across all 825 scenarios.

Adopted protocol: generate the complete 825 × 100 corpus (82,500 events / 3,300,000 windows). `2.5M` is recorded as an unreconciled paper total, not used to truncate scenarios or runs.

## Training delivery order

Paper evidence: BASE §4.2 requires sliding windows from the same time step to stay together when assigning train/validation/test splits. It does not prescribe minibatch ordering within a split.

Adopted protocol: shuffle the fixed training events deterministically by seed, then interleave windows from an eight-event working set. Every selected training window appears exactly once per epoch; no window, label, split membership, loss, or hyperparameter changes. The event working set matches the dataset cache and avoids repeatedly decompressing the same event file, allowing GPU computation to remain the dominant runtime.

## Layer order

Paper states: BASE §4.3 lists batch normalization, ReLU, and dropout without an unambiguous order.

Implementation: batch normalization → ReLU → dropout → classifier in `NodeClassifier`.

Reason: a single shared order makes the baseline variants comparable.

Verification: shared module is used by every baseline.

Status: Assumed.

## F1 definition

Paper states: score averaging is not fully specified in the available plan.

Implementation: macro F1, labels 0–25, `zero_division=0`, saved in resolved configs and metrics.

Reason: this is predeclared, including absent pilot classes.

Verification: evaluator calls scikit-learn with a fixed label order.

Status: Assumed.
