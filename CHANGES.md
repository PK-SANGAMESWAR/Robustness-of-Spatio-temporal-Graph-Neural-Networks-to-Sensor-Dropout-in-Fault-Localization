## 1. No statistical significance test (Major)

You now report mean ± std, which is much better.

However, nowhere do you show

paired t-test
Wilcoxon test
p-values
confidence intervals

A reviewer could ask:

"Are these improvements statistically significant?"

This is probably the biggest remaining experimental weakness.

## 2. No runtime analysis (Major)

You repeatedly state

"no inference-time cost"

because only training changes.

But you never measure

inference latency
training overhead
GPU memory
throughput

A simple table would make this claim much stronger.

## 10. Missing implementation details

The paper still doesn't state things like

total training time
convergence speed
epochs until convergence

Not essential, but reviewers sometimes ask.

## 11. Slight overstatement in the abstract

The abstract says

"Clean accuracy predicts robustness poorly."

Strictly speaking,

you evaluated only five architectures on one benchmark.

Safer wording would be

"Within the evaluated architectures..."

Not a major issue.

Severity: ★☆☆☆☆

## Add a statistical significance analysis (or explain why it isn't included).

## Add a small runtime/efficiency table to support the "no inference-time cost" claim.

## Add one more paragraph discussing why dropout-aware training works, beyond reporting that it works.