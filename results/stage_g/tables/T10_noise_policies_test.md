# T10 — measurement noise and sensor-loss policies, test split, R5 vs R0ⁿ

| condition | k | R5 | R0ⁿ | ΔF1 | seeds > 0 | p (raw) |
|---|---|---|---|---|---|---|
| noise σ=0 | 0 | 0.894 | 0.833 | +0.061 | 3/3 | 0.0129 |
| noise σ=0.0005 | 0 | 0.893 | 0.832 | +0.061 | 3/3 | 0.0109 |
| noise σ=0.002 | 0 | 0.871 | 0.814 | +0.057 | 3/3 | 0.0155 |
| noise σ=0.01 | 0 | 0.747 | 0.720 | +0.028 | 3/3 | 0.1208 |
| noise σ=0 | 10 | 0.851 | 0.647 | +0.203 | 3/3 | 0.0011 |
| noise σ=0.0005 | 10 | 0.848 | 0.648 | +0.200 | 3/3 | 0.0012 |
| noise σ=0.002 | 10 | 0.806 | 0.637 | +0.169 | 3/3 | 0.0027 |
| noise σ=0.01 | 10 | 0.677 | 0.574 | +0.103 | 3/3 | 0.0109 |
| policy random | 1 | 0.889 | 0.819 | +0.070 | 3/3 | 0.0103 |
| policy random | 2 | 0.884 | 0.805 | +0.079 | 3/3 | 0.0079 |
| policy random | 5 | 0.871 | 0.757 | +0.114 | 3/3 | 0.0034 |
| policy random | 10 | 0.851 | 0.647 | +0.203 | 3/3 | 0.0011 |
| policy persistent | 1 | 0.888 | 0.819 | +0.069 | 3/3 | 0.0102 |
| policy persistent | 2 | 0.884 | 0.805 | +0.078 | 3/3 | 0.0080 |
| policy persistent | 5 | 0.870 | 0.759 | +0.111 | 3/3 | 0.0042 |
| policy persistent | 10 | 0.851 | 0.646 | +0.205 | 3/3 | 0.0011 |
| policy contiguous | 1 | 0.889 | 0.819 | +0.070 | 3/3 | 0.0100 |
| policy contiguous | 2 | 0.883 | 0.801 | +0.082 | 3/3 | 0.0067 |
| policy contiguous | 5 | 0.866 | 0.746 | +0.120 | 3/3 | 0.0055 |
| policy contiguous | 10 | 0.838 | 0.648 | +0.190 | 3/3 | 0.0014 |
| policy targeted | 1 | 0.855 | 0.714 | +0.141 | 3/3 | 0.0136 |
| policy targeted | 2 | 0.835 | 0.661 | +0.174 | 3/3 | 0.0066 |
| policy targeted | 5 | 0.795 | 0.548 | +0.246 | 3/3 | 0.0019 |
| policy targeted | 10 | 0.741 | 0.496 | +0.245 | 3/3 | 0.0032 |

- Mean over seeds 42/43/44; ΔF1 = R5 − R0ⁿ; p = paired two-sided t over seeds (df = 2), not corrected.
- Policies at σ = 0: random (S4.4 protocol), persistent (per event), contiguous (connected block), targeted (nearest the fault; no-fault windows fall back to random).
- n = 3: a sign in 3/3 seeds without p < 0.05 is a trend, not a result.
