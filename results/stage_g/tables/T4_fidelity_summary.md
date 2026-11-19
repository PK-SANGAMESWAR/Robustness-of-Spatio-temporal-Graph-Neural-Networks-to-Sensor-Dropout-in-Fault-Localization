# T4 Operator fidelity on the training events

| Subset | n events | median |e| (max channel, p.u.) or fraction | p95 (p.u.) | max (p.u.) |
|---|---|---|---|---|
| all | 28875 | 1.31e-03 | 6.27e-03 | 1.32e-02 |
| R = 0.1 Ω | 9634 | 8.84e-04 | 6.16e-03 | 8.34e-03 |
| R = 1 Ω | 9587 | 1.36e-03 | 6.20e-03 | 1.25e-02 |
| R = 10 Ω | 9654 | 2.38e-03 | 6.44e-03 | 1.32e-02 |
| AG | 2623 | 1.28e-03 | 6.42e-03 | 1.15e-02 |
| BG | 2619 | 1.80e-03 | 6.41e-03 | 8.34e-03 |
| CG | 2642 | 1.74e-03 | 6.43e-03 | 1.25e-02 |
| AB | 2658 | 1.48e-03 | 6.24e-03 | 1.21e-02 |
| BC | 2659 | 1.87e-03 | 6.16e-03 | 1.32e-02 |
| CA | 2570 | 1.14e-03 | 6.46e-03 | 7.38e-03 |
| ABG | 2620 | 1.13e-03 | 6.22e-03 | 1.20e-02 |
| BCG | 2648 | 1.55e-03 | 6.21e-03 | 7.51e-03 |
| CAG | 2624 | 1.22e-03 | 6.26e-03 | 1.22e-02 |
| ABC | 2600 | 9.05e-04 | 6.18e-03 | 1.19e-02 |
| ABCG | 2612 | 8.35e-04 | 6.19e-03 | 1.19e-02 |
| L* within 1 step | 28875 | 0.9725 |  |  |
| L* exact | 28875 | 0.8319 |  |  |
| τ* correct | 28875 | 0.9093 |  |  |
| R* within 1 step | 28875 | 0.9974 |  |  |

Source: results/pcstgnn/audit/s1_4_7/s1_4_7_summary.json. The |e| rows are the table-versus-corpus voltage gap; the audit note calls S1.4 a construction check, not an independent fidelity test. Fraction rows are in column 3.
