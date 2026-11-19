# T11 — model mismatch (perturbed signature table at inference), validation split

| table | R5 > R0ⁿ (of 21) | graceful (rule) | k10_sigma0.0: R5 table / R5 nominal / R0ⁿ | k0_sigma0.01: R5 table / R5 nominal / R0ⁿ | retrain / nominal weights on ±5 % / nominal on nominal / R0ⁿ |
|---|---|---|---|---|---|
| tap +1 step (all regulators) | 15/21 | no | 0.676 / 0.857 / 0.655 | 0.748 / 0.748 / 0.724 |  |
| tap −1 step (all regulators) | 18/21 | no | 0.710 / 0.857 / 0.655 | 0.748 / 0.748 / 0.724 |  |
| line impedance ±5 % (per line) | 21/21 | yes | 0.837 / 0.857 / 0.655 | 0.747 / 0.748 / 0.724 |  |
| retrain on ±5 % table, seed 42, k0_sigma0.0 | 1 seed: no rule | — |  |  | 0.894 / 0.888 / 0.895 / 0.846 |
| retrain on ±5 % table, seed 42, k10_sigma0.0 | 1 seed: no rule | — |  |  | 0.839 / 0.837 / 0.856 / 0.669 |

- Rule (14, Amendment 6): graceful if R5 with the perturbed table > R0ⁿ in all 3 seeds in the 7 H2/H4 cells (21 comparisons). R0ⁿ does not read the table.
- The retrain row is 1 seed (42): descriptive only.
- Source: results/stage_g/analysis/summary.json (scripts/summarize_stage_g.py).
