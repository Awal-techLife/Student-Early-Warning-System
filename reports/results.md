| Version | Model | CV recall | CV precision | CV ROC-AUC | Test recall | Test precision | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| A: start of year | Dummy (always 'pass') | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| A: start of year | Rule: failures >= 1 | 0.53 ± 0.16 | 0.51 ± 0.15 | 0.71 ± 0.10 | 0.35 | 0.44 | 0.39 | 0.63 |
| A: start of year | Logistic Regression | 0.80 ± 0.11 | 0.32 ± 0.04 | 0.79 ± 0.08 | 0.65 | 0.30 | 0.41 | 0.76 |
| A: start of year | Random Forest | 0.80 ± 0.06 | 0.37 ± 0.04 | 0.84 ± 0.06 | 0.80 | 0.36 | 0.50 | 0.81 |
| A: start of year | Gradient Boosting | 0.80 ± 0.05 | 0.33 ± 0.04 | 0.80 ± 0.05 | 0.75 | 0.28 | 0.41 | 0.78 |
| A + absences (proxy) | Dummy (always 'pass') | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| A + absences (proxy) | Rule: failures >= 1 | 0.53 ± 0.16 | 0.51 ± 0.15 | 0.71 ± 0.10 | 0.35 | 0.44 | 0.39 | 0.63 |
| A + absences (proxy) | Logistic Regression | 0.80 ± 0.11 | 0.33 ± 0.05 | 0.79 ± 0.08 | 0.65 | 0.30 | 0.41 | 0.76 |
| A + absences (proxy) | Random Forest | 0.80 ± 0.07 | 0.36 ± 0.05 | 0.84 ± 0.06 | 0.85 | 0.35 | 0.50 | 0.81 |
| A + absences (proxy) | Gradient Boosting | 0.80 ± 0.05 | 0.34 ± 0.02 | 0.81 ± 0.05 | 0.70 | 0.29 | 0.41 | 0.77 |
| B: after first test | Dummy (always 'pass') | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| B: after first test | Rule: failures >= 1 | 0.53 ± 0.16 | 0.51 ± 0.15 | 0.71 ± 0.10 | 0.35 | 0.44 | 0.39 | 0.63 |
| B: after first test | Rule: G1 < 10 | 0.89 ± 0.11 | 0.61 ± 0.08 | 0.89 ± 0.06 | 0.85 | 0.45 | 0.59 | 0.83 |
| B: after first test | Logistic Regression | 0.81 ± 0.10 | 0.56 ± 0.12 | 0.93 ± 0.03 | 0.85 | 0.52 | 0.64 | 0.90 |
| B: after first test | Random Forest | 0.80 ± 0.11 | 0.65 ± 0.11 | 0.94 ± 0.02 | 0.75 | 0.52 | 0.61 | 0.89 |
| B: after first test | Gradient Boosting | 0.85 ± 0.13 | 0.60 ± 0.09 | 0.94 ± 0.03 | 0.90 | 0.46 | 0.61 | 0.92 |