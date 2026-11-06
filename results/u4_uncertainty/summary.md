## U4 · Uncertainty: ensembles and conformal prediction

_Profile: **quick** · generated 2026-11-06 22:30_

Deep ensembles on f2 (trained on [-2, 2], evaluated on [-3, 3]):

| members | RMSE inside | RMSE outside | mean sd inside | mean sd outside | outside: abs error ≤ 2 sd |
|---|---|---|---|---|---|
| 5 × tanh | 0.0122 | 2.82 | 6.67e-03 | 0.148 | 0 % |
| 5 × relu | 0.0133 | 2.6 | 0.0131 | 0.458 | 11 % |

Split conformal on f1 with heteroscedastic noise (mean over 200 splits):

| nominal | plain coverage | plain width | weighted coverage | weighted width |
|---|---|---|---|---|
| 0.50 | 0.503 | 0.0873 | 0.503 | 0.123 |
| 0.70 | 0.702 | 0.162 | 0.702 | 0.185 |
| 0.80 | 0.802 | 0.23 | 0.802 | 0.226 |
| 0.90 | 0.902 | 0.366 | 0.902 | 0.288 |
| 0.95 | 0.952 | 0.489 | 0.951 | 0.346 |

Coverage at nominal 0.90 in the lowest / highest-noise bin: plain 1.000 / 0.611, weighted 0.956 / 0.883.
