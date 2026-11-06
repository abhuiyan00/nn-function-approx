## U2 · KAN vs MLP

_Profile: **quick** · generated 2026-11-06 22:27_

| target | model | best RMSE | at params | best with 100-250 params | median s/run |
|---|---|---|---|---|---|
| f7 | MLP 2×H tanh | 2.11e-04 | 105 | 2.11e-04 @ 105 | 0.8 |
| f7 | KAN [d,5,1] | 1.34e-04 | 216 | 1.34e-04 @ 216 | 5.9 |
| f7 | KAN [2,1,1] | 3.83e-05 | 74 | - | 1.1 |
| f5 | MLP 2×H tanh | 1.39e-04 | 337 | 1.70e-04 @ 205 | 0.8 |
| f5 | KAN [d,5,1] | 2.17e-04 | 216 | 2.17e-04 @ 216 | 5.8 |
| f4 | MLP 2×H tanh | 7.08e-07 | 193 | 7.08e-07 @ 193 | 0.7 |
| f4 | KAN [d,5,1] | 4.65e-07 | 246 | 4.65e-07 @ 246 | 5.4 |
| f6 | MLP 2×H tanh | 1.13e-04 | 193 | 1.13e-04 @ 193 | 0.7 |
| f6 | KAN [d,5,1] | 5.03e-05 | 246 | 5.03e-05 @ 246 | 4.7 |

Symbolic check of KAN [2,1,1] on f7 (affine fit R² of each learned edge):

| edge | candidate | 1 − R² |
|---|---|---|
| inner x | sin(πx) | 6.2e-08 |
| inner y | y² | 4.3e-09 |
| outer | exp(s/0.60) | 1.8e-07 |

Grid extension run: final validation RMSE KAN 4.96e-05 (366 params) vs MLP 2×12 5.12e-04 (205 params).
