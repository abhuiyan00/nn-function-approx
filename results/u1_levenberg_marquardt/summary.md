## U1 · Levenberg-Marquardt (second-order training)

_Profile: **quick** · generated 2026-11-06 22:24_

| target | method | median RMSE | best | worst | median seconds |
|---|---|---|---|---|---|
| f4 | Adam | 2.01e-04 | 6.50e-05 | 5.17e-04 | 1.6 |
| f4 | LM | 2.48e-06 | 9.08e-07 | 4.28e-06 | 3.5 |
| f4 | Adam → LM | 2.00e-06 | 4.81e-07 | 1.73e-05 | 3.8 |
| f5 | Adam | 7.73e-03 | 6.41e-03 | 9.32e-03 | 2.2 |
| f5 | LM | 1.44e-04 | 1.10e-04 | 3.39e-04 | 3.9 |
| f5 | Adam → LM | 1.27e-04 | 1.02e-04 | 1.72e-04 | 4.2 |

Median Adam/LM error ratio: f4 ×81, f5 ×54 (same network, same data; 6 seeds).
