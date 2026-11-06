## E1 · Architecture (depth × width)

_Profile: **quick** · generated 2026-11-06 22:14_

Mean test RMSE over seeds (columns: width per hidden layer).

| target | depth | w=4 | w=8 | w=16 | w=32 | w=64 |
|---|---|---|---|---|---|---|
| f1 | 1 | 0.467 | 0.279 | 0.267 | 0.204 | 0.255 |
| f1 | 2 | 0.0381 | 0.0243 | 5.78e-03 | 1.98e-03 | 2.80e-03 |
| f1 | 3 | 0.0164 | 3.22e-03 | 2.71e-03 | 2.86e-03 | 4.32e-03 |
| f4 | 1 | 6.27e-03 | 2.10e-03 | 1.07e-03 | 6.87e-04 | 1.01e-03 |
| f4 | 2 | 9.99e-04 | 5.90e-04 | 2.91e-04 | 1.67e-04 | 1.25e-04 |
| f4 | 3 | 6.28e-04 | 3.72e-04 | 1.61e-04 | 2.59e-04 | 2.12e-04 |
| f5 | 1 | 0.0659 | 0.0425 | 0.0345 | 0.0313 | 0.0344 |
| f5 | 2 | 0.0348 | 0.0147 | 7.83e-03 | 6.57e-03 | 5.68e-03 |
| f5 | 3 | 0.032 | 8.90e-03 | 3.07e-03 | 1.58e-03 | 2.07e-03 |

Same parameter count, different shape:

- f1: 1×32 (97 params) RMSE 0.204 vs 2×8 (97 params) 0.0243 → ratio 8.4
- f4: 1×32 (97 params) RMSE 6.87e-04 vs 2×8 (97 params) 5.90e-04 → ratio 1.2

Best Franke network: [32, 32, 32] (2241 params), RMSE 1.36e-03.
