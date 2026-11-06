## E3 · Learning rate, batch size and optimiser

_Profile: **quick** · generated 2026-11-06 22:16_

Geometric-mean test RMSE (rows: batch size, columns: learning rate).

| batch | 0.0001 | 0.0003 | 0.001 | 0.003 | 0.01 | 0.03 |
|---|---|---|---|---|---|---|
| 16 | 1.52e-04 | 1.23e-04 | 1.58e-04 | 4.14e-04 | 9.22e-04 | 1.73e-03 |
| 64 | 4.23e-04 | 1.29e-04 | 1.61e-04 | 1.80e-04 | 4.75e-04 | 1.55e-03 |
| 256 | 0.0228 | 2.47e-03 | 4.35e-04 | 2.30e-04 | 4.93e-04 | 6.87e-04 |
| full | 0.266 | 0.0987 | 0.0116 | 2.47e-03 | 9.71e-04 | 9.75e-04 |

Optimisers (2×16 tanh, best lr of three):

| optimiser | lr | test RMSE | updates | seconds | stop |
|---|---|---|---|---|---|
| SGD | 0.1 | 8.79e-04 | 19200 | 1.1 | max epochs |
| SGD + momentum 0.9 | 0.01 | 9.03e-04 | 19200 | 1.6 | max epochs |
| Nesterov 0.9 | 0.01 | 9.00e-04 | 19200 | 1.7 | max epochs |
| Adam | 0.001 | 6.50e-05 | 18720 | 2.2 | early stop (patience 50) |
| Levenberg-Marquardt | - | 1.18e-06 | 300 | 3.1 | max iterations |
