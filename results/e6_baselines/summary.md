## E6 · Classical baselines at matched size

_Profile: **quick** · generated 2026-11-06 22:23_

| target | model | best RMSE | at size | best with 80-400 params |
|---|---|---|---|---|
| f1 | MLP 2×H · Adam | 1.54e-03 | 1153 | 3.10e-03 @ 321 |
| f1 | MLP 2×H · LM | 2.03e-05 | 321 | 2.03e-05 @ 321 |
| f1 | Chebyshev LS | 9.12e-16 | 33 | 1.66e-15 @ 97 |
| f1 | cubic spline LS | 6.87e-08 | 128 | 6.87e-08 @ 128 |
| f4 | MLP 2×H · Adam | 6.50e-05 | 321 | 6.50e-05 @ 321 |
| f4 | MLP 2×H · LM | 7.08e-07 | 193 | 7.08e-07 @ 193 |
| f4 | Chebyshev LS | 5.32e-09 | 97 | 5.32e-09 @ 97 |
| f4 | cubic spline LS | 2.01e-07 | 128 | 2.01e-07 @ 128 |
| f5 | MLP 2×H · Adam | 5.16e-03 | 1185 | 6.51e-03 @ 337 |
| f5 | MLP 2×H · LM | 1.10e-04 | 501 | 1.39e-04 @ 337 |
| f5 | Chebyshev LS | 1.31e-03 | 289 | 1.31e-03 @ 289 |
| f5 | cubic spline LS | 8.62e-05 | 400 | 8.62e-05 @ 400 |
| f6 | MLP 2×H · Adam | 1.09e-03 | 4353 | 2.16e-03 @ 97 |
| f6 | MLP 2×H · LM | 1.13e-04 | 193 | 1.13e-04 @ 193 |
| f6 | Chebyshev LS | 6.61e-04 | 97 | 6.61e-04 @ 97 |
| f6 | cubic spline LS | 2.19e-04 | 128 | 2.19e-04 @ 128 |

Runge function, interpolation:

| interpolant | coefficients | max error | RMSE |
|---|---|---|---|
| equispaced nodes, degree 10 | 11 | 1.92 | 0.58 |
| equispaced nodes, degree 20 | 21 | 59.8 | 11.9 |
| Chebyshev nodes, degree 20 | 21 | 0.0153 | 7.50e-03 |
