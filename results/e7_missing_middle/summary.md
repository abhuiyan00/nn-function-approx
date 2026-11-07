## E7 · The missing middle

_Profile: **quick** · generated 2026-11-07 12:52_

Hole of 30 % of each axis, medians over seeds:

| target | σ | closest in the hole | RMSE | closest network | RMSE in the hole | its RMSE with data | best on validation | its RMSE in the hole |
|---|---|---|---|---|---|---|---|---|
| m1 | 0 | Chebyshev LS | 9.07e-13 | MLP ReLU · Adam | 0.202 | 8.97e-03 | Chebyshev LS | 9.07e-13 |
| m1 | 0.05 | Chebyshev LS | 0.188 | MLP ReLU · Adam | 0.214 | 0.0169 | MLP tanh · LM | 0.604 |
| m2 | 0 | Chebyshev LS | 1.07e-03 | MLP tanh · LM | 0.121 | 4.90e-07 | Chebyshev LS | 1.07e-03 |
| m2 | 0.05 | Chebyshev LS | 0.0375 | MLP tanh · LM | 0.176 | 6.60e-03 | cubic spline LS | 3.54 |
| m3 | 0 | Chebyshev LS | 0.0114 | MLP tanh · LM | 0.032 | 8.75e-06 | Chebyshev LS | 0.0114 |
| m3 | 0.05 | MLP tanh · LM | 0.092 | MLP tanh · LM | 0.092 | 4.97e-03 | cubic spline LS | 3.61 |
| m4 | 0 | Chebyshev LS | 1.48e-16 | MLP tanh · LM | 9.95e-04 | 4.37e-05 | Chebyshev LS | 1.48e-16 |
| m4 | 0.05 | MLP tanh · LM | 0.0455 | MLP tanh · LM | 0.0455 | 5.93e-03 | cubic spline LS | 3.6 |
| m5 | 0 | KAN · LM | 1.20e-04 | KAN · LM | 1.20e-04 | 1.82e-04 | KAN · LM | 1.20e-04 |
| m5 | 0.05 | MLP tanh · LM | 0.0243 | MLP tanh · LM | 0.0243 | 0.0126 | MLP tanh · LM | 0.0243 |

RMSE in the hole as it widens:

| target | σ | closest network at 30 % | hole 10 % | hole 20 % | hole 30 % | hole 40 % |
|---|---|---|---|---|---|---|
| m1 | 0 | MLP ReLU · Adam | 0.0131 | 0.0876 | 0.202 | 0.507 |
| m1 | 0.05 | MLP ReLU · Adam | 0.0211 | 0.0947 | 0.214 | 0.507 |
| m2 | 0 | MLP tanh · LM | 2.35e-04 | 0.0129 | 0.121 | 0.234 |
| m2 | 0.05 | MLP tanh · LM | 0.0362 | 0.126 | 0.176 | 0.341 |
| m3 | 0 | MLP tanh · LM | 0.0139 | 0.0255 | 0.032 | 0.0427 |
| m3 | 0.05 | MLP tanh · LM | 0.0329 | 0.0691 | 0.092 | 0.0364 |
| m4 | 0 | MLP tanh · LM | 3.07e-05 | 2.41e-04 | 9.95e-04 | 6.82e-03 |
| m4 | 0.05 | MLP tanh · LM | 2.51e-03 | 0.0231 | 0.0455 | 0.0665 |
| m5 | 0 | KAN · LM | 1.31e-04 | 1.34e-04 | 1.20e-04 | 4.24e-04 |
| m5 | 0.05 | MLP tanh · LM | 6.86e-03 | 8.98e-03 | 0.0243 | 0.011 |
