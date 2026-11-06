## E2 · Activation and initialisation

_Profile: **quick** · generated 2026-11-06 22:15_

Mean test RMSE, 2×32 network.

| variant | f1 | f4 | f6 |
|---|---|---|---|
| tanh | 1.98e-03 | 1.67e-04 | 1.61e-03 |
| sigmoid | 0.653 | 3.55e-04 | 0.0143 |
| SiLU | 2.12e-03 | 5.12e-04 | 2.17e-03 |
| ReLU, zero bias | 5.01e-03 | 9.14e-04 | 5.40e-05 |
| ReLU, kink init | 5.57e-03 | 1.03e-03 | 5.33e-05 |
| SIREN w0=10 | 6.54e-04 | 6.21e-04 | 1.08e-03 |
| SIREN w0=30 | 2.03e-03 | 1.97e-03 | 1.25e-03 |

ReLU 1×32 on f1 (hinge placement at initialisation):

| bias init | RMSE per seed | R² per seed |
|---|---|---|
| zero (all kinks at x=0) | 0.308, 0.0242 | 0.8100, 0.9988 |
| kink | 0.0163, 9.53e-03 | 0.9995, 0.9998 |
