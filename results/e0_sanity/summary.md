## E0 · Engine sanity and universal approximation

_Profile: **quick** · generated 2026-11-06 22:13_

### Gradient check (64 random 2-D points)

| model | params | loss-gradient error | Jacobian error |
|---|---|---|---|
| tanh 2x16 | 337 | 1.6e-10 | 9.1e-11 |
| sigmoid 2x16 | 337 | 1.1e-10 | 2.3e-10 |
| silu 2x16 | 337 | 1.6e-10 | 1.7e-10 |
| sine (SIREN) 2x16 | 337 | 1.3e-10 | 1.3e-10 |
| relu 2x16 (kink init) | 337 | 8.7e-11 | 1.8e-10 |
| Fourier feat. + tanh | 289 | 2.3e-10 | 1.3e-10 |
| KAN [2,4,1] G=5 | 113 | 1.5e-10 | 1.9e-10 |

Gradient error = max |numerical − analytic| / max |analytic| over 200 random coordinates. A ReLU network can exceed 1e-6 only when a ±1e-6 probe pushes a pre-activation across 0 (the derivative does not exist there); smooth layers are not affected.

### Universal approximation, target f1 = sin(2πx)

| hidden units | params | test RMSE | max error |
|---|---|---|---|
| 1 | 4 | 0.651 | 1.13 |
| 2 | 7 | 0.257 | 0.842 |
| 4 | 13 | 3.81e-03 | 0.0249 |
| 8 | 25 | 1.03e-04 | 8.95e-04 |
