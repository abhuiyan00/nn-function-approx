## U3 · Spectral bias

_Profile: **quick** · generated 2026-11-06 22:29_

Full-batch Adam, 3000 steps, lr 1e-3, one seed.

| model | params | final RMSE | step k=1 < 10 % | step k=4 < 10 % | step k=16 < 10 % |
|---|---|---|---|---|---|
| tanh MLP | 8513 | 0.111 | 100 | 650 | never |
| ReLU MLP | 8513 | 0.0962 | 25 | 225 | never |
| Fourier features + tanh | 8385 | 1.13e-03 | 25 | 50 | 100 |
| SIREN | 8513 | 0.0932 | 25 | 25 | 50 |
| KAN | 4065 | 2.28e-03 | 50 | 75 | 550 |
