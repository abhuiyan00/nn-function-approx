## E4 · Data size and noise

_Profile: **quick** · generated 2026-11-06 22:19_

Geometric-mean test RMSE vs exact f (rows σ and trainer, columns N). Trainer: Adam 2×32 batch 64 with constant or cosine-annealed learning rate, or Levenberg-Marquardt (lm) 2×16 full batch.

| σ | trainer | N=100 | N=250 | N=500 | N=1000 | N=2000 | N=5000 |
|---|---|---|---|---|---|---|---|
| 0 | constant | 1.81e-03 | 1.90e-03 | 1.71e-03 | 1.78e-03 | 1.93e-03 | 3.51e-03 |
| 0 | cosine | 2.15e-03 | 3.02e-03 | 2.84e-03 | 3.53e-03 | 3.66e-03 | 4.99e-03 |
| 0 | lm | 5.47e-05 | 3.47e-05 | 3.53e-05 | 4.17e-05 | 3.36e-05 | 3.94e-05 |
| 0.05 | constant | 0.0178 | 0.0148 | 0.0116 | 0.0118 | 0.0102 | 0.011 |
| 0.05 | cosine | 0.018 | 0.0161 | 0.0125 | 0.0128 | 0.012 | 0.0115 |
| 0.05 | lm | 0.0216 | 0.0131 | 9.27e-03 | 5.06e-03 | 4.55e-03 | 3.58e-03 |
| 0.1 | constant | 0.0405 | 0.0353 | 0.0189 | 0.0196 | 0.0152 | 0.0178 |
| 0.1 | cosine | 0.0379 | 0.0292 | 0.0207 | 0.0218 | 0.0178 | 0.017 |
| 0.1 | lm | 0.043 | 0.0221 | 0.0184 | 0.0162 | 0.0105 | 7.45e-03 |
| 0.2 | constant | 0.0756 | 0.0627 | 0.0415 | 0.0408 | 0.0229 | 0.0281 |
| 0.2 | cosine | 0.0745 | 0.0552 | 0.0432 | 0.0517 | 0.0287 | 0.0293 |
| 0.2 | lm | 0.0817 | 0.0466 | 0.0393 | 0.0251 | 0.0141 | 0.0147 |

LM is full batch and uses at most 4,000 training points (N = 5000 is subsampled).

Overfitting run (N=100, σ=0.2, 2×128): best validation epoch 3264; test MSE there 9.03e-03, at the last epoch 0.0184 (×2.0); final train MSE 0.0305 vs noise variance 0.04.
