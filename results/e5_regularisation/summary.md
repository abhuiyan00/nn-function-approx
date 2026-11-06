## E5 · Early stopping and weight decay

_Profile: **quick** · generated 2026-11-06 22:21_

| patience | mean test RMSE | mean best epoch |
|---|---|---|
| 5 | 0.656 | 3 |
| 15 | 0.652 | 18 |
| 50 | 0.652 | 18 |
| 200 | 0.0379 | 1994 |
| 1000 | 0.0398 | 3139 |

| λ (no early stop) | mean test RMSE |
|---|---|
| 0 | 0.0428 |
| 1e-05 | 0.0376 |
| 0.0001 | 0.0285 |
| 0.001 | 0.652 |
| 0.01 | 0.652 |
