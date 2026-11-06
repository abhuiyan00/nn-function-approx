# Final report outline

The supervisor asks for five parts. This file maps each part to the code, experiments and
figures that feed it, and names the owner (A / B, see [PROJECT_PLAN.md](PROJECT_PLAN.md)).
Target length: 15-20 pages + appendix.

## 1. Short description of the problem and assumptions · owner A
* The "device" view of function approximation; why the exact f is known in our study.
* Assumptions A1-A6 (PROJECT_PLAN §2). Universal approximation in two sentences, with E0 as the picture.
* Seven targets and the property each one probes (`functions.py`, README table).
* Figure: `e0_sanity_units.png`.

## 2. Training and testing sets, pre-processing · owner A
* Sampling: uniform random training/validation points, 10,000-point exact test grid; N and σ values used.
* Pre-processing: domain scaling to [-1, 1], z-score of labels with training statistics (`data.py`), and why
  (tanh saturation, comparable learning rates across targets, same scaling for extrapolation points).
* Noise model and the heteroscedastic case (U4).
* Table: data sets per experiment (from EXPERIMENTS.md).

## 3. Output data and post-processing · owner A
* Network output is a z-score; inverse transform to original units (`metrics.predict_original`).
* Metrics: RMSE, max error, R², NRMSE, edge-band vs interior error; why the test uses the exact f
  and the validation uses noisy labels (E4 overfitting panel).
* Post-processing for uncertainty: ensemble mean ± 2 sd, conformal intervals (U4).
* Figure: `e4_data_noise_n_sigma.png` (right panel).

## 4. Tuned topology and parameters of the engine · owner B
* Engine architecture: layers, per-sample backprop, Jacobian, optimisers, LM, KAN, Fourier features,
  SIREN (`src/fapprox`), tests and gradient check (E0 table).
* Sensitivity study and the chosen settings, each with its evidence:
  - topology: E1 (depth vs width) · activation and init: E2 · lr, batch, optimiser: E3
  - regularisation: E5 · data budget: E4
* Final recommended configurations per target (table: architecture, optimiser, params, test RMSE).
* Figures: `e1_architecture_params.png`, `e2_activation_init_dotplot.png`, `e3_training_lr_batch.png`,
  `e5_regularisation_regularisation.png`.

## 5. Results analysis and final remarks · owner B (remarks: both)
* Is the network fitted to the problem? E6 scaling plot + Runge picture; when to prefer classical methods.
* U1: optimisation vs approximation error. U2: KAN vs MLP and interpretability.
* U3: spectral bias and its fixes. U4: where the approximation can be trusted.
* Limitations: synthetic targets, ≤ 2-D inputs, LM cost grows as P³, seeds.
* Final remarks: 5-7 bullet conclusions, each tied to a figure.

## Appendix
* A. Reproduction: `pip install -e .`, `python experiments/run_all.py --profile full`, run time.
* B. Full result tables (docs/RESULTS.md).
* C. Code listing of the core: `layers.py`, `train.py` (LM), `bspline.py`.
