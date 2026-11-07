# Experiment protocol

Common settings unless an experiment says otherwise.

| Setting | Value |
|---|---|
| Training inputs | uniform random in the domain box, data seed 0 |
| Labels | y = f(x) + ε, ε ~ N(0, σ²); σ = 0 unless stated |
| Validation | separate uniform sample, same noise; used only for early stopping |
| Test | 10,000-point grid (100 × 100 in 2-D), **exact** f, never seen in training |
| Pre-processing | x → [-1, 1] from the known box; y → z-score with training mean / std |
| Post-processing | inverse z-score; metrics in original units |
| Default network | 2 hidden layers × 32 tanh units, Glorot init, zero biases |
| Default training | Adam, lr 1e-3, batch 64, ≤ 600 epochs (quick), patience 50, best weights restored |
| Levenberg-Marquardt | full batch (≤ 4,000 points), μ₀ = 1e-3, ×10 / ×0.1, ≤ 300 iterations (quick) |
| Seeds | quick: 1, 2 · full: 1-5 (U1: 6 / 10) |
| Metrics | RMSE, max error, R², NRMSE = RMSE / std(f), RMSE in the edge band (outer 5 % of the box) vs interior |

Profiles: `quick` = 2,000 train / 500 val points; `full` = 20,000 / 5,000 points and longer budgets.
Every row of every run is stored in `results/<exp>/*.csv` with its full spec.

---

### E0 · Engine sanity and universal approximation
* **Question** Are analytic gradients and the LM Jacobian correct? How does one hidden layer build f?
* **Varied** layer type (Dense with 5 activations, Fourier features, KAN); hidden units H ∈ {1, 2, 4, 8}
* **Measured** max finite-difference error (scaled by the largest gradient entry); test RMSE of sin 2πx
* **Figures** `e0_sanity_gradcheck.png`, `e0_sanity_units.png`

### E1 · Architecture
* **Question** Deep or wide at equal parameter count?
* **Varied** depth {1, 2, 3} × width {4, 8, 16, 32, 64}; targets f1, f4, f5
* **Measured** test RMSE vs number of parameters (band = seed min-max)
* **Figures** `e1_architecture_params.png`, `e1_architecture_franke.png`

### E2 · Activation and initialisation
* **Question** Which non-linearity for which target; does the starting point matter?
* **Varied** tanh, sigmoid, SiLU, ReLU (zero bias / kink init), SIREN ω₀ ∈ {10, 30}; f1, f4, f6.
  Extra: 1 × 32 ReLU zero vs kink bias on f1.
* **Figures** `e2_activation_init_dotplot.png`, `e2_activation_init_fits.png`

### E3 · Training parameters
* **Question** Sensitivity to learning rate and batch size; which optimiser?
* **Varied** lr {1e-4 … 3e-2} × batch {16, 64, 256, full}; optimisers SGD, momentum, Nesterov, Adam (best of 3 lrs each), LM. Target f4.
* **Figures** `e3_training_lr_batch.png` (cell text = updates made), `e3_training_optimizers.png`

### E4 · Data size and noise
* **Question** How does error scale with N and σ? What does overfitting look like against the exact f?
* **Varied** N {100 … 5,000} × σ {0, 0.05, 0.1, 0.2}; trainer Adam (constant / cosine lr) or LM. Equal update budget for every N.
* **Extra** N = 100, σ = 0.2, 2 × 128 without early stopping: train / val / exact-test curves.
* **Figure** `e4_data_noise_n_sigma.png`

### E5 · Regularisation
* **Question** Early stopping vs L2 weight decay, and the sensitivity to their knob.
* **Setting** f1, N = 250, σ = 0.1, 2 × 128 (16.9k params, 68× more parameters than points)
* **Varied** patience {5, 15, 50, 200, 1000}; λ {0, 1e-5, 1e-4, 1e-3, 1e-2}
* **Figure** `e5_regularisation_regularisation.png`

### E6 · Classical baselines
* **Question** Is a neural network the right tool for this problem?
* **Compared** MLP 2×H (Adam, LM), Chebyshev least squares, cubic-spline least squares at equal size; Runge interpolation demo.
* **Figures** `e6_baselines_scaling.png`, `e6_baselines_runge.png`

### E7 · The missing middle
* **Question** How close can a curve get where it saw no data, and can you tell from the data which curve is closest?
* **Setting** A centred hole with no training or validation point; the exact f still scores the whole range. N = 1,000 train / 250 val (full: 2,000 / 500).
* **Varied** targets m1-m5; hole width {10, 20, 30, 40} % of each axis; σ {0, 0.05}; models MLP tanh (LM, Adam), MLP ReLU, Fourier features + tanh, SIREN ω₀ = 10, KAN, Chebyshev LS, cubic-spline LS. Classical sizes are chosen on the validation points.
* **Measured** RMSE inside the hole, outside it and overall; which model the validation error alone would pick
* **Figures** `e7_missing_middle_fits.png`, `e7_missing_middle_width.png`

### U1 · Levenberg-Marquardt
* **Question** Is the gap to classical methods due to the network or to the optimiser?
* **Compared** Adam · LM · Adam (100 epochs) → LM on the same 2 × 16 network; f4, f5; 6 seeds (full: 10)
* **Figure** `u1_levenberg_marquardt_lm_vs_adam.png` (progress vs time, damping μ, seed robustness)

### U2 · KAN vs MLP
* **Question** Do KANs approximate better at equal size? Can we read the learned formula?
* **Compared** KAN [d, 5, 1] with G ∈ {3, 5, 10, 20}, KAN [2, 1, 1] (f7), MLP 2×H; all trained with LM; f7, f5, f4, f6
* **Extras** grid extension 3 → 5 → 10 → 20 during training; affine fit of the learned edges to sin πx, y², exp
* **Figures** `u2_kan_vs_mlp_params.png`, `u2_kan_vs_mlp_interpret.png`

### U3 · Spectral bias
* **Question** When is each frequency of f3 learned, and which models remove the delay?
* **Compared** tanh MLP, ReLU MLP, Fourier features + tanh, SIREN, KAN; full-batch Adam, 3,000 steps (full: 10,000)
* **Measured** relative error of the amplitude of sin(kπx), k = 1, 4, 16, every 25 steps; residual spectrum
* **Figures** `u3_spectral_bias_components.png`, `u3_spectral_bias_spectrum.png`

### U4 · Uncertainty
* **(a)** 5 networks (tanh, ReLU) on f2 inside [-2, 2], queried on [-3, 3]: spread vs error.
* **(b)** f1 with σ(x) = 0.02 + 0.2((x + 1)/2)²; split conformal, plain vs weighted score; 200 random calibration splits; coverage overall and per x-bin.
* **Figure** `u4_uncertainty_uncertainty.png`

---

## How to add an experiment

1. Copy the closest script in `experiments/`, keep `parse_profile`, `run_many`, `save_rows`, `write_summary`.
2. Put one run in one dict (`run_one` understands every key of `common.DEFAULTS`).
3. Name figures `<exp>_<what>.png` via `fig(EXP, "what")` so `run_all.py` collects them.
4. Add the name to `ORDER` in `run_all.py` and to the VS Code task list.
5. Optional: describe its interactive chart in `ui/src/lib/experimentCharts.ts`. Without one, the interface still lists it, with its tables, figures and raw data.
