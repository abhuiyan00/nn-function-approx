# Findings so far (quick profile)

Plain-language reading of the first complete run (`run_all.py --profile quick`, 6 Nov 2026:
2 seeds, 2,000 training points). Numbers come from [RESULTS.md](RESULTS.md). The full
profile (5 seeds, 20,000 points) must confirm them before they go into the report; effects
smaller than about 1.5× are within seed spread (pilot, 17 Oct) and are not claimed.

## Headlines

1. **The engine is correct.** Analytic gradients and the Levenberg-Marquardt Jacobian agree
   with finite differences to about 1e-10 for every layer type (E0).
2. **Most of the error of an Adam-trained MLP is optimisation error, not a limit of the
   network.** The same 2 × 16 network trained with our Levenberg-Marquardt is 54-81× more
   accurate (median of 6 seeds, U1); on Runge it reaches RMSE ≈ 1e-6 (E3, U1).
3. **The network is not always the right tool.** On smooth 1-D functions a Chebyshev
   least-squares polynomial is 2-10 orders of magnitude better at equal size (E6). Networks
   become competitive or better on the 2-D Franke function and at the kink of |x|.
4. **KANs win where the target has Kolmogorov-Arnold structure, and then they explain it.**
   A 74-parameter KAN [2,1,1] beats every MLP on exp(sin πx + y²) and its three learned edge
   functions match sin πx, y² and exp(·) (U2). On Franke the MLP is better.
5. **Spectral bias is real and fixable.** A tanh MLP has not learned the sin 16πx component
   after 3,000 full-batch steps; with Fourier features it is learned in 100 steps and the
   final error is ~100× lower (U3).
6. **Trust has to be measured.** Ensemble spread grows 20-35× outside the training box but
   still covers almost none of the extrapolation error; split conformal intervals hit their
   nominal 90 % coverage, and a noise-weighted score keeps it in every region (U4).

## Per experiment

### E0 · Sanity and universal approximation
* Gradient error 1e-10 for tanh, sigmoid, SiLU, SIREN, ReLU (away from kinks), Fourier
  features and KAN.
* With LM, 4 tanh units give RMSE 3.8e-3 on sin 2πx and 8 units give 1.0e-4: a handful of
  shifted S-curves is enough. 1-2 units cannot make two periods.

### E1 · Depth vs width
* At 97 parameters on sin 2πx, 2 × 8 is 8× more accurate than 1 × 32 (Adam). On Franke the
  best network is 3 × 32 (RMSE 1.4e-3).
* With Adam, a single wide layer stalls on f1 (RMSE ≈ 0.2) although E0 shows 8 units suffice
  with LM → the one-layer failure is about training, not capacity.

### E2 · Activation and initialisation
* tanh is best on Runge (1.7e-4); ReLU is 30× better than tanh on |x| (5e-5 vs 1.6e-3)
  because a ReLU network *is* piecewise linear.
* Sigmoid fails on sin 2πx within the budget (RMSE 0.65 = predicting the mean): not
  zero-centred, saturates.
* SIREN with ω₀ = 10 is the best on sin 2πx; ω₀ = 30 is worse → ω₀ is a sensitive knob.
* 1 × 32 ReLU on f1: zero-bias start (all hinges at x = 0) ends at R² 0.81 for one seed; kink
  initialisation reaches R² 0.9995 for both seeds (confirms the pilot).

### E3 · Learning rate, batch, optimiser
* Broad good band: batch 16-64 with lr 1e-4…3e-3 (RMSE 1.2-4e-4). Full batch needs lr ≥ 1e-2
  because a fixed epoch budget gives it only 600 updates.
* Optimisers on 2 × 16 (best lr each): SGD 8.8e-4, momentum 9.0e-4, Nesterov 9.0e-4,
  Adam 6.5e-5, **LM 1.2e-6** in 300 iterations.

### E4 · Data size and noise
* Noise-free: N hardly matters beyond 100 points for this smooth 1-D target with Adam; the
  error floor is the optimiser (LM: ≈ 4e-5 at every N).
* Noisy: test error vs the exact f is 0.14-0.22 σ at N = 5,000, i.e. the network averages
  the noise away. With mini-batch Adam the gain stops after N ≈ 500; LM keeps improving
  (σ = 0.05: 0.022 → 0.0036 from N = 100 to 5,000). A cosine learning-rate schedule did not
  change this.
* Overfitting made visible: N = 100, σ = 0.2, 2 × 128. After the best validation epoch the
  error against the exact f doubles while training MSE keeps falling below σ².

### E5 · Early stopping vs weight decay
* Patience 5-50 epochs stops on the initial plateau and returns the "mean" function
  (RMSE 0.65); patience 200 gives 0.038 → patience must be longer than the plateau.
* L2 λ = 1e-4 is best (0.029, better than no regularisation 0.043); λ ≥ 1e-3 underfits
  completely (0.65).

### E6 · Classical baselines
* sin 2πx: Chebyshev reaches machine precision with 33 coefficients; the best network
  (LM, 321 params) stops at 2e-5.
* Franke (2-D): MLP-LM 1.4e-4 with 337 params vs Chebyshev 1.3e-3 with 289; splines 8.6e-5
  with 400.
* |x|: MLP-LM 1.1e-4 (193 params) vs spline 2.2e-4 (128) and Chebyshev 6.6e-4 (97).
* Runge: degree-20 interpolation on equispaced nodes has max error 59.8 (pilot: 59.8, same
  setting), on Chebyshev nodes 0.015. The phenomenon belongs to the nodes, not to polynomials.

### U1 · Levenberg-Marquardt
* Median Adam/LM error ratio ×81 on Runge, ×54 on Franke. The worst LM seed is still better
  than the best Adam seed on both targets. Hybrid Adam → LM is similar to LM alone.
* μ falls from 10 to 1e-3 (Franke) and 1e-5 (Runge): LM moves from gradient-descent-like
  steps to Gauss-Newton steps.
* Cost: one LM iteration solves a P × P system, so it suits networks up to a few hundred
  weights (MATLAB's guidance agrees).

### U2 · KAN vs MLP (both trained with LM)
* f7 = exp(sin πx + y²): KAN [2,1,1] 3.8e-5 with 74 params; best MLP 2.1e-4.
* f4, f6: KAN [1,5,1] 1.5-2.2× better than the best MLP of similar size; f5 Franke: MLP better.
* Grid extension 3 → 5 → 10 → 20 gives a staircase learning curve; final 5e-5 vs 5e-4 for an
  MLP 2 × 12 trained as long.
* A KAN run takes 6-8× longer than an MLP run in our NumPy code (B-spline evaluation).
* The finest grid (G = 20) makes KAN [2,5,1] *worse* on both 2-D targets (best seed: f7 2.4e-3, f5 2.4e-2):
  likely too many spline coefficients (230 in the first layer) for 2,000 points → tune G like width.

### U3 · Spectral bias
* tanh MLP: k = 1 learned at step 100, k = 4 at 650, k = 16 never (3,000 steps) → RMSE 0.11.
* Fourier features: all three by step 100 → RMSE 1.1e-3. KAN: k = 16 by step 550 → 2.3e-3.
* SIREN (ω₀ = 30): all components by step 50, but wiggles between the 1,024 samples → 0.093.

### U4 · Uncertainty
* Ensembles on f2 outside [-2, 2]: mean spread 0.15 (tanh) / 0.46 (ReLU) vs error ≈ 2.7 → the
  spread flags extrapolation but under-states the error by an order of magnitude.
* Split conformal at 90 %: coverage 0.902 for both scores. Plain intervals over-cover the quiet
  region (1.00) and under-cover the noisy one (0.61); weighted intervals stay at 0.86-0.96
  per region and are 21 % narrower on average.

## Open points for the full run
* Re-check E4 with 20,000 points (pilot range) and E1 with 5 seeds.
* U1 with 10 seeds; U2 with 5 seeds and Adam-trained KANs as a second optimiser.
* E2 SIREN ω₀ sweep {5, 10, 20, 30}; U3 SIREN with ω₀ = 10.
