# Research notes

What the literature says, and which design decision in this repository each source drives.
Sources checked November 2026.

## 1. Why a neural network can approximate a function at all

* **Universal approximation.** A network with one hidden layer of sigmoidal (Cybenko, 1989) or
  any non-polynomial (Hornik, 1991; Leshno et al., 1993) activation can approximate any
  continuous function on a compact set to any accuracy, given enough units. The theorem says
  nothing about *how many* units or whether training finds the weights.
  → **E0** shows it literally: the fit is a weighted sum of shifted tanh curves (1, 2, 4, 8 units).
* **Depth.** Deeper networks can represent some functions with far fewer parameters than shallow
  ones (Telgarsky, 2016; Yarotsky, 2017 for ReLU approximation rates).
  → **E1** compares depth 1-3 at equal parameter count.

## 2. Spectral bias and how to beat it

* **Spectral bias / frequency principle.** Rahaman et al. (ICML 2019, *On the Spectral Bias of
  Neural Networks*) showed with Fourier analysis that networks learn low-frequency components
  first: the learning speed depends on frequency. Xu et al. call the same effect the frequency
  principle. → **U3** measures the learned amplitude of each component of f3 during training.
  https://proceedings.mlr.press/v97/rahaman19a
* **Fourier features.** Tancik et al. (NeurIPS 2020) map the input through
  γ(x) = [sin 2πBx, cos 2πBx] with random B ~ N(0, s²) before the MLP; this widens the band of
  frequencies the network can learn quickly. Recent work (e.g. Molina et al., 2024, *Understanding
  the dynamics of the frequency bias*) analyses exactly this model.
  → `FourierFeatures` layer, U3. https://arxiv.org/abs/2405.14957
* **SIREN.** Sitzmann et al. (NeurIPS 2020) use sin(ω₀ z) activations with a special
  initialisation; ω₀ sets the bandwidth. → `Sine` activation, SIREN init in `Dense`; E2, U3.
  Our U3 run shows the price: with ω₀ = 30 and 1,024 random points SIREN learns every component
  fast but oscillates between samples.
* **KANs and spectral bias.** Wang et al. (2024, *On the expressiveness and spectral bias of KANs*)
  argue that KANs suffer less from spectral bias than MLPs. → KAN is included in U3.
  https://arxiv.org/abs/2410.01803

## 3. Kolmogorov-Arnold Networks (KAN)

* **Original paper.** Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark, *KAN:
  Kolmogorov-Arnold Networks*, ICLR 2025 (arXiv 2404.19756). Instead of fixed activations on
  nodes, every edge carries a learnable univariate function (B-spline + SiLU base); grid extension
  refines the splines without restarting training; trained KANs can be simplified into formulas.
  → `KANLayer` (own vectorised Cox-de Boor B-splines and their derivative), `refit` for grid
  update / extension, U2. https://arxiv.org/abs/2404.19756
* **Fair comparisons.** Yu, Yu & Wang (2024, *KAN or MLP: A Fairer Comparison*) matched
  parameters/FLOPs and found KAN ahead only on symbolic-formula tasks; an MLP with B-spline
  activations matched KAN elsewhere. Hou et al. (2024-25, *A critical assessment*) reach the same
  conclusion. A 2025 practitioner's guide (Noorizadegan, Wang & Ling) stresses that the outcome
  depends on the basis and that conclusions in the literature diverge.
  → U2 compares **at equal parameter count and with the same optimiser** (our LM) and reports
  targets where KAN wins *and* where it loses.
  https://arxiv.org/abs/2407.16674 · https://arxiv.org/abs/2407.11075 · https://arxiv.org/abs/2510.25781

## 4. Optimisers: first order vs second order

* **Adam** (Kingma & Ba, 2015) is the default first-order method; SGD with (Nesterov) momentum
  is the classical alternative. → `optim.py`, E3.
* **Levenberg-Marquardt.** Levenberg (1944), Marquardt (1963); applied to MLP training by Hagan &
  Menhaj (1994). Update θ ← θ − (JᵀJ + μI)⁻¹Jᵀe, with the Jacobian J of the outputs w.r.t. the
  weights. The MATLAB documentation of `trainlm` describes it as apparently the fastest method for
  moderate-sized feed-forward networks (up to several hundred weights); MATLAB R2024b added an LM
  solver to its deep-learning `trainnet`. Bilski et al. (JAISCR 2020, 2023) work on making LM
  cheaper for bigger networks. → `fit_lm` with Jacobian from our own per-sample backprop; U1.
  https://www.mathworks.com/help/deeplearning/ref/trainlm.html
* **Second-order methods are back in scientific ML.** Kiyani, Shukla, Urbán, Darbon & Karniadakis
  (2025, *Which optimizer works best for physics-informed neural networks and Kolmogorov-Arnold
  networks?*) report orders-of-magnitude accuracy gains of quasi-Newton methods (BFGS,
  self-scaled Broyden) over Adam, without other tricks. Our U1 is the regression analogue with LM.
  https://arxiv.org/abs/2501.16371

## 5. Uncertainty: where to trust the approximation

* **Deep ensembles.** Lakshminarayanan, Pritzel & Blundell (NeurIPS 2017): train M networks from
  different seeds; their spread is a simple, strong uncertainty estimate. → U4(a), extrapolation of f2.
* **Conformal prediction.** Vovk, Gammerman & Shafer (2005); Lei, G'Sell, Rinaldo, Tibshirani &
  Wasserman (JASA 2018, *Distribution-free predictive inference for regression*). Split conformal
  wraps *any* fitted model: absolute residuals on a held-out calibration set give an interval with
  finite-sample marginal coverage ≥ 1 − α, assuming only exchangeability. The guarantee is
  marginal, not per region; locally weighted scores |y − f̂(x)| / g(x) adapt the width to
  heteroscedastic noise. → U4(b): plain vs weighted, coverage per x-region.

## 6. Classical approximation theory (the yardstick)

* **Runge (1901).** Polynomial interpolation on equispaced nodes diverges for 1/(1 + 25x²); on
  Chebyshev nodes it converges. → E6 Runge figure (degree-20 equispaced max error 59.8 vs 0.015 on
  Chebyshev nodes).
* **Chebyshev approximation.** For analytic functions, Chebyshev expansions converge
  geometrically (Trefethen, *Approximation Theory and Approximation Practice*, 2013).
  → `ChebyshevLS`; explains why a polynomial wins on smooth 1-D targets in E6.
* **Splines.** Cubic B-splines converge as O(h⁴) for smooth f and are local, so kinks do not spoil
  the whole fit (de Boor, *A Practical Guide to Splines*). → `SplineLS`, same B-spline code as KAN.
* **Franke's function** (Franke, 1979, Naval Postgraduate School report) is the standard 2-D test
  for scattered-data approximation. → f5.

## 7. Initialisation

* Glorot & Bengio (2010) for tanh/sigmoid, He et al. (2015) for ReLU; SIREN initialisation
  (Sitzmann et al., 2020). Our pilot found that zero-bias ReLU puts every hinge at x = 0 in 1-D;
  the `kink` initialisation spreads the hinges over the box. → `Dense(weight_init, bias_init)`, E2.

## 8. What we deliberately do not use

* No PyTorch / TensorFlow / JAX / scikit-learn in the engine: the assignment asks for our own
  software. SciPy is not needed either.
* No GPU: networks with ≤ 20k parameters train in seconds on one CPU core.
