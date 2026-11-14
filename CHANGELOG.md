# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Eraser on the draw pad.** Draw a curve, erase a part of it, and run the Curve Finder or the Training
  Lab. The erased part stays on the pad as a dashed line: no model trains on it, and every model is scored on it.
- **Sensitivity page.** Varies one setting (learning rate, batch, L2, training points, noise, patience,
  epochs, width, depth or KAN grid) over 2 to 12 values and 1 to 5 seeds, and says whether the setting or
  the starting weights matter more. API: `POST /api/sweeps`.
- **Network in 3-D.** The Training Lab draws the network while it trains, one forward pass at a time.
- **Analysis of a finished run.** Share of each hidden unit, slope against the exact derivative, loss
  landscape with the path training took, KAN edges with their closest simple formula, and a replay of
  the whole training.
- **Trust band.** A 90 % band from split conformal prediction on fresh points, left out where there was no data.
- **RBF network baseline.** Gaussian units with centres from k-means and output weights by least squares,
  in any number of inputs. The Curve Finder now ranks nine models.
- **Formulas of three inputs**, using `x`, `y` and `z`, shown as a slice.
- **Export.** A trained or saved network as one Python file that needs only NumPy.
- **Textbook tests** (`tests/test_textbook.py`): Runge's phenomenon, Chebyshev convergence, universal
  approximation by width, RBF convergence, noise averaging, no extrapolation, the cost of a missing middle,
  and bit-for-bit reproducibility.
- A browser test of a full class exercise: draw, erase, search, and compare the error inside and outside the hole.
- `npm --prefix ui run capture` makes every screenshot and GIF in the README again.
- Issue templates, a pull request template and `CITATION.cff`.

### Changed

- Dependencies live only in `pyproject.toml`; `requirements.txt` is gone. Install with `pip install -e ".[dev,ui]"`.
- The README walkthrough covers the new pages and panels, and every screenshot was made again.

### Removed

- Six interface components that no page used.

## [0.1.0] - 2026-11-07

### Added

- Engine in NumPy only: MLP, SIREN, Fourier features and KAN layers, SGD, momentum, Nesterov, Adam and
  Levenberg-Marquardt, early stopping, gradient and Jacobian checks.
- Twelve target functions, five of them with a missing middle, plus typed formulas and measured points.
- Experiments E0 to E7 and U1 to U4 with quick-profile results and figures.
- Command line: `funcprox train`, `predict`, `targets`, `ui`.
- Browser interface: Overview, Curve Finder, Training Lab, Experiments, Run Explorer, Saved Models, Experiment Runner, Documents.
- Continuous integration on Ubuntu and Windows.

[Unreleased]: https://github.com/abhuiyan00/nn-function-approx/compare/c866183...HEAD
[0.1.0]: https://github.com/abhuiyan00/nn-function-approx/tree/c866183
