# Neural network as a device for function approximation

[![CI](https://github.com/abhuiyan00/nn-function-approx/actions/workflows/ci.yml/badge.svg)](https://github.com/abhuiyan00/nn-function-approx/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Engine: NumPy only](https://img.shields.io/badge/engine-NumPy%20only-013243)
[![License: MIT](https://img.shields.io/badge/license-MIT-97CA00)](LICENSE)

Train neural networks built from scratch in NumPy, watch them learn in your browser, and measure
how close they get to the **exact** function: how well, why, and when a classical method is the
better tool.

![A network learning a multi-scale function live in the Training Lab](docs/screenshots/training-lab-live.gif)

*A live run, sped up. The network picks up the slow waves first and misses the fastest one: spectral bias, experiment U3.*

**[Run it](#run-it)** · **[Walkthrough](#walkthrough)** · **[Results](#results)** · **[How it works](#how-it-works)** · **[Documents](#documents)**

Soft Computing, Project I, Topic 11 · Wrocław University of Science and Technology, 2026/27 · Supervisor: Prof. Jacek Mazurkiewicz

## What is in this repository

| | |
|---|---|
| **Engine** | MLP, SIREN, Fourier features and Kolmogorov-Arnold networks, with Adam and Levenberg-Marquardt training. No ML framework |
| **Experiments** | 11 studies with their results and figures committed, so you can read them without training anything |
| **Web interface** | Train networks live, explore every recorded run, query saved models, re-run experiments |

## Run it

You need [Python](https://www.python.org/downloads/) 3.10 or newer and [Node.js](https://nodejs.org/) 20.19 or newer.

**1. Get the code and create an environment**

```bash
git clone https://github.com/abhuiyan00/nn-function-approx.git
cd nn-function-approx
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

**2. Install the project and build the interface**

```bash
pip install -e ".[ui]"
npm --prefix ui ci
npm --prefix ui run build
```

**3. Start it**

```bash
python -m funcprox ui              # opens http://127.0.0.1:8765
```

Next time you only need to activate the environment and repeat step 3. Press `Ctrl+C` in the terminal to stop.

## Walkthrough

### 1. Start at the Overview

The headline numbers and the main question: is a network the right tool for this target? Everything
is computed from the result files in the repository.

![Overview page](docs/screenshots/overview.png)

### 2. Train a network in the Training Lab

Click any preset at the top. It loads the settings of a finding and starts training, and you watch the
fit, the error against the exact function, the learning curves and, for a KAN, the learned edge
functions. Afterwards the Lab compares the network with classical fits no larger than itself.
A good first pair: **Adam on Runge**, then **LM on Runge**. Same network, roughly 80 times less error in experiment U1.

To try your own idea, change anything in the panel on the left and press **Train**.

![Training Lab after a finished run](docs/screenshots/training-lab.png)

### 3. Read an experiment

**Experiments** lists all 11 studies. Each one shows its question, hypothesis and findings, then
interactive charts, summary tables, the report figures and the raw runs.

![Experiment list](docs/screenshots/experiments.png)

![Interactive results of an experiment](docs/screenshots/experiment-detail.png)

### 4. Compare runs in the Run Explorer

Plot any measurement against any setting across all recorded runs. The address keeps your choices, so a
view can be shared by copying the link.

![Run Explorer](docs/screenshots/run-explorer.png)

### 5. Save a model and query it

After a run in the Lab, press the **Save** button. **Saved Models** reloads the file, checks it against the
exact function and answers for any input you type.

![Saved Models page](docs/screenshots/saved-models.png)

### 6. Re-run the experiments

**Experiment Runner** runs the scripts one at a time with a live log, and the rest of the interface picks up
the new results when a run ends.

> A run overwrites the tracked files in `results/` and `docs/figures/`. Commit your work first, so you can
> compare or go back. The **quick** profile takes minutes per experiment, the **full** profile takes hours.

![Experiment Runner](docs/screenshots/runner.png)

### 7. Read the documents

**Documents** shows this README and everything in `docs/`, always the version on disk.

![Documents page](docs/screenshots/documents.png)

### Prefer the terminal?

The engine works without the interface or Node.js:

```bash
pip install -e .
python -m funcprox targets
python -m funcprox train --target f4 --hidden 16,16 --optimizer lm --epochs 300 --out runs/f4.npz
python -m funcprox predict runs/f4.npz 0 0.5 1
python experiments/run_all.py --profile quick        # all 11 experiments, about 20 min on 2 cores
```

## Results

The assignment asks for a working engine and a sensitivity study (E0-E6). Four more studies build on recent
research, in the same engine. These numbers come from the quick profile.

| | Study | Result |
|---|---|---|
| **U1** | **Levenberg-Marquardt** second-order training, with the Jacobian from per-sample backprop | Same 2×16 network, same data: **54-81× lower error** than Adam |
| **U2** | **Kolmogorov-Arnold Network** (ICLR 2025) with B-splines and grid extension | A 74-parameter KAN beats every MLP on f7 and **recovers the formula** exp(sin πx + y²) |
| **U3** | **Spectral bias** measured per frequency, fixed with Fourier features, SIREN or a KAN | A tanh MLP never learns the 16π component in 3,000 steps; Fourier features learn it in 100 |
| **U4** | **Trust**: deep ensembles and split **conformal prediction** | 90 % intervals hit 90.2 % coverage; a noise-weighted version lifts the noisiest region from 61 % to 88 % |

And the honest conclusion the supervisor asks for ("check if the soft-computing solution is correctly fitted to
the problem"): **for smooth 1-D functions a Chebyshev polynomial beats every network we trained by 2 to 10 orders
of magnitude. Networks pay off in 2-D, at kinks and on compositional structure** (E6, U2).

![Networks against classical approximators](docs/figures/e6_baselines_scaling.png)

**U1 · Same network, different optimiser.** Levenberg-Marquardt reaches 1e-6 on Runge where Adam stalls near 1e-4.

![Levenberg-Marquardt against Adam](docs/figures/u1_levenberg_marquardt_lm_vs_adam.png)

**U2 · A KAN reads back the formula.** The learned edge functions are sin(πx), y² and exp(·) up to scale and shift.

![Learned KAN edge functions](docs/figures/u2_kan_vs_mlp_interpret.png)

**U3 · Spectral bias.** Plain networks learn low frequencies first; input encodings remove the delay.

![Error per frequency over training](docs/figures/u3_spectral_bias_components.png)

**U4 · Trust.** Ensemble spread grows outside the data but is far too small; conformal intervals keep their promise inside.

![Uncertainty estimates](docs/figures/u4_uncertainty_uncertainty.png)

**E2 · Activation matters.** ReLU networks represent the kink of |x| exactly; tanh rounds it off.

![Fits by activation function](docs/figures/e2_activation_init_fits.png)

All numbers are in [docs/RESULTS.md](docs/RESULTS.md), and the plain-language reading is in [docs/FINDINGS.md](docs/FINDINGS.md).

## How it works

Every error is measured against the exact function on a dense grid, so real approximation quality can be told
apart from fitting the noise.

```mermaid
flowchart LR
    F["Target f(x)<br/>f1…f7, known formula"] --> S["Sampling<br/>uniform x, y = f(x) + noise"]
    S --> PRE["Pre-processing<br/>x → [-1, 1] (domain box)<br/>y → z-score (train stats)"]
    PRE --> ENG
    subgraph ENG["NumPy engine (src/funcprox)"]
        direction TB
        M1["MLP: tanh · sigmoid · ReLU · SiLU"]
        M2["SIREN · Fourier features"]
        M3["KAN: B-spline edges"]
        T1["Adam / SGD / momentum<br/>early stopping, best checkpoint"]
        T2["Levenberg-Marquardt<br/>(J from per-sample backprop)"]
    end
    ENG --> POST["Post-processing<br/>inverse z-score"]
    POST --> EVAL["Metrics vs EXACT f<br/>10,000-point grid<br/>RMSE · max · edge band"]
    EVAL --> FIG["Figures + tables<br/>docs/figures, results/"]
    BASE["Baselines<br/>Chebyshev LS · spline LS"] --> EVAL
```

### Target functions

| key | f(x) | domain | what it tests |
|---|---|---|---|
| f1 | sin(2πx) | [-1, 1] | smooth reference |
| f2 | 2x² | [-2, 2] (test [-3, 3]) | extrapolation |
| f3 | sin πx + ½ sin 4πx + ¼ sin 16πx | [-1, 1] | spectral bias |
| f4 | 1 / (1 + 25x²) | [-1, 1] | Runge phenomenon |
| f5 | Franke's function | [0, 1]² | classic 2-D benchmark |
| f6 | \|x\| | [-1, 1] | non-differentiable point |
| f7 | exp(sin πx + y²) | [-1, 1]² | compositional (KAN toy example) |

### Experiments

| id | question | figure |
|---|---|---|
| E0 | Are the derivatives right? How does one hidden layer build a function? | [gradcheck](docs/figures/e0_sanity_gradcheck.png) · [units](docs/figures/e0_sanity_units.png) |
| E1 | Deep or wide at equal parameter count? | [params](docs/figures/e1_architecture_params.png) · [Franke](docs/figures/e1_architecture_franke.png) |
| E2 | Which activation / initialisation for which target? | [dot plot](docs/figures/e2_activation_init_dotplot.png) · [fits](docs/figures/e2_activation_init_fits.png) |
| E3 | Learning rate × batch size; which optimiser? | [heat map](docs/figures/e3_training_lr_batch.png) · [optimisers](docs/figures/e3_training_optimizers.png) |
| E4 | How does error scale with data size and noise? What does overfitting look like? | [N, σ](docs/figures/e4_data_noise_n_sigma.png) |
| E5 | Early stopping vs weight decay | [regularisation](docs/figures/e5_regularisation_regularisation.png) |
| E6 | Is a network the right tool at all? | [scaling](docs/figures/e6_baselines_scaling.png) · [Runge](docs/figures/e6_baselines_runge.png) |
| U1 | Is the gap to classical methods the network's fault or the optimiser's? | [LM vs Adam](docs/figures/u1_levenberg_marquardt_lm_vs_adam.png) |
| U2 | KAN vs MLP at equal size; can we read the formula back? | [params](docs/figures/u2_kan_vs_mlp_params.png) · [interpret](docs/figures/u2_kan_vs_mlp_interpret.png) |
| U3 | Which frequencies are learned when? | [components](docs/figures/u3_spectral_bias_components.png) · [spectrum](docs/figures/u3_spectral_bias_spectrum.png) |
| U4 | Where can the approximation be trusted? | [uncertainty](docs/figures/u4_uncertainty_uncertainty.png) |

### Engine rules

* No ML framework: the forward pass, backward pass, Jacobian, optimisers and B-splines are implemented in this
  repository. NumPy for arrays, Matplotlib for figures only.
* Every model, seed and data split can be reproduced from the run spec in `results/<exp>/runs.csv`.
* Two profiles: **quick** (2 seeds, 2k points, minutes) for development, **full** (5 seeds, 20k points) for the report.

## Repository layout

```
src/funcprox/            the engine (NumPy only)
  functions.py          7 target functions
  data.py               sampling, noise, pre/post-processing
  activations.py        tanh, sigmoid, ReLU, SiLU, sine (SIREN)
  layers.py             Dense, FourierFeatures, KANLayer (+ per-sample backprop)
  bspline.py            vectorised Cox-de Boor basis and derivative
  model.py              Network, build_mlp, build_kan, Jacobian
  optim.py              SGD, momentum, Nesterov, Adam
  train.py              mini-batch loop, early stopping, cosine lr, Levenberg-Marquardt
  baselines.py          Chebyshev LS, spline LS, polynomial interpolation
  metrics.py            RMSE, max error, R², edge-band error, vs exact f
  gradcheck.py          finite-difference checks
  serialize.py, cli.py  model files, command line
src/funcprox_ui/         the interface's API (FastAPI): lab, results catalog, saved models, job runner
ui/                     the interface's frontend (React, TypeScript, Vite)
experiments/            e0…e6, u1…u4, run_all.py, common.py (profiles quick / full)
tests/                  pytest suite
results/<exp>/          runs.csv + summary.md per experiment
docs/                   plan, research notes, protocol, results, figures/, screenshots/
```

## Development

```bash
pip install -e ".[dev,ui]"
pytest                               # gradient, Jacobian, engine and API checks
npm --prefix ui test                 # frontend logic
```

Browser tests and frontend development are described in [ui/README.md](ui/README.md), the git workflow in
[CONTRIBUTING.md](CONTRIBUTING.md). In VS Code, accept the recommended extensions and use
*Terminal → Run Task*.

The interface listens on the loopback address only and has no login, so do not expose it to a network. Lab runs
live in memory and end when the server stops; saved models stay in `runs/`.

## Documents

* [Findings so far](docs/FINDINGS.md): what the first full quick run shows, in plain words
* [Project plan](docs/PROJECT_PLAN.md): assumptions, hypotheses, team split, timeline
* [Research notes](docs/RESEARCH.md): literature behind every design choice, with links
* [Experiment protocol](docs/EXPERIMENTS.md): variables, fixed settings, metrics per experiment
* [Report outline](docs/REPORT_OUTLINE.md): mapping to the five report parts required by the supervisor
* [Results](docs/RESULTS.md): all tables and figures, generated by the experiments

## License

[MIT](LICENSE)
