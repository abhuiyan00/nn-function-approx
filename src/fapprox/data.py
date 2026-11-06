"""Data sets, pre-processing and post-processing.

Pre-processing
    * inputs:  min-max scaling of the *known* domain box to [-1, 1] per dimension
               (the box is a design assumption, not estimated from data, so the
               same scaling is valid for test and extrapolation points);
    * outputs: z-score with the mean and standard deviation of the *training labels*.

Post-processing
    * network outputs are mapped back with the inverse z-score;
    * metrics are always computed in original units against the exact function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .functions import Target

Array = np.ndarray


@dataclass
class InputScaler:
    lo: Array
    hi: Array

    @classmethod
    def from_domain(cls, domain) -> "InputScaler":
        d = np.asarray(domain, dtype=float)
        return cls(d[:, 0], d[:, 1])

    def transform(self, x: Array) -> Array:
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def inverse(self, u: Array) -> Array:
        return (u + 1.0) * 0.5 * (self.hi - self.lo) + self.lo


@dataclass
class OutputScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: Array) -> "OutputScaler":
        s = float(np.std(y))
        return cls(float(np.mean(y)), s if s > 0 else 1.0)

    def transform(self, y: Array) -> Array:
        return (y - self.mean) / self.std

    def inverse(self, z: Array) -> Array:
        return z * self.std + self.mean


@dataclass
class Dataset:
    target: Target
    x_train: Array
    y_train: Array          # noisy labels
    x_val: Array
    y_val: Array            # noisy labels
    x_test: Array           # dense grid
    y_test: Array           # exact f on the grid
    sigma: float
    xs: InputScaler
    ys: OutputScaler

    # scaled views used by the engine
    @property
    def u_train(self) -> Array:
        return self.xs.transform(self.x_train)

    @property
    def u_val(self) -> Array:
        return self.xs.transform(self.x_val)

    @property
    def u_test(self) -> Array:
        return self.xs.transform(self.x_test)

    @property
    def z_train(self) -> Array:
        return self.ys.transform(self.y_train)

    @property
    def z_val(self) -> Array:
        return self.ys.transform(self.y_val)


def sample_box(domain, n: int, rng: np.random.Generator) -> Array:
    d = np.asarray(domain, dtype=float)
    return rng.uniform(d[:, 0], d[:, 1], size=(n, d.shape[0]))


def grid(domain, n_total: int = 10_000) -> Array:
    """Dense evaluation grid: 10,000 points in 1-D, 100 x 100 in 2-D."""
    d = np.asarray(domain, dtype=float)
    if d.shape[0] == 1:
        return np.linspace(d[0, 0], d[0, 1], n_total).reshape(-1, 1)
    m = int(round(n_total ** (1.0 / d.shape[0])))
    axes = [np.linspace(lo, hi, m) for lo, hi in d]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in mesh], axis=1)


def make_dataset(target: Target, n_train: int = 2000, n_val: int = 500, sigma: float = 0.0,
                 data_seed: int = 0, n_test: int = 10_000, noise_fn=None) -> Dataset:
    """Uniform random training/validation inputs, Gaussian label noise, exact test grid.

    ``noise_fn(x) -> std`` gives heteroscedastic noise; otherwise ``sigma`` is constant.
    The data seed is independent of the model seed, so model seeds vary only the
    initialisation and the mini-batch order.
    """
    rng = np.random.default_rng(data_seed)
    x_tr = sample_box(target.domain, n_train, rng)
    x_va = sample_box(target.domain, n_val, rng)

    def noisy(x):
        y = target(x)
        s = noise_fn(x).reshape(-1, 1) if noise_fn is not None else sigma
        return y + rng.normal(0.0, 1.0, size=y.shape) * s

    y_tr, y_va = noisy(x_tr), noisy(x_va)
    x_te = grid(target.domain, n_test)
    return Dataset(target, x_tr, y_tr, x_va, y_va, x_te, target(x_te), sigma,
                   InputScaler.from_domain(target.domain), OutputScaler.fit(y_tr))
