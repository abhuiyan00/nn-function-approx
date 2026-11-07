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
MAX_REDRAWS = 1000  # make_target refuses a function with under 5 % usable inputs, so this ends far sooner


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


def _sample_usable(target: Target, n: int, rng: np.random.Generator) -> Array:
    """Uniform in the box, minus the missing middle and wherever the function is undefined."""
    x = sample_box(target.domain, n, rng)
    for _ in range(MAX_REDRAWS):
        bad = target.in_gap(x) | ~np.isfinite(target(x).ravel())
        if not bad.any():
            return x
        x[bad] = sample_box(target.domain, int(bad.sum()), rng)
    raise ValueError(f"{target.key}: almost no input in the box can be sampled")


def grid(domain, n_total: int = 10_000) -> Array:
    """Dense evaluation grid: 10,000 points in 1-D, 100 x 100 in 2-D."""
    d = np.asarray(domain, dtype=float)
    if d.shape[0] == 1:
        return np.linspace(d[0, 0], d[0, 1], n_total).reshape(-1, 1)
    m = int(round(n_total ** (1.0 / d.shape[0])))
    axes = [np.linspace(lo, hi, m) for lo, hi in d]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in mesh], axis=1)


def _dataset_from_points(target: Target, rng: np.random.Generator) -> Dataset:
    """Measured data has no formula to score against, so a fifth of it is held out instead.

    Points inside the missing middle are never trained on: they show how well the hole was filled.
    """
    x, y = target.points[:, :1], target.points[:, 1:]
    hidden = target.in_gap(x)
    seen = rng.permutation(np.flatnonzero(~hidden))
    n_val = max(1, len(seen) // 5)
    val, train = seen[:n_val], seen[n_val:]
    test = np.concatenate([val, np.flatnonzero(hidden)])
    return Dataset(target, x[train], y[train], x[val], y[val], x[test], y[test], 0.0,
                   InputScaler.from_domain(target.domain), OutputScaler.fit(y[train]))


def make_dataset(target: Target, n_train: int = 2000, n_val: int = 500, sigma: float = 0.0,
                 data_seed: int = 0, n_test: int = 10_000, noise_fn=None) -> Dataset:
    """Uniform random training/validation inputs, Gaussian label noise, exact test grid.

    ``noise_fn(x) -> std`` gives heteroscedastic noise; otherwise ``sigma`` is constant.
    The data seed is independent of the model seed, so model seeds vary only the
    initialisation and the mini-batch order. Measured points ignore the sizes and the noise.
    """
    rng = np.random.default_rng(data_seed)
    if target.points is not None:
        return _dataset_from_points(target, rng)
    x_tr = _sample_usable(target, n_train, rng)
    x_va = _sample_usable(target, n_val, rng)

    def noisy(x):
        y = target(x)
        s = noise_fn(x).reshape(-1, 1) if noise_fn is not None else sigma
        return y + rng.normal(0.0, 1.0, size=y.shape) * s

    y_tr, y_va = noisy(x_tr), noisy(x_va)
    x_te = grid(target.domain, n_test)
    y_te = target(x_te)
    defined = np.isfinite(y_te).ravel()  # an undefined value cannot be scored
    return Dataset(target, x_tr, y_tr, x_va, y_va, x_te[defined], y_te[defined], sigma,
                   InputScaler.from_domain(target.domain), OutputScaler.fit(y_tr))
