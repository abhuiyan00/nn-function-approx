"""Classical approximators used as yardsticks (implemented here, NumPy only).

* Chebyshev least-squares polynomial (1-D; tensor product in 2-D)
* Cubic B-spline least squares on a uniform grid (1-D; tensor product in 2-D)
* Polynomial *interpolation* on equispaced or Chebyshev nodes (Runge phenomenon)
* Radial-basis-function network: Gaussian units, output weights by least squares (any dimension)

All work on inputs scaled to [-1, 1]. ``n_coef`` = number of fitted coefficients,
the counterpart of the network's parameter count.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C

from . import bspline

Array = np.ndarray


def _lstsq(A: Array, y: Array, ridge: float = 0.0) -> Array:
    if ridge:
        n = A.shape[1]
        return np.linalg.solve(A.T @ A + ridge * np.eye(n), A.T @ y)
    return np.linalg.lstsq(A, y, rcond=None)[0]


class ChebyshevLS:
    def __init__(self, degree: int):
        self.degree = degree

    def _design(self, u: Array) -> Array:
        if u.shape[1] == 1:
            return C.chebvander(u[:, 0], self.degree)
        if u.shape[1] == 2:
            return C.chebvander2d(u[:, 0], u[:, 1], [self.degree, self.degree])
        raise ValueError("1-D or 2-D only")

    def fit(self, u: Array, y: Array) -> "ChebyshevLS":
        self.coef = _lstsq(self._design(u), y.ravel())
        self.n_coef = self.coef.size
        return self

    def predict(self, u: Array) -> Array:
        return (self._design(u) @ self.coef).reshape(-1, 1)


class SplineLS:
    """Cubic B-spline least squares with ``n_basis`` functions per dimension."""

    def __init__(self, n_basis: int, order: int = 3, ridge: float = 1e-10):
        if n_basis <= order:
            raise ValueError("n_basis must exceed the spline order")
        self.n_basis, self.order, self.ridge = n_basis, order, ridge
        self.grid = n_basis - order

    def _design(self, u: Array) -> Array:
        d = u.shape[1]
        # tiny outward pad so that u = 1 falls inside the last interval
        knots = bspline.make_knots(np.full(d, -1.0 - 1e-9), np.full(d, 1.0 + 1e-9), self.grid, self.order)
        B = bspline.basis(u, knots, self.order)  # (N, d, M)
        if d == 1:
            return B[:, 0, :]
        if d == 2:
            return np.einsum("na,nb->nab", B[:, 0, :], B[:, 1, :]).reshape(len(u), -1)
        raise ValueError("1-D or 2-D only")

    def fit(self, u: Array, y: Array) -> "SplineLS":
        self.coef = _lstsq(self._design(u), y.ravel(), self.ridge)
        self.n_coef = self.coef.size
        return self

    def predict(self, u: Array) -> Array:
        return (self._design(u) @ self.coef).reshape(-1, 1)


class RBFNet:
    """Radial-basis-function network (Broomhead & Lowe, 1988; Moody & Darken, 1989).

        y(u) = sum_k w_k exp(-|u - c_k|^2 / (2 s^2)) + b

    The centres c_k come from k-means on the training inputs, the common width s is twice the
    mean distance from a centre to its nearest neighbour, and the output weights are solved by
    least squares: no gradient training at all. ``n_coef`` counts centres, weights and bias.
    """

    def __init__(self, n_units: int, seed: int = 0, ridge: float = 1e-8, steps: int = 20):
        self.n_units, self.seed, self.ridge, self.steps = n_units, seed, ridge, steps

    @staticmethod
    def _dist2(u: Array, c: Array) -> Array:
        return np.maximum((u * u).sum(axis=1)[:, None] + (c * c).sum(axis=1)[None, :] - 2.0 * u @ c.T, 0.0)

    def _design(self, u: Array) -> Array:
        return np.hstack([np.exp(-self._dist2(u, self.centres) / (2.0 * self.width ** 2)), np.ones((len(u), 1))])

    def fit(self, u: Array, y: Array) -> "RBFNet":
        k = min(self.n_units, len(u))
        c = u[np.random.default_rng(self.seed).choice(len(u), k, replace=False)].copy()
        for _ in range(self.steps):  # Lloyd's k-means
            nearest = self._dist2(u, c).argmin(axis=1)
            for j in range(k):
                members = u[nearest == j]
                if len(members):
                    c[j] = members.mean(axis=0)
        self.centres = c
        d2 = self._dist2(c, c)
        np.fill_diagonal(d2, np.inf)
        self.width = 2.0 * float(np.sqrt(d2.min(axis=1)).mean()) if k > 1 else 1.0
        self.coef = _lstsq(self._design(u), y.ravel(), self.ridge)
        self.n_coef = c.size + self.coef.size
        return self

    def predict(self, u: Array) -> Array:
        return (self._design(u) @ self.coef).reshape(-1, 1)


class PolyInterpolant:
    """Degree-n interpolating polynomial through n+1 nodes (1-D)."""

    def __init__(self, degree: int, nodes: str = "equispaced"):
        self.degree, self.nodes = degree, nodes

    def node_points(self) -> Array:
        n = self.degree
        if self.nodes == "equispaced":
            return np.linspace(-1.0, 1.0, n + 1)
        if self.nodes == "chebyshev":
            k = np.arange(n + 1)
            return np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
        raise KeyError(self.nodes)

    def fit_function(self, f_scaled) -> "PolyInterpolant":
        """``f_scaled(u)`` evaluates the target at scaled inputs u in [-1, 1]."""
        u = self.node_points()
        self.coef = np.linalg.solve(C.chebvander(u, self.degree), f_scaled(u.reshape(-1, 1)).ravel())
        self.n_coef = self.coef.size
        return self

    def predict(self, u: Array) -> Array:
        return C.chebval(u[:, 0], self.coef).reshape(-1, 1)
