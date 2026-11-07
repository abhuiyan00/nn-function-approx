"""Uniform cubic (order-k) B-spline basis, vectorised Cox-de Boor recursion.

Used by the KAN layer and by the spline least-squares baseline, so the network and
its classical competitor share one implementation.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def make_knots(lo, hi, grid_size: int, order: int) -> Array:
    """Uniform knots on [lo, hi] with ``order`` extra knots on each side.

    ``lo`` and ``hi`` may be arrays (one interval per input feature); the result then
    has shape (n_features, grid_size + 2 * order + 1).
    """
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    h = (hi - lo) / grid_size
    steps = np.arange(-order, grid_size + order + 1, dtype=float)
    return lo[:, None] + h[:, None] * steps[None, :]


def basis(x: Array, knots: Array, order: int, with_lower: bool = False):
    """B-spline basis values.

    x      : (N, d) points
    knots  : (d, T) knot vectors, one per feature
    returns: B of shape (N, d, T - order - 1); with ``with_lower`` also the order-1
             basis of shape (N, d, T - order), needed for the derivative.
    """
    x = x[:, :, None]
    t = knots[None, :, :]
    b = ((x >= t[..., :-1]) & (x < t[..., 1:])).astype(float)
    lower = b
    for p in range(1, order + 1):
        lower = b
        left = (x - t[..., : -(p + 1)]) / (t[..., p:-1] - t[..., : -(p + 1)]) * b[..., :-1]
        right = (t[..., p + 1:] - x) / (t[..., p + 1:] - t[..., 1:-p]) * b[..., 1:]
        b = left + right
    if with_lower:
        return b, lower
    return b


def basis_and_derivative(x: Array, knots: Array, order: int):
    """Basis (N, d, M) and its derivative d/dx (N, d, M) for *uniform* knots."""
    b, lower = basis(x, knots, order, with_lower=True)
    h = (knots[:, 1] - knots[:, 0])[None, :, None]
    db = (lower[..., :-1] - lower[..., 1:]) / h
    return b, db
