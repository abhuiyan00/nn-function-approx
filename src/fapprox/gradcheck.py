"""Central finite-difference checks of the analytic gradient and Jacobian."""
from __future__ import annotations

import numpy as np

from .model import Network


def loss_and_grad(net: Network, x, z):
    y = net.forward(x)
    e = y - z
    grads = net.backward((2.0 / len(x)) * e)
    return float(np.mean(e * e)), np.concatenate([g.ravel() for g in grads])


def gradient_error(net: Network, x, z, eps: float = 1e-6, n_probe: int = 60, seed: int = 0) -> float:
    """Scale-aware error max_i |num_i - g_i| / max_j |g_j| on random coordinates.

    Normalising by the largest gradient entry (not entry-wise) avoids false alarms on
    coordinates whose true gradient is ~1e-10, where finite differences are pure noise
    (e.g. KAN spline coefficients with almost no data in their support).
    ReLU networks fail the check only if a probe moves a pre-activation across 0.
    """
    _, g = loss_and_grad(net, x, z)
    scale = max(float(np.max(np.abs(g))), 1e-12)
    theta = net.get_flat()
    rng = np.random.default_rng(seed)
    idx = rng.choice(theta.size, size=min(n_probe, theta.size), replace=False)
    worst = 0.0
    for i in idx:
        t = theta.copy()
        t[i] += eps
        net.set_flat(t)
        lp = float(np.mean((net.forward(x) - z) ** 2))
        t[i] -= 2 * eps
        net.set_flat(t)
        lm = float(np.mean((net.forward(x) - z) ** 2))
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - g[i]) / scale)
    net.set_flat(theta)
    return worst


def jacobian_error(net: Network, x, eps: float = 1e-6, n_probe: int = 30, seed: int = 0) -> float:
    """Max abs error between analytic Jacobian columns and finite differences."""
    _, J = net.jacobian(x)
    theta = net.get_flat()
    rng = np.random.default_rng(seed)
    idx = rng.choice(theta.size, size=min(n_probe, theta.size), replace=False)
    worst = 0.0
    for i in idx:
        t = theta.copy()
        t[i] += eps
        net.set_flat(t)
        yp = net.forward(x).ravel()
        t[i] -= 2 * eps
        net.set_flat(t)
        ym = net.forward(x).ravel()
        worst = max(worst, float(np.max(np.abs((yp - ym) / (2 * eps) - J[:, i]))))
    net.set_flat(theta)
    return worst
