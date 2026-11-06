"""First-order optimisers updating parameter arrays in place."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


class SGD:
    def __init__(self, params: list[Array], lr: float = 1e-2, momentum: float = 0.0,
                 nesterov: bool = False):
        self.params, self.lr, self.mu, self.nesterov = params, lr, momentum, nesterov
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[Array]) -> None:
        for p, g, v in zip(self.params, grads, self.v):
            if self.mu:
                v *= self.mu
                v += g
                p -= self.lr * (g + self.mu * v if self.nesterov else v)
            else:
                p -= self.lr * g


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, params: list[Array], lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[Array]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def make_optimizer(name: str, params: list[Array], lr: float, momentum: float = 0.9):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr)
    if name == "sgd":
        return SGD(params, lr)
    if name in ("momentum", "sgdm"):
        return SGD(params, lr, momentum=momentum)
    if name == "nesterov":
        return SGD(params, lr, momentum=momentum, nesterov=True)
    raise KeyError(f"unknown optimizer {name!r}")
