"""Activation functions with analytic derivatives.

``df(z, a)`` receives both the pre-activation ``z`` and the activation ``a = f(z)``,
so each derivative can use whichever is cheaper (tanh and sigmoid reuse ``a``).
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


class Activation:
    name = "base"

    def f(self, z: Array) -> Array:  # pragma: no cover - interface
        raise NotImplementedError

    def df(self, z: Array, a: Array) -> Array:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class Identity(Activation):
    name = "identity"

    def f(self, z):
        return z

    def df(self, z, a):
        return np.ones_like(z)


class Tanh(Activation):
    name = "tanh"

    def f(self, z):
        return np.tanh(z)

    def df(self, z, a):
        return 1.0 - a * a


class Sigmoid(Activation):
    name = "sigmoid"

    def f(self, z):
        return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free logistic

    def df(self, z, a):
        return a * (1.0 - a)


class ReLU(Activation):
    name = "relu"

    def f(self, z):
        return np.maximum(z, 0.0)

    def df(self, z, a):
        return (z > 0.0).astype(z.dtype)


class SiLU(Activation):
    """SiLU / swish, x * sigmoid(x). Also the KAN base function b(x)."""
    name = "silu"

    def f(self, z):
        return z * 0.5 * (1.0 + np.tanh(0.5 * z))

    def df(self, z, a):
        s = 0.5 * (1.0 + np.tanh(0.5 * z))
        return s + z * s * (1.0 - s)


class Sine(Activation):
    """SIREN activation sin(w0 * z) (Sitzmann et al., 2020)."""

    def __init__(self, w0: float = 30.0):
        self.w0 = float(w0)
        self.name = f"sine(w0={self.w0:g})"

    def f(self, z):
        return np.sin(self.w0 * z)

    def df(self, z, a):
        return self.w0 * np.cos(self.w0 * z)


def get_activation(name: str, **kw) -> Activation:
    name = name.lower()
    table = {"identity": Identity, "linear": Identity, "tanh": Tanh, "sigmoid": Sigmoid,
             "relu": ReLU, "silu": SiLU, "swish": SiLU}
    if name in ("sine", "siren", "sin"):
        return Sine(kw.get("w0", 30.0))
    if name not in table:
        raise KeyError(f"unknown activation {name!r}")
    return table[name]()
