"""Layers of the engine: Dense (MLP), FourierFeatures (input encoding), KANLayer.

Every layer implements

    forward(x)                      -> y, caches what backward needs
    backward(dy, per_sample=False)  -> (dx, grads)

``grads`` is a list aligned with ``layer.params``. With ``per_sample=True`` each
entry has shape (N, param.size): the per-sample gradients that form the Jacobian
used by Levenberg-Marquardt. ``dx`` is already per-sample in both modes.
"""
from __future__ import annotations

import numpy as np

from . import bspline
from .activations import Activation, SiLU, get_activation

Array = np.ndarray


class Layer:
    params: list[Array]
    trainable = True

    def forward(self, x: Array) -> Array:  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, dy: Array, per_sample: bool = False):  # pragma: no cover
        raise NotImplementedError

    def l2_mask(self) -> list[bool]:
        """Which params are penalised by weight decay (weights yes, biases no)."""
        return [False] * len(self.params)

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))


# --------------------------------------------------------------------------- Dense
class Dense(Layer):
    """y = act(x W + b).

    weight_init: 'xavier' (Glorot uniform), 'he' (He normal), 'siren_first', 'siren_hidden'
    bias_init  : 'zero', or 'kink' (ReLU: place each unit's kink at a random point of the
                 input box, so the hinges cover the domain instead of all sitting at 0)
    """

    def __init__(self, n_in: int, n_out: int, activation: Activation | str = "tanh",
                 rng: np.random.Generator | None = None, weight_init: str = "xavier",
                 bias_init: str = "zero", w0: float = 30.0):
        rng = rng or np.random.default_rng(0)
        self.act = get_activation(activation, w0=w0) if isinstance(activation, str) else activation
        self.n_in, self.n_out = n_in, n_out
        if weight_init == "xavier":
            lim = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-lim, lim, (n_in, n_out))
        elif weight_init == "he":
            W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
        elif weight_init == "siren_first":
            W = rng.uniform(-1.0 / n_in, 1.0 / n_in, (n_in, n_out))
        elif weight_init == "siren_hidden":
            lim = np.sqrt(6.0 / n_in) / w0
            W = rng.uniform(-lim, lim, (n_in, n_out))
        else:
            raise KeyError(f"unknown weight_init {weight_init!r}")
        if bias_init == "zero":
            b = np.zeros(n_out)
        elif bias_init == "kink":
            centres = rng.uniform(-1.0, 1.0, (n_out, n_in))  # inputs are scaled to [-1, 1]
            b = -np.einsum("oi,io->o", centres, W)
        elif bias_init == "siren":
            b = rng.uniform(-1.0 / np.sqrt(n_in), 1.0 / np.sqrt(n_in), n_out)
        else:
            raise KeyError(f"unknown bias_init {bias_init!r}")
        self.W, self.b = W, b
        self.params = [self.W, self.b]
        self.weight_init, self.bias_init = weight_init, bias_init

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.act.f(self.z)
        return self.a

    def backward(self, dy, per_sample=False):
        dz = dy * self.act.df(self.z, self.a)
        dx = dz @ self.W.T
        if per_sample:
            n = dz.shape[0]
            gW = (self.x[:, :, None] * dz[:, None, :]).reshape(n, -1)
            return dx, [gW, dz]
        return dx, [self.x.T @ dz, dz.sum(axis=0)]

    def l2_mask(self):
        return [True, False]

    def __repr__(self):
        return f"Dense({self.n_in}->{self.n_out}, {self.act})"


# ----------------------------------------------------------------- FourierFeatures
class FourierFeatures(Layer):
    """Random Fourier feature encoding (Tancik et al., 2020), fixed, not trained.

    gamma(x) = [sin(2 pi x B), cos(2 pi x B)],  B ~ N(0, scale^2), shape (n_in, m)
    """
    trainable = False

    def __init__(self, n_in: int, m: int = 32, scale: float = 2.0,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.B = rng.normal(0.0, scale, (n_in, m))
        self.n_in, self.m, self.scale = n_in, m, scale
        self.n_out = 2 * m
        self.params = []

    def forward(self, x):
        self.proj = 2.0 * np.pi * (x @ self.B)
        self.s, self.c = np.sin(self.proj), np.cos(self.proj)
        return np.concatenate([self.s, self.c], axis=1)

    def backward(self, dy, per_sample=False):
        ds, dc = dy[:, : self.m], dy[:, self.m:]
        dproj = ds * self.c - dc * self.s
        return (2.0 * np.pi) * dproj @ self.B.T, []

    def __repr__(self):
        return f"FourierFeatures({self.n_in}->{self.n_out}, scale={self.scale:g})"


# ----------------------------------------------------------------------- KANLayer
class KANLayer(Layer):
    """Kolmogorov-Arnold layer (Liu et al., 2024): a learnable function on every edge.

        y_o = sum_i [ wb_io * silu(x_i) + sum_m C_iom * B_m(x_i) ] + b_o

    B_m are uniform B-splines of order ``k`` on ``G`` grid intervals, one grid per input.
    ``refit`` re-expresses the splines on a new grid (range update or grid extension)
    by least squares, so the function is preserved and training can continue.
    """

    def __init__(self, n_in: int, n_out: int, grid_size: int = 5, order: int = 3,
                 grid_range=(-1.0, 1.0), rng: np.random.Generator | None = None,
                 noise_scale: float = 0.1):
        rng = rng or np.random.default_rng(0)
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, grid_size, order
        lo = np.full(n_in, float(grid_range[0]))
        hi = np.full(n_in, float(grid_range[1]))
        self.knots = bspline.make_knots(lo, hi, grid_size, order)
        M = grid_size + order
        self.C = rng.normal(0.0, noise_scale / np.sqrt(n_in), (n_in, n_out, M))
        lim = np.sqrt(6.0 / (n_in + n_out))
        self.wb = rng.uniform(-lim, lim, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.params = [self.C, self.wb, self.b]
        self.base = SiLU()

    @property
    def n_basis(self) -> int:
        return self.G + self.k

    def _spline_matrix(self):
        # (n_in * M, n_out) with row index i * M + m, matching B.reshape(N, n_in * M)
        return self.C.transpose(0, 2, 1).reshape(self.n_in * self.n_basis, self.n_out)

    def forward(self, x):
        self.x = x
        self.B, self.dB = bspline.basis_and_derivative(x, self.knots, self.k)
        self.s = self.base.f(x)
        n = x.shape[0]
        return self.B.reshape(n, -1) @ self._spline_matrix() + self.s @ self.wb + self.b

    def backward(self, dy, per_sample=False):
        n = dy.shape[0]
        # dx: spline slope + base slope
        tmp = np.einsum("no,iom->nim", dy, self.C)
        dx = (tmp * self.dB).sum(axis=2) + (dy @ self.wb.T) * self.base.df(self.x, self.s)
        if per_sample:
            gC = np.einsum("nim,no->niom", self.B, dy).reshape(n, -1)
            gwb = np.einsum("ni,no->nio", self.s, dy).reshape(n, -1)
            return dx, [gC, gwb, dy]
        gC = (self.B.reshape(n, -1).T @ dy).reshape(self.n_in, self.n_basis, self.n_out)
        return dx, [gC.transpose(0, 2, 1), self.s.T @ dy, dy.sum(axis=0)]

    def l2_mask(self):
        return [True, True, False]

    def refit(self, x: Array, new_grid: int | None = None, adapt_range: bool = True,
              margin: float = 0.05, ridge: float = 1e-8):
        """Least-squares transfer of the spline part to a new grid (same base part)."""
        B_old = bspline.basis(x, self.knots, self.k)
        S = np.einsum("nim,iom->nio", B_old, self.C)  # spline value per edge
        G = new_grid or self.G
        if adapt_range:
            lo, hi = x.min(axis=0), x.max(axis=0)
            pad = margin * np.maximum(hi - lo, 1e-3)
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = self.knots[:, self.k], self.knots[:, -self.k - 1]
        knots = bspline.make_knots(lo, hi, G, self.k)
        B_new = bspline.basis(x, knots, self.k)
        M = G + self.k
        C = np.empty((self.n_in, self.n_out, M))
        for i in range(self.n_in):
            A = B_new[:, i, :]
            lhs = A.T @ A + ridge * np.eye(M)
            C[i] = np.linalg.solve(lhs, A.T @ S[:, i, :]).T
        self.knots, self.G, self.C = knots, G, C
        self.params = [self.C, self.wb, self.b]

    def __repr__(self):
        return f"KANLayer({self.n_in}->{self.n_out}, G={self.G}, k={self.k})"
