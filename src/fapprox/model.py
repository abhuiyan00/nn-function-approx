"""Network = ordered list of layers, with flat-parameter access and Jacobian.

Builders:
    build_mlp  - multilayer perceptron (optionally with Fourier features or SIREN)
    build_kan  - Kolmogorov-Arnold network
"""
from __future__ import annotations

import copy

import numpy as np

from .layers import Dense, FourierFeatures, KANLayer, Layer

Array = np.ndarray


class Network:
    def __init__(self, layers: list[Layer], config: dict | None = None):
        self.layers = layers
        self.config = config or {}

    # ------------------------------------------------------------------ params
    @property
    def params(self) -> list[Array]:
        return [p for layer in self.layers for p in layer.params]

    @property
    def l2_mask(self) -> list[bool]:
        return [m for layer in self.layers for m in layer.l2_mask()]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def get_flat(self) -> Array:
        return np.concatenate([p.ravel() for p in self.params]) if self.params else np.zeros(0)

    def set_flat(self, vec: Array) -> None:
        i = 0
        for p in self.params:
            p[...] = vec[i: i + p.size].reshape(p.shape)
            i += p.size

    def copy(self) -> "Network":
        return copy.deepcopy(self)

    # ----------------------------------------------------------------- compute
    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    predict = forward

    def backward(self, dy: Array) -> list[Array]:
        """Gradients of sum(dy * y) w.r.t. all params, ordered like ``params``."""
        grads_per_layer = []
        for layer in reversed(self.layers):
            dy, g = layer.backward(dy)
            grads_per_layer.append(g)
        return [g for gl in reversed(grads_per_layer) for g in gl]

    def jacobian(self, x: Array) -> tuple[Array, Array]:
        """Output y (N, 1) and Jacobian J = dy_n / dtheta, shape (N, n_params)."""
        y = self.forward(x)
        if y.shape[1] != 1:
            raise ValueError("jacobian() supports scalar-output networks only")
        dy = np.ones_like(y)
        blocks = []
        for layer in reversed(self.layers):
            dy, g = layer.backward(dy, per_sample=True)
            blocks.append(g)
        return y, np.concatenate([b for bl in reversed(blocks) for b in bl], axis=1)

    def input_gradient(self, x: Array) -> Array:
        """dy/dx, (N, n_in): used for derivative-accuracy diagnostics."""
        y = self.forward(x)
        dy = np.ones_like(y)
        for layer in reversed(self.layers):
            dy, _ = layer.backward(dy)
        return dy

    # --------------------------------------------------------------------- KAN
    def kan_refit(self, x: Array, new_grid: int | None = None) -> None:
        """Grid update / extension for every KAN layer, in order (see KANLayer.refit)."""
        for layer in self.layers:
            if isinstance(layer, KANLayer):
                layer.refit(x, new_grid=new_grid, adapt_range=True)
            x = layer.forward(x)

    def __repr__(self) -> str:
        body = " -> ".join(repr(layer) for layer in self.layers)
        return f"Network[{self.n_params} params]: {body}"


# ---------------------------------------------------------------------- builders
def build_mlp(n_in: int, hidden: list[int], activation: str = "tanh", n_out: int = 1,
              seed: int = 0, bias_init: str | None = None, fourier: tuple | None = None,
              w0: float = 30.0) -> Network:
    """MLP with ``hidden`` units per layer.

    activation 'sine' builds a SIREN (Sitzmann init, first-layer w0, hidden w0).
    bias_init  None -> 'zero'; use 'kink' for ReLU with randomly placed hinges.
    fourier    (m, scale) prepends a fixed random Fourier feature encoding.
    """
    rng = np.random.default_rng(seed)
    layers: list[Layer] = []
    width = n_in
    if fourier is not None:
        m, scale = fourier
        ff = FourierFeatures(n_in, m=int(m), scale=float(scale), rng=rng)
        layers.append(ff)
        width = ff.n_out
    act = activation.lower()
    for j, h in enumerate(hidden):
        if act in ("sine", "siren"):
            winit = "siren_first" if j == 0 else "siren_hidden"
            binit = bias_init or "siren"
        else:
            winit = "he" if act in ("relu", "silu") else "xavier"
            binit = bias_init or "zero"
            if binit == "kink" and j > 0:
                binit = "zero"  # hinge placement only makes sense on the input layer
        layers.append(Dense(width, h, act, rng, weight_init=winit, bias_init=binit, w0=w0))
        width = h
    out_init = "siren_hidden" if act in ("sine", "siren") else "xavier"
    layers.append(Dense(width, n_out, "identity", rng, weight_init=out_init, w0=w0))
    cfg = dict(arch="mlp", n_in=n_in, hidden=list(hidden), activation=activation, n_out=n_out,
               seed=seed, bias_init=bias_init, fourier=list(fourier) if fourier else None, w0=w0)
    return Network(layers, cfg)


def build_kan(widths: list[int], grid_size: int = 5, order: int = 3, seed: int = 0,
              noise_scale: float = 0.1) -> Network:
    """KAN with layer widths, e.g. [2, 5, 1] (input 2, one hidden layer of 5, output 1)."""
    rng = np.random.default_rng(seed)
    layers: list[Layer] = [KANLayer(a, b, grid_size, order, rng=rng, noise_scale=noise_scale)
                           for a, b in zip(widths[:-1], widths[1:])]
    cfg = dict(arch="kan", widths=list(widths), grid_size=grid_size, order=order, seed=seed,
               noise_scale=noise_scale)
    return Network(layers, cfg)


def mlp_param_count(n_in: int, hidden: list[int], n_out: int = 1, fourier_m: int = 0) -> int:
    widths = [2 * fourier_m if fourier_m else n_in] + list(hidden) + [n_out]
    return int(sum(a * b + b for a, b in zip(widths[:-1], widths[1:])))


def kan_param_count(widths: list[int], grid_size: int, order: int = 3) -> int:
    return int(sum(a * b * (grid_size + order) + a * b + b for a, b in zip(widths[:-1], widths[1:])))
