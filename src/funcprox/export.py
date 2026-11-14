"""A trained network as one standalone Python file: the device, without the engine.

The file needs only NumPy and repeats the forward pass of each layer type, so it can be
handed to someone who never installs funcprox.
"""
from __future__ import annotations

import inspect

from . import bspline
from .data import InputScaler, OutputScaler
from .layers import Dense, FourierFeatures, KANLayer
from .model import Network

TEMPLATE = '''"""{title}

Exported from funcprox as a standalone function. Needs only NumPy.

    predict([{example}])   # one row per point, {dim} input value{plural} per row
"""
import numpy as np

X_LO, X_HI = np.array({lo}), np.array({hi})
Y_MEAN, Y_STD = {mean!r}, {std!r}
LAYERS = [
{layers}
]


def _logistic(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


ACTIVATIONS = {{
    "identity": lambda z: z,
    "tanh": np.tanh,
    "sigmoid": _logistic,
    "relu": lambda z: np.maximum(z, 0.0),
    "silu": lambda z: z * _logistic(z),
}}


{basis}

def predict(x):
    """The network's answer at each row of ``x``, in the units of the original function."""
    u = np.asarray(x, dtype=float).reshape(-1, X_LO.size)
    u = 2.0 * (u - X_LO) / (X_HI - X_LO) - 1.0
    for layer in LAYERS:
        if layer["kind"] == "dense":
            z = u @ np.array(layer["W"]) + np.array(layer["b"])
            u = np.sin(layer["w0"] * z) if layer["act"] == "sine" else ACTIVATIONS[layer["act"]](z)
        elif layer["kind"] == "fourier":
            proj = 2.0 * np.pi * (u @ np.array(layer["B"]))
            u = np.concatenate([np.sin(proj), np.cos(proj)], axis=1)
        else:  # KAN: a spline plus a SiLU on every edge
            b = basis(u, np.array(layer["knots"]), layer["k"])
            u = (np.einsum("nim,iom->no", b, np.array(layer["C"]))
                 + ACTIVATIONS["silu"](u) @ np.array(layer["wb"]) + np.array(layer["b"]))
    return (u * Y_STD + Y_MEAN).ravel()


if __name__ == "__main__":
    print(predict([{example}]))
'''


def _layer(layer) -> str:
    if isinstance(layer, Dense):
        act = layer.act.name.split("(")[0]
        fields = dict(kind="dense", act=act, w0=getattr(layer.act, "w0", 1.0), W=layer.W.tolist(), b=layer.b.tolist())
    elif isinstance(layer, FourierFeatures):
        fields = dict(kind="fourier", B=layer.B.tolist())
    elif isinstance(layer, KANLayer):
        fields = dict(kind="kan", k=layer.k, knots=layer.knots.tolist(), C=layer.C.tolist(), wb=layer.wb.tolist(), b=layer.b.tolist())
    else:
        raise TypeError(f"cannot export a {type(layer).__name__} layer")
    return f"    {fields!r},"


def export_source(net: Network, xs: InputScaler, ys: OutputScaler, title: str = "A trained network") -> str:
    """Python source of ``predict(x)`` for this network, with its weights and scaling written in."""
    middle = ", ".join(repr(float(v)) for v in 0.5 * (xs.lo + xs.hi))
    has_kan = any(isinstance(layer, KANLayer) for layer in net.layers)
    return TEMPLATE.format(
        title=title.replace('"""', "'''"), dim=xs.lo.size, plural="" if xs.lo.size == 1 else "s", example=f"[{middle}]",
        lo=xs.lo.tolist(), hi=xs.hi.tolist(), mean=float(ys.mean), std=float(ys.std),
        layers="\n".join(_layer(layer) for layer in net.layers),
        # the engine's own spline code, word for word; it annotates with the name Array
        basis="Array = np.ndarray\n\n\n" + inspect.getsource(bspline.basis) + "\n" if has_kan else "",
    )
