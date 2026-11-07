"""Save / load a trained network with its pre/post-processing parameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import InputScaler, OutputScaler
from .layers import KANLayer
from .model import Network, build_kan, build_mlp


def save(path: str | Path, net: Network, xs: InputScaler, ys: OutputScaler, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(net.config)
    kan_grids = [layer.G for layer in net.layers if isinstance(layer, KANLayer)]
    if kan_grids:
        cfg["grid_sizes"] = kan_grids
    header = dict(config=cfg, x_lo=xs.lo.tolist(), x_hi=xs.hi.tolist(),
                  y_mean=ys.mean, y_std=ys.std, meta=meta or {})
    arrays = {"theta": net.get_flat()}
    for j, layer in enumerate(net.layers):
        if isinstance(layer, KANLayer):
            arrays[f"knots_{j}"] = layer.knots
        if hasattr(layer, "B") and not isinstance(layer, KANLayer):
            arrays[f"fourier_B_{j}"] = layer.B
    np.savez(path, header=json.dumps(header), **arrays)
    return path


def load(path: str | Path):
    data = np.load(Path(path), allow_pickle=False)
    header = json.loads(str(data["header"]))
    cfg = header["config"]
    if cfg["arch"] == "mlp":
        net = build_mlp(cfg["n_in"], cfg["hidden"], cfg["activation"], cfg["n_out"], cfg["seed"],
                        cfg.get("bias_init"), tuple(cfg["fourier"]) if cfg.get("fourier") else None,
                        cfg.get("w0", 30.0))
    else:
        net = build_kan(cfg["widths"], cfg["grid_size"], cfg["order"], cfg["seed"], cfg["noise_scale"])
        kan_layers = [layer for layer in net.layers if isinstance(layer, KANLayer)]
        for layer, g in zip(kan_layers, cfg.get("grid_sizes", [])):
            if g != layer.G:  # rebuild arrays at the saved grid size
                layer.G = g
                layer.C = np.zeros((layer.n_in, layer.n_out, g + layer.k))
                layer.params = [layer.C, layer.wb, layer.b]
    for j, layer in enumerate(net.layers):
        if f"knots_{j}" in data:
            layer.knots = data[f"knots_{j}"]
        if f"fourier_B_{j}" in data:
            layer.B = data[f"fourier_B_{j}"]
    net.set_flat(data["theta"])
    xs = InputScaler(np.array(header["x_lo"]), np.array(header["x_hi"]))
    ys = OutputScaler(header["y_mean"], header["y_std"])
    return net, xs, ys, header.get("meta", {})
