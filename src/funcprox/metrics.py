"""Error measures in original units against the exact target on the test grid."""
from __future__ import annotations

import numpy as np

from .data import Dataset

Array = np.ndarray


def regression_metrics(y_hat: Array, y: Array, u: Array | None = None,
                       edge: float = 0.9) -> dict:
    """RMSE, MAE, max error, R^2, NRMSE (= RMSE / std f) and edge/interior split.

    ``u`` are inputs scaled to [-1, 1]; a point is in the edge band when any
    coordinate has |u| > edge (outer 5 % of each side of the box).
    """
    e = (y_hat - y).ravel()
    std = float(np.std(y))
    out = {
        "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))),
        "max_err": float(np.max(np.abs(e))),
        "r2": float(1.0 - np.mean(e * e) / (std * std)) if std > 0 else float("nan"),
        "nrmse": float(np.sqrt(np.mean(e * e)) / std) if std > 0 else float("nan"),
    }
    if u is not None:
        band = np.any(np.abs(u) > edge, axis=1)
        if band.any() and (~band).any():
            out["rmse_edge"] = float(np.sqrt(np.mean(e[band] ** 2)))
            out["rmse_interior"] = float(np.sqrt(np.mean(e[~band] ** 2)))
            out["argmax_in_edge"] = bool(band[np.argmax(np.abs(e))])
    return out


def predict_original(net, ds: Dataset, x: Array) -> Array:
    """Pre-process x, run the network, post-process the output."""
    return ds.ys.inverse(net.forward(ds.xs.transform(x)))


def evaluate(net, ds: Dataset) -> dict:
    y_hat = predict_original(net, ds, ds.x_test)
    m = regression_metrics(y_hat, ds.y_test, ds.u_test)
    # validation error against noisy labels, original units (what a user without f sees)
    e_val = predict_original(net, ds, ds.x_val) - ds.y_val
    m["val_mse_noisy"] = float(np.mean(e_val ** 2))
    m["test_mse"] = m["rmse"] ** 2
    m["n_params"] = int(net.n_params)
    return m
