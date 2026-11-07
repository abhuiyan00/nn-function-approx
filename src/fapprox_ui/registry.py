"""Saved models: the same .npz files the CLI writes (runs/ is ignored by git)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from fapprox import serialize
from fapprox.functions import TARGETS

from .lab import LabRun, clean, make_grid, round_array
from .schemas import MODEL_NAME


def path_for(directory: Path, name: str) -> Path | None:
    return directory / f"{name}.npz" if re.fullmatch(MODEL_NAME, name) else None


def save(path: Path, run: LabRun) -> None:
    serialize.save(path, run.net, run.xs, run.ys,
                   meta=clean(dict(target=run.spec.target, label=run.spec.describe(), metrics=run.final["metrics"],
                                   train=run.config.to_dict(), spec=run.spec.model_dump())))


def _describe(path: Path) -> dict:
    stat = path.stat()
    base = dict(name=path.stem, size=stat.st_size, modified=stat.st_mtime)
    try:
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
    except (OSError, ValueError, KeyError) as exc:
        return dict(base, error=f"unreadable: {exc}")
    meta, config = header.get("meta", {}), header.get("config", {})
    metrics = meta.get("metrics", {})
    return dict(base, target=meta.get("target"), arch=config.get("arch"), label=meta.get("label"), config=config,
                rmse=metrics.get("rmse"), max_err=metrics.get("max_err"), r2=metrics.get("r2"), n_params=metrics.get("n_params"))


def list_models(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    return clean(sorted((_describe(p) for p in directory.glob("*.npz")), key=lambda m: -m["modified"]))


def model_detail(path: Path) -> dict:
    net, xs, ys, meta = serialize.load(path)
    out = _describe(path)
    target = TARGETS.get(meta.get("target"))
    if target is None:
        return clean(dict(out, grid=None, pred=None, residual=None))
    x, grid = make_grid(target)
    pred = ys.inverse(net.forward(xs.transform(x))).ravel()
    return clean(dict(out, formula=target.formula, dim=target.dim, grid=grid, pred=round_array(pred),
                      residual=round_array(pred - target(x).ravel(), 4)))


def predict(path: Path, inputs: list[list[float]]) -> dict:
    """Raises ValueError when the inputs do not fit the model."""
    net, xs, ys, meta = serialize.load(path)
    x = np.asarray(inputs, dtype=float)
    dim = xs.lo.size
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"this model takes {dim} input value(s) per row")
    y = ys.inverse(net.forward(xs.transform(x))).ravel()
    target = TARGETS.get(meta.get("target"))
    exact = target(x).ravel() if target else None
    return clean(dict(y_hat=y.tolist(), exact=exact.tolist() if exact is not None else None,
                      abs_error=np.abs(y - exact).tolist() if exact is not None else None,
                      outside_box=(np.any(x < xs.lo, axis=1) | np.any(x > xs.hi, axis=1)).tolist()))
