"""Shared experiment machinery: profiles, one-run function, parallel map, outputs.

Every experiment script
    1. builds a list of run specs (plain dicts),
    2. runs them with ``run_many`` (2 worker processes by default),
    3. writes results/<exp>/runs.csv + summary.md and docs/figures/<exp>_*.png.

Profiles
    quick : 2 model seeds, 2,000 train / 500 val points - minutes, for development
    full  : 5 model seeds, 20,000 train / 5,000 val points - hours, for the report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fapprox.data import make_dataset  # noqa: E402
from fapprox.functions import get_target  # noqa: E402
from fapprox.metrics import evaluate, predict_original  # noqa: E402
from fapprox.model import build_kan, build_mlp  # noqa: E402
from fapprox.train import TrainConfig, fit  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "results"

PROFILES = {
    "quick": dict(seeds=(1, 2), n_train=2000, n_val=500, epochs=600, patience=50, lm_iters=300, scale=1),
    "full": dict(seeds=(1, 2, 3, 4, 5), n_train=20000, n_val=5000, epochs=1000, patience=50,
                 lm_iters=500, scale=2),
}

DEFAULTS = dict(arch="mlp", hidden=(32, 32), act="tanh", bias_init=None, fourier=None, w0=30.0,
                widths=None, grid=5, optimizer="adam", lr=1e-3, lr_schedule="constant", batch=64, epochs=600, patience=50,
                l2=0.0, n_train=2000, n_val=500, sigma=0.0, seed=1, data_seed=0,
                kan_grid_schedule=None, restore_best=True, return_pred=False, return_hist=False)


def parse_profile(description: str) -> tuple[str, dict]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    p.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    a = p.parse_args()
    prof = dict(PROFILES[a.profile])
    prof["name"], prof["workers"] = a.profile, a.workers
    return a.profile, prof


def build(spec: dict, n_in: int):
    if spec["arch"] == "kan":
        widths = list(spec["widths"] or (n_in, 5, 1))
        return build_kan(widths, spec["grid"], seed=spec["seed"])
    return build_mlp(n_in, list(spec["hidden"]), spec["act"], seed=spec["seed"],
                     bias_init=spec["bias_init"], fourier=spec["fourier"], w0=spec["w0"])


def run_one(spec: dict) -> dict:
    """Train one network described by ``spec`` and return a flat result row."""
    s = dict(DEFAULTS)
    s.update(spec)
    target = get_target(s["target"])
    ds = make_dataset(target, s["n_train"], s["n_val"], s["sigma"], s["data_seed"])
    net = build(s, target.dim)
    cfg = TrainConfig(optimizer=s["optimizer"], lr=s["lr"], lr_schedule=s["lr_schedule"], batch=s["batch"], epochs=s["epochs"],
                      patience=s["patience"], l2=s["l2"], seed=s["seed"], restore_best=s["restore_best"],
                      kan_grid_schedule=dict(s["kan_grid_schedule"] or {}))
    t0 = time.perf_counter()
    hist = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg)
    m = evaluate(net, ds)
    row = {k: v for k, v in s.items() if k not in ("return_pred", "return_hist")}
    for k in ("hidden", "widths", "fourier", "kan_grid_schedule"):
        row[k] = json.dumps(list(row[k]) if isinstance(row[k], tuple) else row[k])
    row.update(m)
    row.update(seconds=time.perf_counter() - t0, best_epoch=hist.best_epoch, epochs_run=len(hist.epoch),
               updates=hist.updates[-1] if hist.updates else 0, stop=hist.stop_reason)
    if s["return_pred"]:
        row["_x"] = ds.x_test
        row["_pred"] = predict_original(net, ds, ds.x_test)
        row["_true"] = ds.y_test
    if s["return_hist"]:
        sd = ds.ys.std ** 2  # scaled MSE -> original units
        row["_hist"] = dict(epoch=hist.epoch, train_mse=[v * sd for v in hist.train_mse],
                            val_mse=[v * sd for v in hist.val_mse], wall=hist.wall,
                            updates=hist.updates, mu=hist.mu)
    return row


def run_many(specs: list[dict], workers: int = 2, fn=run_one, label: str = "") -> list[dict]:
    t0 = time.perf_counter()
    if workers > 1 and len(specs) > 1:
        import multiprocessing as mp
        # Windows has no fork; spawn re-imports the script, which every experiment guards with __main__
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(workers) as pool:
            out = pool.map(fn, specs, chunksize=1)
    else:
        out = [fn(s) for s in specs]
    print(f"  {label or fn.__name__}: {len(specs)} runs in {time.perf_counter() - t0:.0f} s")
    return out


def save_rows(exp: str, rows: list[dict], name: str = "runs") -> Path:
    d = RES_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    keys: list[str] = []
    for r in clean:
        keys += [k for k in r if k not in keys]
    path = d / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in clean:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def save_json(exp: str, obj, name: str) -> Path:
    d = RES_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path


def write_summary(exp: str, title: str, body: str, profile: str) -> Path:
    d = RES_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    path = d / "summary.md"
    stamp = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(f"## {title}\n\n_Profile: **{profile}** · generated {stamp}_\n\n{body.strip()}\n", encoding="utf-8")
    return path


def fig(exp: str, name: str) -> Path:
    return FIG_DIR / f"{exp}_{name}.png"


def agg(rows: list[dict], keys: tuple, metric: str = "rmse") -> dict:
    """Group rows by ``keys`` -> dict(key tuple -> (mean, min, max, n))."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r[metric])
    return {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) for k, v in groups.items()}


def gmean(values) -> float:
    v = np.asarray(list(values), dtype=float)
    return float(np.exp(np.mean(np.log(v))))


def fmt(x: float) -> str:
    if x == 0 or not np.isfinite(x):
        return str(x)
    if abs(x) >= 1e-2 and abs(x) < 1e3:
        return f"{x:.3g}"
    return f"{x:.2e}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)
