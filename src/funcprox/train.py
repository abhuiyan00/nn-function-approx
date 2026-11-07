"""Training: mini-batch first-order loop and full-batch Levenberg-Marquardt.

Loss: mean squared error on *scaled* labels (z-score), plus optional L2 on weights.
Early stopping monitors the validation MSE; the best weights are restored.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .model import Network
from .optim import make_optimizer

Array = np.ndarray


@dataclass
class TrainConfig:
    optimizer: str = "adam"        # adam | sgd | momentum | nesterov | lm
    lr: float = 1e-3
    lr_schedule: str = "constant"  # constant | cosine (anneal to lr_min over `epochs`)
    lr_min: float = 1e-5
    batch: int = 64                # 0 or >= N means full batch
    epochs: int = 300              # for LM: maximum iterations
    patience: int = 50             # epochs without val improvement (0 = off)
    restore_best: bool = True
    l2: float = 0.0
    momentum: float = 0.9
    seed: int = 0                  # mini-batch order
    # Levenberg-Marquardt
    mu: float = 1e-3
    mu_inc: float = 10.0
    mu_dec: float = 0.1
    mu_max: float = 1e10
    lm_max_points: int = 4000      # LM is full batch; larger sets are subsampled
    # KAN schedule {epoch: new_grid_size}; grid range is also updated at these epochs
    kan_grid_schedule: dict = field(default_factory=dict)
    target_mse: float = 0.0        # stop when train MSE (scaled) falls below this

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class History:
    epoch: list = field(default_factory=list)
    train_mse: list = field(default_factory=list)   # scaled units
    val_mse: list = field(default_factory=list)     # scaled units
    wall: list = field(default_factory=list)        # seconds since start
    updates: list = field(default_factory=list)     # cumulative parameter updates
    mu: list = field(default_factory=list)          # LM damping after each iteration
    best_epoch: int = -1
    best_val: float = np.inf
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def mse(net: Network, x: Array, z: Array) -> float:
    e = net.forward(x) - z
    return float(np.mean(e * e))


def _loss_grads(net: Network, x: Array, z: Array, l2: float):
    y = net.forward(x)
    e = y - z
    grads = net.backward((2.0 / len(x)) * e)
    if l2:
        for g, p, m in zip(grads, net.params, net.l2_mask):
            if m:
                g += 2.0 * l2 * p
    return float(np.mean(e * e)), grads


def fit(net: Network, x_tr: Array, z_tr: Array, x_va: Array, z_va: Array,
        cfg: TrainConfig, callback: Callable[[int, Network], object] | None = None,
        history: History | None = None) -> History:
    """Train ``net``. ``callback(epoch, net)`` runs after every epoch; a truthy return stops
    training (best weights are still restored). ``history`` lets a caller read progress live."""
    if cfg.optimizer.lower() == "lm":
        return fit_lm(net, x_tr, z_tr, x_va, z_va, cfg, callback, history)

    rng = np.random.default_rng([cfg.seed, 7])
    opt = make_optimizer(cfg.optimizer, net.params, cfg.lr, cfg.momentum)
    n = len(x_tr)
    bs = n if cfg.batch <= 0 or cfg.batch >= n else cfg.batch
    hist = History() if history is None else history
    best = net.get_flat()
    wait, updates, t0 = 0, 0, time.perf_counter()
    if callback:
        callback(0, net)
    for ep in range(1, cfg.epochs + 1):
        if ep in cfg.kan_grid_schedule:
            net.kan_refit(x_tr, new_grid=cfg.kan_grid_schedule[ep])
            opt = make_optimizer(cfg.optimizer, net.params, cfg.lr, cfg.momentum)
            best = net.get_flat()  # parameter vector changed size
            hist.best_val = mse(net, x_va, z_va)
        if cfg.lr_schedule == "cosine":
            opt.lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + np.cos(np.pi * (ep - 1) / cfg.epochs))
        order = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            idx = order[s: s + bs]
            _, grads = _loss_grads(net, x_tr[idx], z_tr[idx], cfg.l2)
            opt.step(grads)
            updates += 1
        tr, va = mse(net, x_tr, z_tr), mse(net, x_va, z_va)
        if not np.isfinite(tr):
            hist.stop_reason = "diverged"
            break
        hist.epoch.append(ep)
        hist.train_mse.append(tr)
        hist.val_mse.append(va)
        hist.wall.append(time.perf_counter() - t0)
        hist.updates.append(updates)
        if callback and callback(ep, net):
            hist.stop_reason = "stopped by callback"
            break
        if va < hist.best_val:
            hist.best_val, hist.best_epoch, wait = va, ep, 0
            best = net.get_flat()
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                hist.stop_reason = f"early stop (patience {cfg.patience})"
                break
        if cfg.target_mse and tr < cfg.target_mse:
            hist.stop_reason = "target reached"
            break
    else:
        hist.stop_reason = "max epochs"
    if cfg.restore_best and hist.best_epoch > 0 and best.size == net.n_params:
        net.set_flat(best)
    return hist


def fit_lm(net: Network, x_tr: Array, z_tr: Array, x_va: Array, z_va: Array,
           cfg: TrainConfig, callback: Callable[[int, Network], object] | None = None,
           history: History | None = None) -> History:
    """Levenberg-Marquardt (Levenberg 1944, Marquardt 1963; for MLPs: Hagan & Menhaj 1994).

        theta <- theta - (J^T J + mu I)^-1 J^T e

    mu shrinks (x mu_dec) after a step that lowers the loss and grows (x mu_inc) until
    one does: large mu = small gradient-descent step, small mu = Gauss-Newton step.
    """
    if len(x_tr) > cfg.lm_max_points:
        idx = np.random.default_rng([cfg.seed, 11]).choice(len(x_tr), cfg.lm_max_points, replace=False)
        x_tr, z_tr = x_tr[idx], z_tr[idx]
    hist = History() if history is None else history
    mu = cfg.mu
    theta = net.get_flat()
    best = theta.copy()
    t0, wait, updates = time.perf_counter(), 0, 0
    if callback:
        callback(0, net)
    y, J = net.jacobian(x_tr)
    e = (y - z_tr).ravel()
    loss = float(e @ e) / len(e)
    eye = np.eye(theta.size)
    for it in range(1, cfg.epochs + 1):
        if it in cfg.kan_grid_schedule:
            net.kan_refit(x_tr, new_grid=cfg.kan_grid_schedule[it])
            theta = net.get_flat()
            best = theta.copy()
            eye = np.eye(theta.size)
            y, J = net.jacobian(x_tr)
            e = (y - z_tr).ravel()
            loss = float(e @ e) / len(e)
            hist.best_val = mse(net, x_va, z_va)
        g = J.T @ e
        H = J.T @ J
        accepted = False
        while mu <= cfg.mu_max:
            try:
                step = np.linalg.solve(H + mu * eye, g)
            except np.linalg.LinAlgError:
                mu *= cfg.mu_inc
                continue
            net.set_flat(theta - step)
            e_new = (net.forward(x_tr) - z_tr).ravel()
            loss_new = float(e_new @ e_new) / len(e_new)
            if np.isfinite(loss_new) and loss_new < loss:
                theta = theta - step
                mu = max(mu * cfg.mu_dec, 1e-20)
                accepted = True
                break
            mu *= cfg.mu_inc
        if not accepted:
            net.set_flat(theta)
            hist.stop_reason = "mu > mu_max (no descent step)"
            break
        updates += 1
        y, J = net.jacobian(x_tr)
        e = (y - z_tr).ravel()
        loss = float(e @ e) / len(e)
        va = mse(net, x_va, z_va)
        hist.epoch.append(it)
        hist.train_mse.append(loss)
        hist.val_mse.append(va)
        hist.wall.append(time.perf_counter() - t0)
        hist.updates.append(updates)
        hist.mu.append(mu)
        if callback and callback(it, net):
            hist.stop_reason = "stopped by callback"
            break
        if va < hist.best_val:
            hist.best_val, hist.best_epoch, wait = va, it, 0
            best = theta.copy()
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                hist.stop_reason = f"early stop (patience {cfg.patience})"
                break
        if cfg.target_mse and loss < cfg.target_mse:
            hist.stop_reason = "target reached"
            break
    else:
        hist.stop_reason = "max iterations"
    if cfg.restore_best and hist.best_epoch > 0 and best.size == net.n_params:
        net.set_flat(best)
    return hist
