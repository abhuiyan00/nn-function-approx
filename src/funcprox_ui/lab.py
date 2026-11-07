"""Training lab: runs the engine in a background thread and exposes its progress.

One worker thread: runs queue up, so wall-clock numbers stay comparable between runs.
A snapshot (prediction on a coarse grid) is taken from inside the training callback,
because ``Network.forward`` caches per-layer state and must not race with training.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from funcprox import bspline
from funcprox.baselines import ChebyshevLS, SplineLS
from funcprox.data import make_dataset
from funcprox.functions import Target
from funcprox.layers import KANLayer
from funcprox.metrics import evaluate, regression_metrics
from funcprox.model import Network, build_kan, build_mlp
from funcprox.train import History, TrainConfig, fit

from .schemas import RunSpec

GRID_1D, GRID_2D = 512, 48
SPECTRUM_POINTS, SPECTRUM_BINS = 1024, 48
MAX_SAMPLES = 256         # training points drawn on the fit chart
MAX_LIVE_EDGES = 24       # KAN edge functions per snapshot
MIN_SNAPSHOT_S = 0.1
MAX_KEPT_RUNS = 40
PREVIEW_SAMPLES = 150
# The size sweeps of experiment E6, as (name, model, sizes, coefficients of a size).
# Least squares on random points turns unstable far beyond these: degree 256 on 2,000 points gives RMSE 2e3 on |x|.
BASELINES = {
    1: (("Chebyshev LS", ChebyshevLS, (3, 5, 8, 12, 16, 24, 32, 48, 64, 96), lambda k: k + 1),
        ("Cubic spline LS", SplineLS, (6, 8, 12, 16, 24, 32, 48, 64, 96, 128), lambda k: k)),
    2: (("Chebyshev LS", ChebyshevLS, (2, 3, 4, 6, 8, 10, 12, 16, 20), lambda k: (k + 1) ** 2),
        ("Cubic spline LS", SplineLS, (5, 6, 8, 10, 12, 14, 16, 20), lambda k: k * k)),
}

OUTER_SCALES = np.linspace(0.05, 5.0, 200)  # the inner edges have an arbitrary scale a, so scan it as U2 does


def round_array(a, sig: int = 6) -> list:
    return [float(f"{v:.{sig}g}") if math.isfinite(v) else None for v in np.asarray(a, dtype=float).ravel().tolist()]


def clean(o):
    """Make a value JSON-safe: non-finite floats become null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def make_grid(target: Target, n1: int = GRID_1D, n2: int = GRID_2D) -> tuple[np.ndarray, dict]:
    """Display grid. In 1-D it also covers the extrapolation zone, if the target has one."""
    extra = dict(gap=[list(g) for g in target.gap] if target.gap else None, known=target.known)
    if target.dim == 1:
        lo, hi = (target.extrapolation_domain or target.domain)[0]
        x = np.linspace(lo, hi, n1).reshape(-1, 1)
        return x, dict(kind="1d", x=round_array(x), exact=round_array(target(x)), box=list(target.domain[0]), **extra)
    axes = [np.linspace(lo, hi, n2) for lo, hi in target.domain]
    mesh = np.meshgrid(*axes, indexing="ij")
    x = np.stack([m.ravel() for m in mesh], axis=1)
    return x, dict(kind="2d", x=round_array(axes[0]), y=round_array(axes[1]), exact=round_array(target(x)),
                   box=[list(d) for d in target.domain], **extra)


def samples_of(ds, limit: int = MAX_SAMPLES) -> dict:
    """Training points as the charts draw them: (x, y) in 1-D, input positions in 2-D."""
    x, y = ds.x_train[:limit], ds.y_train[:limit]
    if ds.target.dim == 1:
        return dict(x=round_array(x, 5), y=round_array(y, 5))
    return dict(x=round_array(x[:, 0], 5), y=round_array(x[:, 1], 5))


def preview(target: Target) -> dict:
    """What a function looks like before training, and where its data will fall."""
    _, grid = make_grid(target, n1=320, n2=36)
    grid["exact"] = round_array(grid["exact"], 4)
    ds = make_dataset(target, PREVIEW_SAMPLES, 5, n_test=2)
    return clean(dict(key=target.key, formula=target.formula, dim=target.dim, grid=grid, samples=samples_of(ds, PREVIEW_SAMPLES)))


def build_network(spec: RunSpec, dim: int) -> Network:
    if spec.arch == "kan":
        return build_kan(spec.widths, spec.grid, seed=spec.seed)
    fourier = (spec.fourier_m, spec.fourier_scale) if spec.fourier_m else None
    return build_mlp(dim, spec.hidden, spec.act, seed=spec.seed, bias_init=spec.bias_init, fourier=fourier, w0=spec.w0)


def _affine_fit(phi: np.ndarray, g: np.ndarray) -> tuple[float, np.ndarray]:
    """R^2 and values of the best a * g + b: R^2 near 1 means the edge has the form g."""
    a = np.stack([g, np.ones_like(g)], axis=1)
    fitted = a @ np.linalg.lstsq(a, phi, rcond=None)[0]
    ss = float(np.sum((phi - phi.mean()) ** 2))
    return (1.0 - float(np.sum((phi - fitted) ** 2)) / ss if ss > 0 else 0.0), fitted


def _read_f7_edge(j: int, i: int, xs: np.ndarray, phi: np.ndarray) -> dict:
    """Compare an edge of a KAN [2, 1, 1] with the known form of f7 = exp(sin(pi x) + y^2)."""
    if j == 0:
        name, g = ("sin(πx)", np.sin(np.pi * xs)) if i == 0 else ("y²", xs ** 2)
        r2, fitted = _affine_fit(phi, g)
    else:
        name = "exp(s/a)"
        r2, fitted = max((_affine_fit(phi, np.exp(xs / a)) for a in OUTER_SCALES), key=lambda t: t[0])
    return dict(name=name, r2=r2, fit=round_array(fitted, 5))


def kan_edges(net: Network, u: np.ndarray, target_key: str, n: int = 64) -> list[dict] | None:
    """Learned function of every KAN edge, over the range its input actually takes."""
    layers = [layer for layer in net.layers if isinstance(layer, KANLayer)]
    if not layers or sum(layer.n_in * layer.n_out for layer in layers) > MAX_LIVE_EDGES:
        return None
    readable = target_key == "f7" and net.config.get("widths") == [2, 1, 1]
    edges, x_in = [], u
    for j, layer in enumerate(layers):
        lo, hi = x_in.min(axis=0), x_in.max(axis=0)
        for i in range(layer.n_in):
            xs = np.linspace(lo[i], hi[i], n)
            b = bspline.basis(xs.reshape(-1, 1).repeat(layer.n_in, axis=1), layer.knots, layer.k)[:, i, :]
            for o in range(layer.n_out):
                phi = layer.wb[i, o] * layer.base.f(xs) + b @ layer.C[i, o]
                edge = dict(layer=j, i=i, o=o, x=round_array(xs, 5), y=round_array(phi, 5))
                if readable and np.all(np.isfinite(phi)):
                    edge["match"] = _read_f7_edge(j, i, xs, phi)
                edges.append(edge)
        x_in = layer.forward(x_in)
    return edges


def matched_baselines(ds, n_params: int, u_curve: np.ndarray | None = None) -> list[dict]:
    """Best Chebyshev and cubic-spline fit no larger than the network.

    The size is chosen on the validation points, the same rule that picks the network's best
    weights, so neither side sees the exact function before it is scored.
    ``u_curve`` adds the predictions at these scaled inputs to each row.
    """
    budget = min(n_params, len(ds.x_train) // 2)
    out = []
    for name, model_class, sizes, coefficients in BASELINES[ds.target.dim]:
        fits = []
        for size in sizes:
            if coefficients(size) > budget and fits:
                break
            try:
                model = model_class(size).fit(ds.u_train, ds.y_train)
            except np.linalg.LinAlgError:
                continue  # one unsolvable size must not cost the finished run its results
            fits.append((float(np.mean((model.predict(ds.u_val) - ds.y_val) ** 2)), model))
        if fits:
            val_mse, best = min(fits, key=lambda fit: fit[0])
            m = regression_metrics(best.predict(ds.u_test), ds.y_test, ds.u_test, hidden=ds.target.in_gap(ds.x_test))
            row = dict(model=name, n_coef=best.n_coef, rmse=m["rmse"], max_err=m["max_err"], val_mse=val_mse,
                       rmse_gap=m.get("rmse_gap"), rmse_seen=m.get("rmse_seen"))
            if u_curve is not None:
                row["pred"] = round_array(best.predict(u_curve))
            out.append(row)
    return out


class LabRun:
    def __init__(self, spec: RunSpec):
        self.id = uuid.uuid4().hex[:10]
        self.spec = spec
        self.target = spec.resolved
        self.created = time.time()
        self.status = "queued"
        self.error: str | None = None
        self.saved_as: str | None = None
        self.stop_requested = False
        self.ds = make_dataset(self.target, spec.n_train, spec.n_val, spec.sigma, spec.data_seed)
        self.net = build_network(spec, self.target.dim)
        self.xs, self.ys = self.ds.xs, self.ds.ys
        self.history = History()
        self.final: dict | None = None
        self.live: dict | None = None
        self.test_epoch: list[int] = []
        self.test_mse: list[float] = []
        self.comp_epoch: list[int] = []
        self.comp_rel: list[list[float]] = []

        self.grid_x, self.grid = make_grid(self.target)
        self.u_grid = self.xs.transform(self.grid_x)
        self.exact = self.target(self.grid_x).ravel()
        self.inside = np.all((self.grid_x >= self.xs.lo) & (self.grid_x <= self.xs.hi), axis=1) & np.isfinite(self.exact)
        scored = self.grid_x[self.inside] if self.target.known else self.ds.x_test
        self.score_hidden = self.target.in_gap(scored)
        self.u_sample = self.ds.u_train[:MAX_SAMPLES]
        self.samples = samples_of(self.ds)
        self._prepare_spectrum()

    # ---------------------------------------------------------------- preparation
    def _prepare_spectrum(self) -> None:
        self.u_spec = self.basis = self.amps = None
        self.grid["spectrum_exact"] = None
        if self.target.dim != 1:
            return
        lo, hi = self.target.domain[0]
        xs = np.linspace(lo, hi, SPECTRUM_POINTS, endpoint=False).reshape(-1, 1)  # periodic grid: bin k = k cycles
        fx = self.target(xs).ravel()
        if not np.isfinite(fx).all():  # no frequency content to compare with where the function has holes
            return
        self.u_spec = self.xs.transform(xs)
        self.grid["spectrum_exact"] = round_array(self._amplitudes(fx), 5)
        if self.target.frequencies:
            self.basis = np.stack([np.sin(k * np.pi * xs[:, 0]) for k in self.target.frequencies], axis=1)
            self.amps = self.basis.T @ fx / (SPECTRUM_POINTS / 2)

    @staticmethod
    def _amplitudes(y: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(y))[:SPECTRUM_BINS] / (SPECTRUM_POINTS / 2)

    # ------------------------------------------------------------------- training
    def _predict(self, u: np.ndarray) -> np.ndarray:
        return self.ys.inverse(self.net.forward(u)).ravel()

    def _snapshot(self, epoch: int, record: bool = True) -> dict:
        pred = self._predict(self.u_grid)
        residual = pred - self.exact
        if self.target.known:
            inner, truth = residual[self.inside], self.exact[self.inside]
        else:  # measured data: the held-out points stand in for the exact function
            truth = self.ds.y_test.ravel()
            inner = self._predict(self.ds.u_test) - truth
        std = float(np.std(truth))
        mse = float(np.mean(inner * inner))
        hole = inner[self.score_hidden]
        snap = dict(epoch=epoch, pred=round_array(pred), residual=round_array(residual, 4),
                    rmse=math.sqrt(mse) if math.isfinite(mse) else None,
                    rmse_gap=math.sqrt(float(np.mean(hole * hole))) if hole.size and np.isfinite(hole).all() else None,
                    max_err=float(np.max(np.abs(inner))),
                    r2=1.0 - mse / (std * std) if std > 0 else None,
                    spectrum=None, edges=kan_edges(self.net, self.u_sample, self.target.key))
        if self.u_spec is not None:
            p = self._predict(self.u_spec)
            snap["spectrum"] = dict(pred=round_array(self._amplitudes(p), 5))
            if self.basis is not None and record:
                a = self.basis.T @ p / (SPECTRUM_POINTS / 2)
                self.comp_rel.append(round_array(np.abs(a - self.amps) / self.amps, 4))
                self.comp_epoch.append(epoch)
        if record and math.isfinite(mse):  # same scaled units as the train / validation curves
            self.test_mse.append(float(f"{mse / self.ys.std ** 2:.6g}"))
            self.test_epoch.append(epoch)
        return snap

    def execute(self) -> None:
        if self.stop_requested:
            self.status = "stopped"
            self.ds = None
            return
        self.status = "running"
        try:
            with np.errstate(all="ignore"):  # a diverging run is a result here, not a warning
                self._train()
            self.status = "stopped" if self.stop_requested else "done"
        except Exception as exc:  # a failed run must not kill the worker thread
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "failed"
        finally:
            self.ds = None

    def _train(self) -> None:
        s, ds = self.spec, self.ds
        cfg = TrainConfig(optimizer=s.optimizer, lr=s.lr, batch=s.batch, epochs=s.epochs, patience=s.patience,
                          l2=s.l2, seed=s.seed, kan_grid_schedule={it: s.grid for it in s.grid_updates})
        self.config = cfg
        last, every = 0.0, MIN_SNAPSHOT_S

        def on_epoch(epoch: int, _net: Network) -> bool:
            nonlocal last, every
            if self.stop_requested:
                return True
            now = time.perf_counter()
            if epoch == 0 or now - last >= every:
                self.live = self._snapshot(epoch)
                done = time.perf_counter()
                every = max(MIN_SNAPSHOT_S, 4.0 * (done - now))  # snapshots take at most ~20 % of the run
                last = done
            return False

        hist = fit(self.net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg, callback=on_epoch, history=self.history)
        # best weights are restored by now: they belong on the curve only if they are the last epoch's
        snap = self._snapshot(len(hist.epoch), record=hist.best_epoch == len(hist.epoch))
        self.live = snap
        self.final = dict(metrics=evaluate(self.net, ds), best_epoch=hist.best_epoch, epochs_run=len(hist.epoch),
                          stop=hist.stop_reason, seconds=hist.wall[-1] if hist.wall else 0.0,
                          pred=snap["pred"], residual=snap["residual"], spectrum=snap["spectrum"], edges=snap["edges"],
                          baselines=matched_baselines(ds, self.net.n_params))

    # ------------------------------------------------------------------- payloads
    @property
    def finished(self) -> bool:
        return self.status in ("done", "stopped", "failed")

    def summary(self) -> dict:
        f, live = self.final, self.live
        m = f["metrics"] if f else {}
        return clean(dict(id=self.id, status=self.status, label=self.spec.describe(), spec=self.spec.model_dump(),
                          n_params=self.net.n_params, created=self.created, error=self.error, saved_as=self.saved_as,
                          rmse=m.get("rmse", live["rmse"] if live else None),
                          max_err=m.get("max_err", live["max_err"] if live else None),
                          r2=m.get("r2", live["r2"] if live else None),
                          seconds=f["seconds"] if f else None, epochs_run=f["epochs_run"] if f else len(self.history.epoch),
                          best_epoch=f["best_epoch"] if f else None, stop=f["stop"] if f else None))

    def _history_count(self) -> int:
        h = self.history  # the training thread appends in this order; read the shortest list
        n = min(len(h.epoch), len(h.train_mse), len(h.val_mse), len(h.wall), len(h.updates))
        return min(n, len(h.mu)) if self.spec.optimizer == "lm" else n

    def _history_slice(self, a: int, b: int) -> dict:
        h = self.history
        return dict(epoch=h.epoch[a:b], train=round_array(h.train_mse[a:b]), val=round_array(h.val_mse[a:b]),
                    wall=[round(v, 4) for v in h.wall[a:b]], updates=h.updates[a:b], mu=round_array(h.mu[a:b], 4))

    def full(self) -> dict:
        n, t, c = self._history_count(), len(self.test_mse), len(self.comp_rel)
        comp = None
        if self.target.frequencies and self.target.dim == 1:
            comp = dict(k=list(self.target.frequencies), epoch=self.comp_epoch[:c], rel=self.comp_rel[:c])
        hist = self._history_slice(0, n)
        hist.update(test_epoch=self.test_epoch[:t], test_mse=self.test_mse[:t])
        return dict(**self.summary(), architecture=repr(self.net), formula=self.target.formula, grid=self.grid,
                    samples=self.samples, history=hist, components=comp, live=clean(self.live), final=clean(self.final))

    def subscribe(self) -> tuple[dict, dict]:
        """Everything so far, and the cursor for ``delta`` that continues exactly after it."""
        live = self.live
        init = self.full()
        comp = init["components"]
        return init, dict(n=len(init["history"]["epoch"]), t=len(init["history"]["test_mse"]),
                          c=len(comp["rel"]) if comp else 0, live=live, status=init["status"])

    def delta(self, cursor: dict) -> dict:
        """What changed since ``cursor`` (updated in place); empty when nothing did."""
        out: dict = {}
        n = self._history_count()
        if n > cursor.get("n", 0):
            out["history"] = self._history_slice(cursor.get("n", 0), n)
            cursor["n"] = n
        t = len(self.test_mse)
        if t > cursor.get("t", 0):
            out["test"] = dict(epoch=self.test_epoch[cursor.get("t", 0):t], mse=self.test_mse[cursor.get("t", 0):t])
            cursor["t"] = t
        c = len(self.comp_rel)
        if c > cursor.get("c", 0):
            out["components"] = dict(epoch=self.comp_epoch[cursor.get("c", 0):c], rel=self.comp_rel[cursor.get("c", 0):c])
            cursor["c"] = c
        if self.live is not cursor.get("live"):
            cursor["live"] = self.live
            out["live"] = clean(self.live)
        if self.status != cursor.get("status"):
            cursor["status"] = self.status
            out["status"] = self.status
        return out


class LabManager:
    def __init__(self):
        self.runs: OrderedDict[str, LabRun] = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab")

    def run_after_queue(self, work) -> None:
        """``work`` runs once everything queued so far has finished."""
        self._pool.submit(work)

    def submit(self, spec: RunSpec) -> LabRun:
        run = LabRun(spec)
        with self._lock:
            self.runs[run.id] = run
            done = [r for r in self.runs.values() if r.finished]
            for old in done[: max(0, len(self.runs) - MAX_KEPT_RUNS)]:
                del self.runs[old.id]
        self._pool.submit(run.execute)
        return run

    def get(self, run_id: str) -> LabRun | None:
        return self.runs.get(run_id)

    def remove(self, run_id: str) -> bool:
        with self._lock:
            run = self.runs.pop(run_id, None)
        if run:
            run.stop_requested = True
        return run is not None

    def all(self) -> list[LabRun]:
        with self._lock:
            return list(reversed(self.runs.values()))

    def shutdown(self) -> None:
        for run in self.all():
            run.stop_requested = True
        self._pool.shutdown(wait=False, cancel_futures=True)
