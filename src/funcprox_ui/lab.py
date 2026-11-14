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
from funcprox.baselines import ChebyshevLS, RBFNet, SplineLS
from funcprox.data import make_dataset
from funcprox.functions import Target
from funcprox.layers import Dense, KANLayer
from funcprox.metrics import evaluate, regression_metrics
from funcprox.model import Network, build_kan, build_mlp
from funcprox.train import History, TrainConfig, fit, mse

from .schemas import RunSpec

GRID_1D, GRID_2D = 512, 48
SPECTRUM_POINTS, SPECTRUM_BINS = 1024, 48
MAX_SAMPLES = 256         # training points drawn on the fit chart
MAX_LIVE_EDGES = 24       # KAN edge functions per snapshot
VIEW_UNITS = 24           # units of one layer drawn in the network view; wider layers are thinned
VIEW_PROBES = 24          # inputs the network view steps through
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

RBF_SIZES = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
UNIT_CURVES = 8           # hidden units drawn one by one, one colour each; the rest are drawn as one curve
CAL_POINTS = 200          # fresh points that set the width of the trust band
BAND_LEVEL = 0.9
LANDSCAPE_N = 25          # the loss surface is LANDSCAPE_N x LANDSCAPE_N forward passes
LANDSCAPE_POINTS = 512    # training points the loss surface is measured on
PATH_FLOATS = 2_000_000   # memory for the weights kept along the way, in numbers
FRAME_FLOATS = 60_000     # the same for the curves kept for the replay

FORM_R2 = 0.999           # the simplest form that reaches this beats a closer but more complex one
SIN_WAVES = np.linspace(0.5, 16.0, 63)        # half-waves across the range of an edge's input
RATES = np.geomspace(0.05, 8.0, 40)           # exp and tanh rates, per half-range of the input


def baselines_for(dim: int) -> tuple:
    """The sweeps of E6 where they exist (1-D and 2-D), and an RBF network, which works in any dimension."""
    return (*BASELINES.get(dim, ()), ("RBF network LS", RBFNet, RBF_SIZES, lambda k: k * (dim + 1) + 1))


class Reel:
    """Keeps at most ``limit`` evenly spread items of a stream whose length is not known in advance."""

    def __init__(self, limit: int):
        self.limit, self.stride, self.count, self.items = max(2, int(limit)), 1, 0, []

    def add(self, make) -> None:
        """``make()`` builds the item, and is only called when the item is kept."""
        if self.count % self.stride == 0:
            self.items = [*self.items, make()]  # a new list each time: readers in other threads see a whole one
            if len(self.items) > self.limit:
                self.items, self.stride = self.items[::2], self.stride * 2
        self.count += 1


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
    """Display grid. In 1-D it also covers the extrapolation zone, if the target has one.

    With more than two inputs it is the slice over the first two, the others held at mid-range.
    """
    fixed = [0.5 * (lo + hi) for lo, hi in target.domain[2:]]
    gap = target.gap
    if gap and any(not lo <= v <= hi for v, (lo, hi) in zip(fixed, gap[2:])):
        gap = None  # the slice passes beside the hole
    extra = dict(gap=[list(g) for g in gap[:2]] if gap else None, known=target.known)
    if target.dim == 1:
        lo, hi = (target.extrapolation_domain or target.domain)[0]
        x = np.linspace(lo, hi, n1).reshape(-1, 1)
        return x, dict(kind="1d", x=round_array(x), exact=round_array(target(x)), box=list(target.domain[0]), **extra)
    axes = [np.linspace(lo, hi, n2) for lo, hi in target.domain[:2]]
    mesh = np.meshgrid(*axes, indexing="ij")
    x = np.stack([m.ravel() for m in mesh] + [np.full(n2 * n2, v) for v in fixed], axis=1)
    return x, dict(kind="2d", x=round_array(axes[0]), y=round_array(axes[1]), exact=round_array(target(x)),
                   box=[list(d) for d in target.domain[:2]], fixed=fixed, **extra)


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


def _pi(w: float) -> str:
    """A frequency as a multiple of pi when it is one, within 0.5 %."""
    q = round(4 * w / np.pi) / 4
    if q > 0 and abs(w / np.pi - q) < 0.005 * q:
        return "π" if q == 1 else f"{q:g}π"
    return f"{w:.3g}·"


def closest_form(x: np.ndarray, phi: np.ndarray, var: str) -> dict | None:
    """The simple form a * g(x) + b closest to an edge function, with its R^2.

    Forms are tried from simple to complex: the first to reach FORM_R2 wins, otherwise the
    closest. A high R^2 suggests a form and does not prove it: fits of single edges are known
    to change with the initial weights (see docs/RESEARCH.md).
    """
    centred = phi - phi.mean()
    total = float(centred @ centred)
    if not np.isfinite(total) or total <= 1e-18:
        return None  # a constant edge has no shape to name
    half = 0.5 * float(x.max() - x.min()) or 1.0

    def fit(*columns):
        a = np.stack([*columns, np.ones_like(x)], axis=1)
        coef = np.linalg.lstsq(a, phi, rcond=None)[0]
        fitted = a @ coef
        return 1.0 - float(np.sum((phi - fitted) ** 2)) / total, fitted, coef

    def scan(make, params):  # the parameter of g whose a * g + b is closest, in closed form
        g = make(params[:, None])
        g = g - g.mean(axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            r2 = (g @ centred) ** 2 / ((g * g).sum(axis=1) * total)
        return float(params[int(np.argmax(np.where(np.isfinite(r2), r2, -1.0)))])

    def waves(ws):  # R^2 of a sin(wx) + c cos(wx) + b for every frequency at once
        arg = ws[:, None] * x[None, :]
        a = np.stack([np.sin(arg), np.cos(arg), np.ones_like(arg)], axis=2)
        res = phi - np.einsum("sni,si->sn", a, np.linalg.pinv(a) @ phi)
        return 1.0 - (res * res).sum(axis=1) / total

    forms = [(var, fit(x)), (f"{var}²", fit(x * x)), (f"{var}³", fit(x ** 3)), (f"|{var}|", fit(np.abs(x)))]
    if x.min() > 0:
        forms.append((f"log({var})", fit(np.log(x))))
    if x.min() >= 0:
        forms.append((f"√{var}", fit(np.sqrt(x))))
    rates = np.concatenate([-RATES[::-1], RATES]) / half
    r = scan(lambda k: np.exp(k * x[None, :]), rates)
    forms.append((f"exp({r:.3g}·{var})", fit(np.exp(r * x))))
    r = scan(lambda k: np.tanh(k * x[None, :]), RATES / half)
    forms.append((f"tanh({r:.3g}·{var})", fit(np.tanh(r * x))))
    coarse = SIN_WAVES * np.pi / (2.0 * half)
    w = coarse[int(np.argmax(waves(coarse)))]
    for reach in (0.25, 0.025):  # two finer passes around the best frequency
        fine = w + np.linspace(-reach, reach, 21) * np.pi / (2.0 * half)
        w = float(fine[int(np.argmax(waves(fine)))])
    r2, fitted, (a_sin, a_cos, _) = fit(np.sin(w * x), np.cos(w * x))
    shift = float(np.arctan(a_cos / a_sin)) if a_sin else np.pi / 2  # the sign of the wave is in a
    # the wave comes last: with a free frequency and shift it can imitate most of the forms above
    forms.append((f"sin({_pi(w)}{var}{f' {shift:+.2f}' if abs(shift) > 0.05 else ''})", (r2, fitted, None)))
    name, (r2, fitted, _) = next((f for f in forms if f[1][0] >= FORM_R2), max(forms, key=lambda f: f[1][0]))
    return dict(name=name, r2=r2, fit=round_array(fitted, 5))


def kan_edges(net: Network, u: np.ndarray, xs, n: int = 64) -> list[dict] | None:
    """Learned function of every KAN edge, over the range its input actually takes.

    Edges of the first layer are drawn over the original input (x, y, z); deeper ones over
    the value s that reaches them. Each one comes with its closest simple form.
    """
    layers = [layer for layer in net.layers if isinstance(layer, KANLayer)]
    if not layers or sum(layer.n_in * layer.n_out for layer in layers) > MAX_LIVE_EDGES:
        return None
    edges, x_in = [], u
    for j, layer in enumerate(layers):
        lo, hi = x_in.min(axis=0), x_in.max(axis=0)
        for i in range(layer.n_in):
            at = np.linspace(lo[i], hi[i], n)
            b = bspline.basis(at.reshape(-1, 1).repeat(layer.n_in, axis=1), layer.knots, layer.k)[:, i, :]
            first = j == 0 and i < 3
            shown = (at + 1.0) * 0.5 * (xs.hi[i] - xs.lo[i]) + xs.lo[i] if first else at
            for o in range(layer.n_out):
                phi = layer.wb[i, o] * layer.base.f(at) + b @ layer.C[i, o]
                edge = dict(layer=j, i=i, o=o, x=round_array(shown, 5), y=round_array(phi, 5))
                if np.all(np.isfinite(phi)):
                    match = closest_form(shown, phi, "xyz"[i] if first else "s")
                    if match:
                        edge["match"] = match
                edges.append(edge)
        x_in = layer.forward(x_in)
    return edges


def _thin(n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, VIEW_UNITS)).round().astype(int))


def anatomy(net: Network, u: np.ndarray) -> dict:
    """What the network view draws: units of every layer, the links between them, their values on ``u``.

    A Dense link is its weight and a Fourier link the fixed frequency it projects on. A KAN link
    is a function, so its strength is the mean size of its output over ``u``.
    """
    keep = _thin(u.shape[1])
    names, sizes, acts, links = ["input"], [u.shape[1]], [u[:, keep]], []
    for layer in net.layers:
        y = layer.forward(u)
        if isinstance(layer, KANLayer):
            w = np.abs(np.einsum("nim,iom->nio", layer.B, layer.C) + layer.s[:, :, None] * layer.wb).mean(axis=0)
            names.append("KAN")
        elif isinstance(layer, Dense):
            w = layer.W
            names.append(str(layer.act))
        else:
            w = np.tile(layer.B, 2)  # the sine and the cosine half share one projection
            names.append("Fourier")
        nxt = _thin(y.shape[1])
        links.append(round_array(w[np.ix_(keep, nxt)], 3))
        sizes.append(y.shape[1])
        acts.append(y[:, nxt])
        u, keep = y, nxt
    names[-1] = "output"
    return dict(names=names, sizes=sizes, shown=[a.shape[1] for a in acts], kan=net.config.get("arch") == "kan",
                act=[round_array(a, 3) for a in acts], w=links)


def matched_baselines(ds, n_params: int, u_curve: np.ndarray | None = None) -> list[dict]:
    """Best Chebyshev and cubic-spline fit no larger than the network.

    The size is chosen on the validation points, the same rule that picks the network's best
    weights, so neither side sees the exact function before it is scored.
    ``u_curve`` adds the predictions at these scaled inputs to each row.
    """
    budget = min(n_params, len(ds.x_train) // 2)
    out = []
    for name, model_class, sizes, coefficients in baselines_for(ds.target.dim):
        fits = []
        for size in sizes:
            if coefficients(size) > budget and fits:
                break
            try:
                model = model_class(size).fit(ds.u_train, ds.y_train)
            except (np.linalg.LinAlgError, ValueError):
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
        box = np.array(self.target.domain, dtype=float)  # the view's inputs cross the box corner to corner
        self.probe = box[:, 0] + np.linspace(0.0, 1.0, VIEW_PROBES)[:, None] * (box[:, 1] - box[:, 0])
        self.u_probe = self.xs.transform(self.probe)
        self.loss_u, self.loss_z = self.ds.u_train[:LANDSCAPE_POINTS], self.ds.z_train[:LANDSCAPE_POINTS]
        self.path = Reel(min(120, max(8, PATH_FLOATS // max(1, self.net.n_params))))  # (epoch, weights) along the way
        self.frames = Reel(max(20, FRAME_FLOATS // len(self.exact)))                  # (epoch, curve) for the replay
        self._landscape: dict | None = None
        self.cal = None
        if self.target.known:  # fresh points, never used to train or to stop: they calibrate the trust band
            fresh = make_dataset(self.target, CAL_POINTS, 5, spec.sigma, spec.data_seed + 7919, n_test=2)
            self.cal = (self.xs.transform(fresh.x_train), fresh.y_train.ravel())
        one_d = self.target.dim == 1
        self.u_units = self.u_grid[::4] if one_d and spec.arch == "mlp" else None
        self.grid["units_x"] = round_array(self.grid_x[::4]) if self.u_units is not None else None
        self.slope_exact = None
        if one_d and self.target.known:
            with np.errstate(all="ignore"):
                self.slope_exact = np.gradient(self.exact, self.grid_x[:, 0])
        self.grid["slope_exact"] = round_array(self.slope_exact, 4) if self.slope_exact is not None else None
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

    def _units(self) -> dict | None:
        """Share of each unit of the last hidden layer in the answer: their sum plus ``offset`` is the prediction."""
        if self.u_units is None:
            return None
        self.net.forward(self.u_units)
        out = self.net.layers[-1]
        share = self.ys.std * self.net.layers[-2].a * out.W[:, 0]
        size = np.nan_to_num(share.max(axis=0) - share.min(axis=0), nan=-1.0)
        order = np.argsort(-size, kind="stable")
        top, rest = order[:UNIT_CURVES], order[UNIT_CURVES:]
        return dict(index=top.tolist(), curves=[round_array(share[:, k], 4) for k in top],
                    rest=round_array(share[:, rest].sum(axis=1), 4) if rest.size else None,
                    offset=self.ys.mean + self.ys.std * float(out.b[0]))

    def _slope(self) -> dict | None:
        """dy/dx of the network, by the chain rule through both scalings, against the slope of the exact f."""
        if self.slope_exact is None:
            return None
        d = self.ys.std * self.net.input_gradient(self.u_grid)[:, 0] * 2.0 / (self.xs.hi[0] - self.xs.lo[0])
        e = (d - self.slope_exact)[self.inside & np.isfinite(self.slope_exact)]
        ok = e.size and np.isfinite(e).all()
        return dict(pred=round_array(d, 4), rmse=float(np.sqrt(np.mean(e * e))) if ok else None)

    def _band(self) -> dict | None:
        """Split conformal band: BAND_LEVEL of fresh points from the same source lie within ``half`` of the network.

        The promise covers inputs drawn like the training data. It says nothing about a missing
        middle or about the zone outside the box, where no point was ever drawn.
        """
        if self.cal is None:
            return None
        r = np.abs(self._predict(self.cal[0]) - self.cal[1])
        r = np.sort(r[np.isfinite(r)])
        if r.size < 20:
            return None
        rank = min(r.size, math.ceil((r.size + 1) * BAND_LEVEL))
        return dict(level=BAND_LEVEL, n=int(r.size), half=float(r[rank - 1]))

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
                    spectrum=None, edges=kan_edges(self.net, self.u_sample, self.xs), units=self._units(), slope=self._slope(),
                    net=dict(anatomy(self.net, self.u_probe), x=round_array(self.probe, 4),
                             y=round_array(self._predict(self.u_probe), 4)))
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
                self.live = snap = self._snapshot(epoch)
                self.frames.add(lambda: (epoch, snap["pred"]))
                self.path.add(lambda: (epoch, self.net.get_flat().astype(np.float32)))
                done = time.perf_counter()
                every = max(MIN_SNAPSHOT_S, 4.0 * (done - now))  # snapshots take at most ~20 % of the run
                last = done
            return False

        hist = fit(self.net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg, callback=on_epoch, history=self.history)
        # best weights are restored by now: they belong on the curve only if they are the last epoch's
        snap = self._snapshot(len(hist.epoch), record=hist.best_epoch == len(hist.epoch))
        self.live = snap
        self.frames.items = [*self.frames.items, (hist.best_epoch or len(hist.epoch), snap["pred"])]  # the replay ends on the device
        self.final = dict(metrics=evaluate(self.net, ds), best_epoch=hist.best_epoch, epochs_run=len(hist.epoch),
                          stop=hist.stop_reason, seconds=hist.wall[-1] if hist.wall else 0.0,
                          pred=snap["pred"], residual=snap["residual"], spectrum=snap["spectrum"], edges=snap["edges"],
                          units=snap["units"], slope=snap["slope"], band=self._band(),
                          baselines=matched_baselines(ds, self.net.n_params))

    # ----------------------------------------------------------------- on request
    def replay(self) -> dict:
        """The curves kept along the way, oldest first."""
        items = self.frames.items
        return dict(epoch=[e for e, _ in items], pred=[p for _, p in items])

    def landscape(self) -> dict:
        """Training loss on the plane through the final weights that holds most of the path there.

        The two directions are the principal components of the path (Li et al., 2018), so the
        path drawn on the surface is close to the one the optimiser took. Computed once.
        """
        if self._landscape is None:
            net = self.net.copy()  # its own caches: a prediction may run at the same time
            theta = net.get_flat()
            kept = [(e, w) for e, w in self.path.items if w.size == theta.size]  # a KAN grid update resizes the weights
            steps = np.array([w for _, w in kept], dtype=float).reshape(len(kept), theta.size) - theta
            s = np.zeros(0)
            if len(kept):
                _, s, vt = np.linalg.svd(steps, full_matrices=False)
            rng, dirs = np.random.default_rng(0), []
            for k in range(2):
                d = vt[k] if k < len(s) and s[k] > 1e-9 * max(float(s[0]), 1e-300) else rng.normal(size=theta.size)
                for other in dirs:
                    d = d - (d @ other) * other
                dirs.append(d / np.linalg.norm(d))
            a, b = steps @ dirs[0], steps @ dirs[1]

            def axis(coords: np.ndarray) -> np.ndarray:
                lo, hi = min(0.0, float(coords.min(initial=0.0))), max(0.0, float(coords.max(initial=0.0)))
                pad = 0.25 * (hi - lo) or 1.0
                return np.linspace(lo - pad, hi + pad, LANDSCAPE_N)

            alpha, beta = axis(a), axis(b)
            loss = np.empty((LANDSCAPE_N, LANDSCAPE_N))
            with np.errstate(all="ignore"):
                for i, x in enumerate(alpha):
                    for j, y in enumerate(beta):
                        net.set_flat(theta + x * dirs[0] + y * dirs[1])
                        loss[i, j] = mse(net, self.loss_u, self.loss_z)
                loss = np.log10(loss)
            energy = float((s ** 2).sum())
            self._landscape = clean(dict(
                alpha=round_array(alpha, 5), beta=round_array(beta, 5), loss=round_array(loss, 4),
                path=dict(a=round_array(a, 5), b=round_array(b, 5), epoch=[e for e, _ in kept]),
                explained=float((s[:2] ** 2).sum()) / energy if energy > 0 else None))
        return self._landscape

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
