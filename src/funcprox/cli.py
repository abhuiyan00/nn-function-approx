"""Command line: train / predict / targets / ui.

Examples
    python -m funcprox targets
    python -m funcprox train --target f4 --hidden 32,32 --act tanh --out runs/f4.npz
    python -m funcprox train --target f7 --arch kan --widths 2,1,1 --grid 10 --grid-updates 25,75 --optimizer lm
    python -m funcprox train --formula "x^3 - x" --domain=-1.5,1.5 --gap=-0.5,0.5 --optimizer lm --hidden 16,16
    python -m funcprox train --points measured.csv --gap=-0.2,0.2
    python -m funcprox predict runs/f4.npz 0.0 0.5 1.0
    python -m funcprox ui
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import numpy as np

from . import serialize
from .custom import make_target, saved_target
from .data import make_dataset
from .functions import TARGETS
from .metrics import evaluate
from .model import build_kan, build_mlp
from .train import TrainConfig, fit


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _pairs(s: str) -> list[tuple[float, float]]:
    """"-1,1,-2,2" means the ranges (-1, 1) and (-2, 2); an empty text means none."""
    v = [float(part) for part in s.split(",") if part.strip()]
    if len(v) % 2:
        raise ValueError(f"{s!r} is not a list of from,to ranges")
    return list(zip(v[::2], v[1::2]))


def _function(a) -> dict:
    """The arguments of make_target, kept with a saved model so it can be rebuilt."""
    if a.formula is not None:
        fn = dict(formula=a.formula)
    elif a.points:
        rows = np.atleast_2d(np.genfromtxt(a.points, delimiter=",", encoding="utf-8-sig"))
        fn = dict(points=rows[np.isfinite(rows).all(axis=1)].tolist())  # a header line reads as NaN
    else:
        fn = dict(target=a.target)
    if a.domain:
        fn["domain"] = _pairs(a.domain)
    if a.gap is not None:
        fn["gap"] = _pairs(a.gap)
    return fn


def cmd_targets(_args) -> int:
    for t in TARGETS.values():
        hole = f"  missing middle: {t.gap}" if t.gap else ""
        print(f"{t.key}  {t.name:<22} {t.formula:<45} domain={t.domain}  probes: {t.probes}{hole}")
    return 0


def cmd_train(a) -> int:
    try:
        function = _function(a)
        target = make_target(**function)
    except (ValueError, OSError) as exc:
        print(f"Cannot use this function: {exc}", file=sys.stderr)
        return 2
    ds = make_dataset(target, a.n_train, a.n_val, a.sigma, a.data_seed)
    if a.arch == "kan":
        net = build_kan(_ints(a.widths) if a.widths else [target.dim, 5, 1], a.grid, seed=a.seed)
    else:
        fourier = (a.fourier_m, a.fourier_scale) if a.fourier_m else None
        net = build_mlp(target.dim, _ints(a.hidden), a.act, seed=a.seed,
                        bias_init=a.bias_init, fourier=fourier, w0=a.w0)
    cfg = TrainConfig(optimizer=a.optimizer, lr=a.lr, batch=a.batch, epochs=a.epochs,
                      patience=a.patience, l2=a.l2, seed=a.seed,
                      kan_grid_schedule={it: a.grid for it in _ints(a.grid_updates)})
    print(net)
    hist = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg)
    m = evaluate(net, ds)
    m.update(best_epoch=hist.best_epoch, epochs_run=len(hist.epoch), stop=hist.stop_reason,
             seconds=round(hist.wall[-1], 2) if hist.wall else 0.0)
    print(json.dumps(m, indent=2))
    if a.out:
        serialize.save(a.out, net, ds.xs, ds.ys, meta=dict(target=target.key, function=function, metrics=m, train=cfg.to_dict()))
        print(f"saved {a.out}")
    return 0


def cmd_predict(a) -> int:
    net, xs, ys, meta = serialize.load(a.model)
    x = np.array(a.x, dtype=float).reshape(-1, net.config.get("n_in", net.layers[0].n_in))
    y = ys.inverse(net.forward(xs.transform(x)))
    target = saved_target(meta)
    for xi, yi in zip(x, y.ravel()):
        exact = f"   exact={float(target(xi.reshape(1, -1))[0, 0]):.6g}" if target and target.known else ""
        print(f"x={xi.tolist()}  y_hat={yi:.6g}{exact}")
    return 0


def cmd_ui(a) -> int:
    try:  # optional extra: the engine itself stays NumPy-only
        from funcprox_ui import serve

        return serve(port=a.port, open_browser=not a.no_browser)
    except ModuleNotFoundError as exc:
        print(f'The interface needs extra packages ({exc}). Install them with: pip install -e ".[ui]"', file=sys.stderr)
        return 1


def main(argv=None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # formulas contain π; a pipe would use the locale encoding
    p = argparse.ArgumentParser(prog="funcprox", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets", help="list target functions").set_defaults(func=cmd_targets)

    t = sub.add_parser("train", help="train one network and report test metrics")
    t.add_argument("--target", default="f1")
    t.add_argument("--formula", default=None, help='your own function of x (and y), e.g. "sin(3*x) * exp(-x^2)"')
    t.add_argument("--points", default="", help="your own data: a CSV file with one x,y pair per line")
    t.add_argument("--domain", default="", help="range of a formula, e.g. --domain=-2,2 or --domain=-1,1,-1,1 (default -1 to 1)")
    t.add_argument("--gap", default=None, help="missing middle, no training data inside: --gap=-0.5,0.5 or --gap=-0.3,0.3,-0.3,0.3")
    t.add_argument("--arch", choices=["mlp", "kan"], default="mlp")
    t.add_argument("--hidden", default="32,32")
    t.add_argument("--act", default="tanh")
    t.add_argument("--bias-init", default=None, choices=[None, "zero", "kink"])
    t.add_argument("--fourier-m", type=int, default=0)
    t.add_argument("--fourier-scale", type=float, default=2.0)
    t.add_argument("--w0", type=float, default=30.0)
    t.add_argument("--widths", default="")
    t.add_argument("--grid", type=int, default=5)
    t.add_argument("--grid-updates", default="", help="KAN: iterations that refit the grid range to the data, e.g. 25,75")
    t.add_argument("--optimizer", default="adam")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--l2", type=float, default=0.0)
    t.add_argument("--n-train", type=int, default=2000)
    t.add_argument("--n-val", type=int, default=500)
    t.add_argument("--sigma", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--data-seed", type=int, default=0)
    t.add_argument("--out", default="")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="evaluate a saved model at given inputs")
    q.add_argument("model")
    q.add_argument("x", nargs="+", type=float)
    q.set_defaults(func=cmd_predict)

    u = sub.add_parser("ui", help="open the visual interface in a browser")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(func=cmd_ui)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
