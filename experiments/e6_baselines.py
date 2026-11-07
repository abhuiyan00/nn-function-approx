"""E6 - Is a neural network the right tool? Classical approximators at matched size.

Question: at an equal number of fitted coefficients, how do MLPs (Adam and
Levenberg-Marquardt) compare with Chebyshev least squares and cubic-spline least
squares? Also the classic Runge picture: polynomial interpolation on equispaced vs
Chebyshev nodes.
Targets f1, f4, f5, f6; noise-free N = 2,000 training points; test on the exact grid.
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import numpy as np

from funcprox import plotting as P
from funcprox.baselines import ChebyshevLS, PolyInterpolant, SplineLS
from funcprox.data import make_dataset
from funcprox.functions import get_target
from funcprox.metrics import regression_metrics

EXP = "e6_baselines"
TARGETS = {"f1": "f1 · sin(2πx)", "f4": "f4 · Runge", "f5": "f5 · Franke (2-D)", "f6": "f6 · |x|"}
SERIES = ["MLP 2×H · Adam", "MLP 2×H · LM", "Chebyshev LS", "cubic spline LS"]


def classical(prof) -> list[dict]:
    rows = []
    for t in TARGETS:
        target = get_target(t)
        ds = make_dataset(target, prof["n_train"], prof["n_val"], 0.0, 0)
        d = target.dim
        degs = [3, 5, 8, 12, 16, 24, 32, 48, 64, 96] if d == 1 else [2, 3, 4, 6, 8, 10, 12, 16, 20]
        nbs = [6, 8, 12, 16, 24, 32, 48, 64, 96, 128] if d == 1 else [5, 6, 8, 10, 12, 14, 16, 20]
        for name, models in (("Chebyshev LS", [ChebyshevLS(k) for k in degs]),
                             ("cubic spline LS", [SplineLS(k) for k in nbs])):
            for mdl in models:
                mdl.fit(ds.u_train, ds.y_train)
                m = regression_metrics(mdl.predict(ds.u_test), ds.y_test, ds.u_test)
                rows.append(dict(target=t, model=name, n_params=mdl.n_coef, **m))
    return rows


def main():
    profile, prof = parse_profile(__doc__)
    base = dict(n_train=prof["n_train"], n_val=prof["n_val"])
    specs = []
    for t in TARGETS:
        for h in (4, 8, 16, 32, 64):
            for s in prof["seeds"]:
                specs.append(dict(target=t, model=SERIES[0], hidden=(h, h), seed=s, epochs=prof["epochs"],
                                  patience=prof["patience"], **base))
        for h in (4, 6, 8, 12, 16, 20):
            for s in prof["seeds"]:
                specs.append(dict(target=t, model=SERIES[1], hidden=(h, h), seed=s, optimizer="lm",
                                  epochs=prof["lm_iters"], patience=30, **base))
    nn_rows = run_many(specs, prof["workers"], label=EXP)
    cl_rows = classical(prof)
    rows = nn_rows + cl_rows
    save_rows(EXP, rows)

    f, axes = P.plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, t in zip(axes, TARGETS):
        for i, name in enumerate(SERIES):
            rs = [r for r in rows if r["target"] == t and r["model"] == name]
            ps = sorted({r["n_params"] for r in rs})
            best = [min(r["rmse"] for r in rs if r["n_params"] == p) for p in ps]
            ax.plot(ps, best, label=name, **P.marker_kw(i))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TARGETS[t])
        ax.set_xlabel("fitted parameters / coefficients")
    axes[0].set_ylabel("test RMSE vs exact f")
    h_, l_ = axes[0].get_legend_handles_labels()
    f.legend(h_, l_, loc="outside upper right", ncol=4)
    f.suptitle("E6 · Networks vs classical approximators at equal size", x=0.01, ha="left")
    P.save(f, fig(EXP, "scaling"), "Noise-free N = 2,000. Networks: best of seeds per size (tanh). "
           "Chebyshev / spline: linear least squares, tensor product in 2-D.")

    # ---- Runge picture ------------------------------------------------------------
    target = get_target("f4")
    u = np.linspace(-1, 1, 2001).reshape(-1, 1)
    f_s = lambda v: target(v)  # domain is already [-1, 1]
    f2, axes = P.plt.subplots(1, 2, figsize=(12, 3.8))
    curves = [("equispaced nodes, degree 10", PolyInterpolant(10, "equispaced"), 1),
              ("equispaced nodes, degree 20", PolyInterpolant(20, "equispaced"), 7),
              ("Chebyshev nodes, degree 20", PolyInterpolant(20, "chebyshev"), 2)]
    lm_best = min((r for r in nn_rows if r["target"] == "f4" and r["model"] == SERIES[1]), key=lambda r: r["rmse"])
    runge_rows = []
    for lab, p, c in curves:
        p.fit_function(f_s)
        y = p.predict(u)
        axes[0].plot(u, np.clip(y, -1.5, 2.0), color=P.color(c), lw=1.5, label=lab)
        axes[0].plot(p.node_points(), f_s(p.node_points().reshape(-1, 1)), ls="none", marker="o", markersize=3.5,
                     color=P.color(c))
        e = np.abs(y - f_s(u)).ravel()
        axes[1].plot(u, np.maximum(e, 1e-12), color=P.color(c), lw=1.2, label=lab)
        runge_rows.append([lab, p.n_coef, fmt(float(e.max())), fmt(float(np.sqrt(np.mean(e ** 2))))])
    axes[0].plot(u, f_s(u), **P.truth_kw())
    axes[0].set_ylim(-1.0, 1.6)
    axes[0].set_title("Polynomial interpolation of Runge's function")
    axes[0].legend(loc="upper center", fontsize=7.5)
    axes[1].axhline(lm_best["max_err"], color=P.color(1), lw=1.2, ls=(0, (3, 2)))
    axes[1].text(-0.98, lm_best["max_err"] * 1.3, f"MLP {lm_best['hidden']} (LM) max error {fmt(lm_best['max_err'])}",
                 fontsize=8, color=P.TEXT_2)
    axes[1].set_yscale("log")
    axes[1].set_title("absolute error (log)")
    for ax in axes:
        ax.set_xlabel("x")
    f2.suptitle("E6 · The Runge phenomenon is a property of equispaced interpolation, not of polynomials", x=0.01, ha="left")
    P.save(f2, fig(EXP, "runge"))

    tbl = []
    for t in TARGETS:
        for name in SERIES:
            rs = [r for r in rows if r["target"] == t and r["model"] == name]
            b = min(rs, key=lambda r: r["rmse"])
            near = [r for r in rs if 80 <= r["n_params"] <= 400]
            nb = min(near, key=lambda r: r["rmse"]) if near else None
            tbl.append([t, name, fmt(b["rmse"]), b["n_params"],
                        f"{fmt(nb['rmse'])} @ {nb['n_params']}" if nb else "-"])
    body = (md_table(["target", "model", "best RMSE", "at size", "best with 80-400 params"], tbl)
            + "\n\nRunge function, interpolation:\n\n"
            + md_table(["interpolant", "coefficients", "max error", "RMSE"], runge_rows))
    write_summary(EXP, "E6 · Classical baselines at matched size", body, profile)


if __name__ == "__main__":
    main()
