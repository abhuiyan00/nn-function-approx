"""E7 - The missing middle: how close can a curve get where it saw no data?

No training or validation point falls inside a hole in the middle of the domain; the
exact function still scores the whole range. Varied: hole width (share of each axis),
noise, and the model. Targets m1-m5 give the hole different jobs: continue a wave (m1),
guess a hidden peak (m2), restore a hidden kink (m3), recover a simple shape (m4), fill a
surface (m5). Each model's closeness is reported inside the hole and outside it, next to
the choice a user could make from validation points alone.
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import numpy as np

from funcprox import plotting as P
from funcprox.baselines import ChebyshevLS, SplineLS
from funcprox.custom import make_target
from funcprox.data import make_dataset
from funcprox.functions import get_target
from funcprox.metrics import regression_metrics

EXP = "e7_missing_middle"
TARGETS = {"m1": "m1 · sine", "m2": "m2 · hidden peak", "m3": "m3 · hidden kink", "m4": "m4 · cubic", "m5": "m5 · surface"}
WIDTHS = (10, 20, 30, 40)   # % of each axis
SIGMAS = (0.0, 0.05)
SHOWN_WIDTH, N_TRAIN, N_VAL = 30, 1000, 250
CLASSICAL = ("Chebyshev LS", "cubic spline LS")
MODELS = {
    "MLP tanh · LM": dict(hidden=(16, 16), optimizer="lm"),
    "MLP tanh · Adam": dict(hidden=(32, 32)),
    "MLP ReLU · Adam": dict(hidden=(32, 32), act="relu", bias_init="kink"),
    "Fourier + tanh · Adam": dict(hidden=(32, 32), fourier=(16, 2.0)),
    "SIREN ω₀=10 · Adam": dict(hidden=(32, 32), act="sine", w0=10.0),
    "KAN · LM": dict(arch="kan", grid=10, optimizer="lm", kan_grid_schedule={25: 10, 75: 10}),
}
SIZES = {1: ((ChebyshevLS, (3, 5, 8, 12, 16, 24, 32, 48, 64, 96)), (SplineLS, (6, 8, 12, 16, 24, 32, 48, 64, 96, 128))),
         2: ((ChebyshevLS, (2, 3, 4, 6, 8, 10, 12, 16, 20)), (SplineLS, (5, 6, 8, 10, 12, 14, 16, 20)))}


def hole(key: str, pct: int) -> list[tuple[float, float]]:
    """Centred, ``pct`` % of every axis of the target."""
    out = []
    for lo, hi in get_target(key).domain:
        middle, half = (lo + hi) / 2, (hi - lo) * pct / 200
        out.append((middle - half, middle + half))
    return out


def classical_rows(prof) -> list[dict]:
    """The same rule as the networks: the size is chosen on validation points outside the hole."""
    rows = []
    for key in TARGETS:
        for pct in WIDTHS:
            for sigma in SIGMAS:
                target = make_target(target=key, gap=hole(key, pct))
                ds = make_dataset(target, prof["n_train"], prof["n_val"], sigma, 0)
                hidden = target.in_gap(ds.x_test)
                for name, (model_class, sizes) in zip(CLASSICAL, SIZES[target.dim]):
                    fits = []
                    for size in sizes:
                        if fits and fits[-1][1].n_coef > prof["n_train"] // 2:
                            break  # least squares turns unstable beyond this, as in the lab
                        try:
                            model = model_class(size).fit(ds.u_train, ds.y_train)
                        except np.linalg.LinAlgError:
                            continue
                        fits.append((float(np.mean((model.predict(ds.u_val) - ds.y_val) ** 2)), model))
                    val, best = min(fits, key=lambda f: f[0])
                    m = regression_metrics(best.predict(ds.u_test), ds.y_test, ds.u_test, hidden=hidden)
                    rows.append(dict(target=key, model=name, gap_pct=pct, sigma=sigma, n_params=best.n_coef, val_mse_noisy=val,
                                     _x=ds.x_test, _pred=best.predict(ds.u_test), _true=ds.y_test, **m))
    return rows


def network_specs(prof) -> list[dict]:
    specs = []
    for key in TARGETS:
        dim = get_target(key).dim
        for pct in WIDTHS:
            for sigma in SIGMAS:
                for name, model in MODELS.items():
                    spec = dict(model, target=key, model=name, gap_pct=pct, gap=hole(key, pct), sigma=sigma,
                                n_train=prof["n_train"], n_val=prof["n_val"], return_pred=pct == SHOWN_WIDTH and sigma == 0.0)
                    if name == "KAN · LM":
                        spec["widths"] = (dim, 5, 1)
                    if spec.get("optimizer") == "lm":
                        spec.update(epochs=prof["lm_iters"], patience=30)
                    else:
                        spec.update(epochs=prof["epochs"], patience=prof["patience"])
                    specs += [dict(spec, seed=s) for s in prof["seeds"]]
    return specs


def medians(rows: list[dict], *keys: str, metric: str = "rmse_gap") -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r[metric])
    return {k: float(np.median(v)) for k, v in groups.items()}


def fits_figure(rows: list[dict]) -> None:
    keys = [k for k in TARGETS if get_target(k).dim == 1]
    shown = ("MLP tanh · LM", "MLP ReLU · Adam", "KAN · LM", "Chebyshev LS")
    f, axes = P.plt.subplots(1, len(keys), figsize=(15, 3.9))
    for ax, key in zip(axes, keys):
        (lo, hi), = hole(key, SHOWN_WIDTH)
        ax.axvspan(lo, hi, color=P.TEXT_2, alpha=0.1, lw=0, label="missing middle")
        truth = None
        for i, name in enumerate(shown):
            rs = [r for r in rows if r["target"] == key and r["model"] == name and r["gap_pct"] == SHOWN_WIDTH and r["sigma"] == 0.0 and "_x" in r]
            r = min(rs, key=lambda v: v["rmse"])
            truth = r
            order = np.argsort(r["_x"].ravel())
            ax.plot(r["_x"].ravel()[order], r["_pred"].ravel()[order], color=P.color(i), lw=1.3, label=name, zorder=3)
        order = np.argsort(truth["_x"].ravel())
        ax.plot(truth["_x"].ravel()[order], truth["_true"].ravel()[order], **P.truth_kw())
        span = truth["_true"].max() - truth["_true"].min()
        ax.set_ylim(truth["_true"].min() - 0.2 * span, truth["_true"].max() + 0.2 * span)
        ax.set_title(TARGETS[key])
        ax.set_xlabel("x")
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles, labels, loc="outside upper right", ncol=6)
    f.suptitle(f"E7 · Curves across a hole of {SHOWN_WIDTH} % of the range (best seed, no noise)", x=0.01, ha="left")
    P.save(f, fig(EXP, "fits"), f"N = {N_TRAIN}. No training or validation point inside the shaded band; a curve is cut off where it leaves the plot.")


def width_figure(rows: list[dict]) -> None:
    names = [*MODELS, *CLASSICAL]
    table = medians(rows, "target", "model", "gap_pct", "sigma")
    f, axes = P.plt.subplots(len(SIGMAS), len(TARGETS), figsize=(15, 6.2), sharex=True)
    for row, sigma in enumerate(SIGMAS):
        for col, key in enumerate(TARGETS):
            ax = axes[row][col]
            for i, name in enumerate(names):
                ax.plot(WIDTHS, [table[(key, name, w, sigma)] for w in WIDTHS], label=name, **P.marker_kw(i))
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(TARGETS[key])
            else:
                ax.set_xlabel("hole width, % of the range")
        axes[row][0].set_ylabel(f"σ = {sigma:g}: RMSE in the hole")
    handles, labels = axes[0][0].get_legend_handles_labels()
    f.legend(handles, labels, loc="outside upper right", ncol=4)
    f.suptitle("E7 · Error inside the hole against its width", x=0.01, ha="left")
    P.save(f, fig(EXP, "width"), f"N = {N_TRAIN}. Networks: median of seeds. Classical fits are deterministic for a data set.")


def summary_tables(rows: list[dict]) -> str:
    in_hole = medians(rows, "target", "model", "gap_pct", "sigma")
    outside = medians(rows, "target", "model", "gap_pct", "sigma", metric="rmse_seen")
    overall = medians(rows, "target", "model", "gap_pct", "sigma", metric="rmse")
    val = medians(rows, "target", "model", "gap_pct", "sigma", metric="val_mse_noisy")
    table = []
    for key in TARGETS:
        for sigma in SIGMAS:
            cell = {m: in_hole[(key, m, SHOWN_WIDTH, sigma)] for m in [*MODELS, *CLASSICAL]}
            networks = {m: cell[m] for m in MODELS}
            best, best_net = min(cell, key=cell.get), min(networks, key=networks.get)
            scores = {m: val[(key, m, SHOWN_WIDTH, sigma)] for m in cell}
            blind = min(scores, key=scores.get)
            table.append([key, f"{sigma:g}", best, fmt(cell[best]), best_net, fmt(cell[best_net]),
                          fmt(outside[(key, best_net, SHOWN_WIDTH, sigma)]), blind, fmt(cell[blind])])
    closest = md_table(["target", "σ", "closest in the hole", "RMSE", "closest network", "RMSE in the hole", "its RMSE with data",
                        "best on validation", "its RMSE in the hole"], table)

    grown = []
    for key in TARGETS:
        for sigma in SIGMAS:
            nets = {m: {w: overall[(key, m, w, sigma)] for w in WIDTHS} for m in MODELS}
            best = min(nets, key=lambda m: nets[m][SHOWN_WIDTH])
            grown.append([key, f"{sigma:g}", best, *[fmt(in_hole[(key, best, w, sigma)]) for w in WIDTHS]])
    width = md_table(["target", "σ", "closest network at 30 %", *[f"hole {w} %" for w in WIDTHS]], grown)
    return (f"Hole of {SHOWN_WIDTH} % of each axis, medians over seeds:\n\n{closest}\n\n"
            f"RMSE in the hole as it widens:\n\n{width}")


def main():
    profile, prof = parse_profile(__doc__)
    prof = dict(prof, n_train=N_TRAIN * prof["scale"], n_val=N_VAL * prof["scale"])
    networks = run_many(network_specs(prof), prof["workers"], label=EXP)
    classical = classical_rows(prof)
    rows = networks + classical
    save_rows(EXP, rows)
    fits_figure(rows)
    width_figure(rows)
    write_summary(EXP, "E7 · The missing middle", summary_tables(rows), profile)


if __name__ == "__main__":
    main()
