"""E1 - Topology: depth x width at equal parameter budget.

Question: for a fixed number of parameters, is it better to go deep or wide?
Grid: depth {1, 2, 3} x width {4, 8, 16, 32, 64}, tanh, Adam, targets f1, f4, f5.

Outputs
    docs/figures/e1_architecture_params.png   test RMSE vs parameter count, one line per depth
    docs/figures/e1_architecture_franke.png   exact Franke surface, best fit, error map
"""
from common import (agg, fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary)

import numpy as np

from fapprox import plotting as P

EXP = "e1_architecture"
TARGETS = ["f1", "f4", "f5"]
DEPTHS = [1, 2, 3]
WIDTHS = [4, 8, 16, 32, 64]
TITLES = {"f1": "f1 · sin(2πx)", "f4": "f4 · Runge 1/(1+25x²)", "f5": "f5 · Franke (2-D)"}


def main():
    profile, prof = parse_profile(__doc__)
    specs = [dict(target=t, hidden=(w,) * d, depth=d, width=w, seed=s, n_train=prof["n_train"],
                  n_val=prof["n_val"], epochs=prof["epochs"], patience=prof["patience"])
             for t in TARGETS for d in DEPTHS for w in WIDTHS for s in prof["seeds"]]
    rows = run_many(specs, prof["workers"], label=EXP)
    save_rows(EXP, rows)

    f, axes = P.plt.subplots(1, 3, figsize=(13, 3.8))
    stats = agg(rows, ("target", "depth", "width"))
    for ax, t in zip(axes, TARGETS):
        for i, d in enumerate(DEPTHS):
            pts = sorted((r["n_params"], *stats[(t, d, r["width"])][:3]) for r in rows
                         if r["target"] == t and r["depth"] == d and r["seed"] == prof["seeds"][0])
            p, mean, lo, hi = map(np.array, zip(*pts))
            ax.fill_between(p, lo, hi, color=P.color(i), alpha=0.15, lw=0)
            ax.plot(p, mean, label=f"{d} hidden layer{'s' if d > 1 else ''}", **P.marker_kw(i))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLES[t])
        ax.set_xlabel("number of parameters")
    axes[0].set_ylabel("test RMSE vs exact f")
    axes[0].legend(loc="lower left")
    f.suptitle("E1 · Depth vs width: test error against parameter count (band = seed min-max)", x=0.01, ha="left")
    P.save(f, fig(EXP, "params"), f"tanh, Adam lr 1e-3, batch 64, ≤{prof['epochs']} epochs, patience {prof['patience']}, "
           f"best checkpoint; {len(prof['seeds'])} seeds; profile {profile}.")

    # ---- Franke surface: exact, best network, error -----------------------------
    best = min((r for r in rows if r["target"] == "f5"), key=lambda r: r["rmse"])
    from common import run_one
    r = run_one(dict(target="f5", hidden=tuple(json_list(best["hidden"])), seed=best["seed"],
                     n_train=prof["n_train"], n_val=prof["n_val"], epochs=prof["epochs"],
                     patience=prof["patience"], return_pred=True))
    m = int(round(np.sqrt(len(r["_x"]))))
    X = r["_x"][:, 0].reshape(m, m)
    Y = r["_x"][:, 1].reshape(m, m)
    T = r["_true"].reshape(m, m)
    Z = r["_pred"].reshape(m, m)
    f, axes = P.plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, data, title, cmap in [(axes[0], T, "exact Franke function", P.SEQ),
                                  (axes[1], Z, f"network {best['hidden']} · {best['n_params']} params", P.SEQ)]:
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto", vmin=T.min(), vmax=T.max())
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(False)
    f.colorbar(im, ax=axes[:2], shrink=0.85, label="f(x, y)")
    err = Z - T
    lim = float(np.max(np.abs(err)))
    im = axes[2].pcolormesh(X, Y, err, cmap=P.DIV, shading="auto", vmin=-lim, vmax=lim)
    axes[2].set_title(f"error (RMSE {fmt(r['rmse'])}, max {fmt(r['max_err'])})")
    axes[2].set_aspect("equal")
    axes[2].grid(False)
    f.colorbar(im, ax=axes[2], shrink=0.85, label="network − exact")
    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    f.suptitle("E1 · Best network on the 2-D Franke function", x=0.01, ha="left")
    P.save(f, fig(EXP, "franke"))

    table = []
    for t in TARGETS:
        for d in DEPTHS:
            table.append([t, d] + [fmt(stats[(t, d, w)][0]) for w in WIDTHS])
    # depth advantage at roughly equal size: 1 x 32 (97 params) vs 2 x 8 (97 params) in 1-D
    eq = []
    for t in ("f1", "f4"):
        a, b = stats[(t, 1, 32)][0], stats[(t, 2, 8)][0]
        eq.append(f"- {t}: 1×32 (97 params) RMSE {fmt(a)} vs 2×8 (97 params) {fmt(b)} → ratio {a / b:.2g}")
    body = ("Mean test RMSE over seeds (columns: width per hidden layer).\n\n"
            + md_table(["target", "depth"] + [f"w={w}" for w in WIDTHS], table)
            + "\n\nSame parameter count, different shape:\n\n" + "\n".join(eq)
            + f"\n\nBest Franke network: {best['hidden']} ({best['n_params']} params), RMSE {fmt(best['rmse'])}.")
    write_summary(EXP, "E1 · Architecture (depth × width)", body, profile)


def json_list(s):
    import json
    return json.loads(s)


if __name__ == "__main__":
    main()
