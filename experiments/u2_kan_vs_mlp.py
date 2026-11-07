"""U2 (unique) - Kolmogorov-Arnold Networks vs MLPs, own B-spline KAN implementation.

Question: KANs (Liu et al., ICLR 2025) put learnable spline functions on the edges
instead of fixed activations on the nodes. Do they approximate better at the *same*
number of parameters, and can we read the learned function back?

Both models trained with the same optimiser (own Levenberg-Marquardt), same data.
    KAN  [d, 5, 1], grid G in {3, 5, 10, 20} (cubic splines), grid-range update at it. 25, 75
    KAN  [2, 1, 1] on f7 - the exact Kolmogorov-Arnold shape of exp(sin(pi x) + y^2)
    MLP  2 x H tanh, H in {4, 6, 8, 12, 16}
Extras: grid extension 3 -> 5 -> 10 -> 20 during training; learned edge functions.
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import time

import numpy as np

from funcprox import bspline
from funcprox import plotting as P
from funcprox.data import make_dataset
from funcprox.functions import get_target
from funcprox.layers import KANLayer
from funcprox.metrics import evaluate
from funcprox.model import build_kan, build_mlp
from funcprox.train import TrainConfig, fit

EXP = "u2_kan_vs_mlp"
TARGETS = {"f7": "f7 · exp(sin πx + y²)", "f5": "f5 · Franke", "f4": "f4 · Runge", "f6": "f6 · |x|"}


def run_model(spec: dict) -> dict:
    t = get_target(spec["target"])
    ds = make_dataset(t, spec["n_train"], spec["n_val"], 0.0, 0)
    if spec["model"].startswith("KAN"):
        net = build_kan(spec["widths"], spec["grid"], seed=spec["seed"])
        sched = {25: spec["grid"], 75: spec["grid"]}
    else:
        net = build_mlp(t.dim, [spec["h"]] * 2, "tanh", seed=spec["seed"])
        sched = {}
    cfg = TrainConfig(optimizer="lm", epochs=spec["iters"], patience=30, seed=spec["seed"], kan_grid_schedule=sched)
    t0 = time.perf_counter()
    h = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg)
    m = evaluate(net, ds)
    return dict(target=spec["target"], model=spec["model"], seed=spec["seed"], n_params=net.n_params,
                grid=spec.get("grid"), h=spec.get("h"), rmse=m["rmse"], max_err=m["max_err"],
                seconds=time.perf_counter() - t0, iters=len(h.epoch))


def edge_function(layer: KANLayer, i: int, o: int, x: np.ndarray) -> np.ndarray:
    B = bspline.basis(x.reshape(-1, 1).repeat(layer.n_in, axis=1), layer.knots, layer.k)[:, i, :]
    return layer.wb[i, o] * layer.base.f(x) + B @ layer.C[i, o]


def affine_r2(phi: np.ndarray, g: np.ndarray) -> tuple[float, np.ndarray]:
    A = np.stack([g, np.ones_like(g)], axis=1)
    coef = np.linalg.lstsq(A, phi, rcond=None)[0]
    fit_ = A @ coef
    return float(1 - np.sum((phi - fit_) ** 2) / np.sum((phi - phi.mean()) ** 2)), fit_


def main():
    profile, prof = parse_profile(__doc__)
    seeds = prof["seeds"]
    base = dict(n_train=prof["n_train"], n_val=prof["n_val"], iters=prof["lm_iters"])
    specs = []
    for t in TARGETS:
        d = get_target(t).dim
        for g in (3, 5, 10, 20):
            specs += [dict(target=t, model="KAN [d,5,1]", widths=[d, 5, 1], grid=g, seed=s, **base) for s in seeds]
        if t == "f7":
            for g in (5, 10, 20):
                specs += [dict(target=t, model="KAN [2,1,1]", widths=[2, 1, 1], grid=g, seed=s, **base) for s in seeds]
        for hh in (4, 6, 8, 12, 16):
            specs += [dict(target=t, model="MLP 2×H tanh", h=hh, seed=s, **base) for s in seeds]
    rows = run_many(specs, prof["workers"], fn=run_model, label=EXP)
    save_rows(EXP, rows)

    names = ["MLP 2×H tanh", "KAN [d,5,1]", "KAN [2,1,1]"]
    f, axes = P.plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, t in zip(axes, TARGETS):
        for i, name in enumerate(names):
            rs = [r for r in rows if r["target"] == t and r["model"] == name]
            if not rs:
                continue
            ps = sorted({r["n_params"] for r in rs})
            ax.plot(ps, [min(r["rmse"] for r in rs if r["n_params"] == p) for p in ps], label=name,
                    **P.marker_kw([1, 0, 2][i]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TARGETS[t])
        ax.set_xlabel("parameters")
    axes[0].set_ylabel("test RMSE (best of seeds)")
    h_, l_ = axes[0].get_legend_handles_labels()
    f.legend(h_, l_, loc="outside upper right", ncol=3)
    f.suptitle("U2 · KAN vs MLP at equal parameter count (both trained with Levenberg-Marquardt)", x=0.01, ha="left")
    P.save(f, fig(EXP, "params"), f"N = {prof['n_train']} noise-free; KAN cubic B-splines, grid G = 3, 5, 10, 20; "
           f"≤{prof['lm_iters']} LM iterations.")

    # ---- grid extension + interpretability on f7 ----------------------------------
    t = get_target("f7")
    ds = make_dataset(t, prof["n_train"], prof["n_val"], 0.0, 0)
    stage = prof["lm_iters"] // 4
    kan = build_kan([2, 5, 1], 3, seed=seeds[0])
    sched = {stage: 5, 2 * stage: 10, 3 * stage: 20}
    hk = fit(kan, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
             TrainConfig(optimizer="lm", epochs=4 * stage, patience=0, kan_grid_schedule=sched))
    mlp = build_mlp(2, [12, 12], "tanh", seed=seeds[0])
    hm = fit(mlp, ds.u_train, ds.z_train, ds.u_val, ds.z_val, TrainConfig(optimizer="lm", epochs=4 * stage, patience=0))
    small = build_kan([2, 1, 1], 10, seed=seeds[0])
    fit(small, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
        TrainConfig(optimizer="lm", epochs=prof["lm_iters"], patience=30, kan_grid_schedule={25: 10, 75: 10}))
    m_small = evaluate(small, ds)

    f, axes = P.plt.subplots(1, 4, figsize=(15, 3.7))
    sd = ds.ys.std
    ax = axes[0]
    ax.plot(hk.epoch, np.sqrt(hk.val_mse) * sd, color=P.color(0), label=f"KAN [2,5,1], G 3→5→10→20 ({kan.n_params} params at end)")
    ax.plot(hm.epoch, np.sqrt(hm.val_mse) * sd, color=P.color(1), label=f"MLP 2×12 ({mlp.n_params} params)")
    for it, g in sched.items():
        ax.axvline(it, color=P.MUTED, lw=1)
        ax.text(it + 3, 0.97, f"G={g}", transform=ax.get_xaxis_transform(), fontsize=7.5, color=P.TEXT_2, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("LM iteration")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Grid extension on f7")
    ax.legend(loc="lower left", fontsize=7)

    l1, l2 = small.layers
    xs = np.linspace(-1, 1, 400)
    checks = []
    for ax, (phi, g, lab) in zip(axes[1:3], [(edge_function(l1, 0, 0, xs), np.sin(np.pi * xs), "sin(πx)"),
                                              (edge_function(l1, 1, 0, xs), xs ** 2, "y²")]):
        r2, fitted = affine_r2(phi, g)
        checks.append((lab, r2))
        ax.plot(xs, phi, color=P.color(0), label="learned edge function φ")
        ax.plot(xs, fitted, **{**P.truth_kw(), "label": f"best a·{lab} + b  (1 − R² = {1 - r2:.1e})"})
        ax.set_xlabel("x" if lab == "sin(πx)" else "y")
        ax.legend(loc="lower center", fontsize=7.5)
    axes[1].set_title("KAN [2,1,1]: inner edge on x")
    axes[2].set_title("KAN [2,1,1]: inner edge on y")
    s_in = small.layers[0].forward(ds.u_test)
    lo, hi = float(s_in.min()), float(s_in.max())
    ss = np.linspace(lo, hi, 400)
    phi_o = edge_function(l2, 0, 0, ss)
    # outer function should be affine in exp(s / a) - read a from the inner fit
    best = max(((a, affine_r2(phi_o, np.exp(ss / a))[0]) for a in np.linspace(0.05, 5, 400)), key=lambda v: v[1])
    r2o, fo = affine_r2(phi_o, np.exp(ss / best[0]))
    checks.append(("exp(s / a)", r2o))
    axes[3].plot(ss, phi_o, color=P.color(0), label="learned outer function")
    axes[3].plot(ss, fo, **{**P.truth_kw(), "label": f"best a·exp(s/{best[0]:.2f}) + b  (1 − R² = {1 - r2o:.1e})"})
    axes[3].set_xlabel("s = φ_x(x) + φ_y(y) + bias")
    axes[3].set_title("KAN [2,1,1]: outer edge")
    axes[3].legend(loc="upper left", fontsize=7.5)
    f.suptitle(f"U2 · Grid extension, and reading the formula back from a KAN [2,1,1] "
               f"({small.n_params} params, test RMSE {fmt(m_small['rmse'])})", x=0.01, ha="left")
    P.save(f, fig(EXP, "interpret"), "Edge functions in scaled units; the dashed curve is an affine fit of the "
           "candidate symbolic form - 1 − R² ≈ 0 means the KAN found that form.")

    tbl = []
    for t in TARGETS:
        for name in names:
            rs = [r for r in rows if r["target"] == t and r["model"] == name]
            if not rs:
                continue
            b = min(rs, key=lambda r: r["rmse"])
            near = [r for r in rs if 100 <= r["n_params"] <= 250]
            nb = min(near, key=lambda r: r["rmse"]) if near else None
            tbl.append([t, name, fmt(b["rmse"]), b["n_params"], f"{fmt(nb['rmse'])} @ {nb['n_params']}" if nb else "-",
                        f"{np.median([r['seconds'] for r in rs]):.1f}"])
    body = (md_table(["target", "model", "best RMSE", "at params", "best with 100-250 params", "median s/run"], tbl)
            + "\n\nSymbolic check of KAN [2,1,1] on f7 (affine fit R² of each learned edge):\n\n"
            + md_table(["edge", "candidate", "1 − R²"], [["inner x", "sin(πx)", f"{1 - checks[0][1]:.1e}"],
                                                        ["inner y", "y²", f"{1 - checks[1][1]:.1e}"],
                                                        ["outer", f"exp(s/{best[0]:.2f})", f"{1 - checks[2][1]:.1e}"]])
            + f"\n\nGrid extension run: final validation RMSE KAN {fmt(np.sqrt(hk.val_mse[-1]) * sd)} "
              f"({kan.n_params} params) vs MLP 2×12 {fmt(np.sqrt(hm.val_mse[-1]) * sd)} ({mlp.n_params} params).")
    write_summary(EXP, "U2 · KAN vs MLP", body, profile)


if __name__ == "__main__":
    main()
