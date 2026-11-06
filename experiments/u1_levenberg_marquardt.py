"""U1 (unique) - Second-order training with an own Levenberg-Marquardt engine.

Question: E6 shows that Adam-trained MLPs stay orders of magnitude behind classical
approximators. Is that a limit of the *network* or of the *optimiser*?

    theta <- theta - (J^T J + mu I)^-1 J^T e        (J: N x P Jacobian, own backprop)

Compared on f4 (Runge) and f5 (Franke), 2 x 16 tanh (321 / 337 params):
    Adam (lr 1e-3, batch 64) | LM from random init | hybrid: 100 Adam epochs, then LM
"""
from common import DEFAULTS, fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import time

import numpy as np

from fapprox import plotting as P
from fapprox.data import make_dataset
from fapprox.functions import get_target
from fapprox.metrics import evaluate
from fapprox.model import build_mlp
from fapprox.train import TrainConfig, fit

EXP = "u1_levenberg_marquardt"
METHODS = ["Adam", "LM", "Adam → LM"]


def run_method(spec: dict) -> dict:
    t = get_target(spec["target"])
    ds = make_dataset(t, spec["n_train"], spec["n_val"], 0.0, 0)
    net = build_mlp(t.dim, [16, 16], "tanh", seed=spec["seed"])
    sd = ds.ys.std ** 2
    curves = dict(wall=[], val=[], mu=[], phase=[])
    t0 = time.perf_counter()

    def run(cfg, phase):
        h = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg)
        off = curves["wall"][-1] if curves["wall"] else 0.0
        curves["wall"] += [off + w for w in h.wall]
        curves["val"] += [np.sqrt(v * sd) for v in h.val_mse]
        curves["mu"] += h.mu or [np.nan] * len(h.wall)
        curves["phase"] += [phase] * len(h.wall)

    adam = TrainConfig(optimizer="adam", epochs=spec["epochs"], patience=spec["patience"], seed=spec["seed"])
    lm = TrainConfig(optimizer="lm", epochs=spec["lm_iters"], patience=30, seed=spec["seed"])
    if spec["method"] == "Adam":
        run(adam, "adam")
    elif spec["method"] == "LM":
        run(lm, "lm")
    else:
        adam.epochs, adam.patience = 100, 0
        run(adam, "adam")
        run(lm, "lm")
    m = evaluate(net, ds)
    return dict(target=spec["target"], method=spec["method"], seed=spec["seed"], n_params=net.n_params,
                rmse=m["rmse"], max_err=m["max_err"], seconds=time.perf_counter() - t0, _curves=curves)


def main():
    profile, prof = parse_profile(__doc__)
    seeds = list(range(1, 7 if profile == "quick" else 11))
    specs = [dict(target=t, method=m, seed=s, n_train=prof["n_train"], n_val=prof["n_val"],
                  epochs=prof["epochs"], patience=prof["patience"], lm_iters=prof["lm_iters"])
             for t in ("f4", "f5") for m in METHODS for s in seeds]
    rows = run_many(specs, prof["workers"], fn=run_method, label=EXP)
    save_rows(EXP, rows)

    f, axes = P.plt.subplots(2, 2, figsize=(12.5, 7.4))
    for k, t in enumerate(("f4", "f5")):
        ax = axes[0, k]
        for i, m in enumerate(METHODS):
            r = min((r for r in rows if r["target"] == t and r["method"] == m and r["seed"] == seeds[0]),
                    key=lambda r: r["rmse"])
            c = r["_curves"]
            ax.plot(np.array(c["wall"]) + 1e-2, c["val"], color=P.color(i), label=f"{m} · test RMSE {fmt(r['rmse'])}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock seconds (1 CPU core)")
        ax.set_ylabel("validation RMSE")
        ax.set_title(f"{'f4 · Runge' if t == 'f4' else 'f5 · Franke (2-D)'}: training progress, seed {seeds[0]}")
        ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1, 0]
    for k, t in enumerate(("f4", "f5")):
        r = next(r for r in rows if r["target"] == t and r["method"] == "LM" and r["seed"] == seeds[0])
        mu = np.array(r["_curves"]["mu"], dtype=float)
        ax.plot(np.arange(1, len(mu) + 1), mu, color=P.color(5 + k), label=f"{t}")
    ax.set_yscale("log")
    ax.set_xlabel("LM iteration")
    ax.set_ylabel("damping μ after the step")
    ax.set_title("LM damping: large μ = gradient step, small μ = Gauss-Newton")
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    for k, t in enumerate(("f4", "f5")):
        for i, m in enumerate(METHODS):
            v = [r["rmse"] for r in rows if r["target"] == t and r["method"] == m]
            x = k * 4 + i
            ax.plot([x] * len(v), v, ls="none", **{**P.marker_kw(i), "alpha": 0.85})
            ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=P.TEXT, lw=1.5)
    ax.set_xticks([1, 5], ["f4 · Runge", "f5 · Franke"])
    ax.set_yscale("log")
    ax.set_ylabel("test RMSE")
    ax.set_title(f"Robustness over {len(seeds)} initialisations (bar = median)")
    for i, m in enumerate(METHODS):
        ax.plot([], [], ls="none", label=m, **P.marker_kw(i))
    ax.legend(loc="upper right", fontsize=7.5)
    f.suptitle("U1 · Levenberg-Marquardt vs Adam on the same 2×16 tanh network", x=0.01, ha="left")
    P.save(f, fig(EXP, "lm_vs_adam"), f"N = {prof['n_train']} noise-free; LM full batch ≤{prof['lm_iters']} iterations; "
           "hybrid = 100 Adam epochs, then LM.")

    tbl = []
    for t in ("f4", "f5"):
        for m in METHODS:
            rs = [r for r in rows if r["target"] == t and r["method"] == m]
            v = np.array([r["rmse"] for r in rs])
            tbl.append([t, m, fmt(float(np.median(v))), fmt(float(v.min())), fmt(float(v.max())),
                        f"{np.median([r['seconds'] for r in rs]):.1f}"])
    ratio = {t: np.median([r["rmse"] for r in rows if r["target"] == t and r["method"] == "Adam"])
             / np.median([r["rmse"] for r in rows if r["target"] == t and r["method"] == "LM"]) for t in ("f4", "f5")}
    body = (md_table(["target", "method", "median RMSE", "best", "worst", "median seconds"], tbl)
            + f"\n\nMedian Adam/LM error ratio: f4 ×{ratio['f4']:.0f}, f5 ×{ratio['f5']:.0f} "
              f"(same network, same data; {len(seeds)} seeds).")
    write_summary(EXP, "U1 · Levenberg-Marquardt (second-order training)", body, profile)


if __name__ == "__main__":
    main()
