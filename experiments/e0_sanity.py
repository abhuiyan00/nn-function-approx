"""E0 - Engine sanity and the universal approximation idea, made visible.

Question: does the engine compute correct gradients, and how does a single hidden
layer of tanh units build a function as a weighted sum of shifted "S" curves?

Outputs
    results/e0_sanity/gradcheck.json
    docs/figures/e0_sanity_gradcheck.png       finite-difference error per layer type
    docs/figures/e0_sanity_units.png           fits with 1, 2, 4, 8 hidden units + unit contributions
"""
from common import fig, fmt, md_table, parse_profile, save_json, write_summary

import numpy as np

from fapprox import plotting as P
from fapprox.data import make_dataset
from fapprox.functions import get_target
from fapprox.gradcheck import gradient_error, jacobian_error
from fapprox.metrics import evaluate, predict_original
from fapprox.model import build_kan, build_mlp
from fapprox.train import TrainConfig, fit

EXP = "e0_sanity"


def gradcheck_table() -> dict:
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, (64, 2))
    z = np.sin(3 * x[:, :1]) + x[:, 1:2] ** 2
    cases = {
        "tanh 2x16": build_mlp(2, [16, 16], "tanh", seed=1),
        "sigmoid 2x16": build_mlp(2, [16, 16], "sigmoid", seed=1),
        "silu 2x16": build_mlp(2, [16, 16], "silu", seed=1),
        "sine (SIREN) 2x16": build_mlp(2, [16, 16], "sine", seed=1, w0=5.0),
        "relu 2x16 (kink init)": build_mlp(2, [16, 16], "relu", seed=1, bias_init="kink"),
        "Fourier feat. + tanh": build_mlp(2, [16], "tanh", seed=1, fourier=(8, 1.0)),
        "KAN [2,4,1] G=5": build_kan([2, 4, 1], 5, seed=1, noise_scale=0.5),
    }
    return {k: dict(grad=gradient_error(n, x, z, n_probe=200), jac=jacobian_error(n, x, n_probe=50),
                    params=n.n_params) for k, n in cases.items()}


def main():
    profile, prof = parse_profile(__doc__)
    gc = gradcheck_table()
    save_json(EXP, gc, "gradcheck")

    f, ax = P.plt.subplots(figsize=(7.2, 3.2))
    names = list(gc)
    y = np.arange(len(names))
    ax.barh(y - 0.17, [max(gc[n]["grad"], 1e-16) for n in names], height=0.32, color=P.color(0), label="loss gradient")
    ax.barh(y + 0.17, [max(gc[n]["jac"], 1e-16) for n in names], height=0.32, color=P.color(1), label="Jacobian (for LM)")
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.axvline(1e-6, color=P.MUTED, lw=1)
    ax.set_xlabel("max finite-difference error (lower is better; ReLU: kinks crossed by the probe)")
    ax.set_title("E0 · Analytic derivatives agree with finite differences")
    ax.legend(loc="lower right")
    P.save(f, fig(EXP, "gradcheck"))

    # ---- universal approximation with H tanh units -----------------------------
    target = get_target("f1")
    ds = make_dataset(target, prof["n_train"], prof["n_val"], 0.0, 0)
    widths = [1, 2, 4, 8]
    f, axes = P.plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    xs = ds.x_test
    rows = []
    for ax, h in zip(axes, widths):
        best = None
        for seed in prof["seeds"] + (11, 12):  # few extra restarts: tiny nets are init-sensitive
            net = build_mlp(1, [h], "tanh", seed=seed)
            fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
                TrainConfig(optimizer="lm", epochs=prof["lm_iters"], patience=30))
            m = evaluate(net, ds)
            if best is None or m["rmse"] < best[1]["rmse"]:
                best = (net, m)
        net, m = best
        rows.append([h, net.n_params, fmt(m["rmse"]), fmt(m["max_err"])])
        # contribution of every hidden unit, in original units: ys.std * v_j * tanh(.)
        hid, out = net.layers[0], net.layers[1]
        a = hid.act.f(ds.xs.transform(xs) @ hid.W + hid.b)
        contrib = ds.ys.std * a * out.W[:, 0]
        contrib -= contrib.mean(axis=0)  # centred, so cancelling pairs stay on screen
        for j in range(h):
            ax.plot(xs, contrib[:, j], color=P.SEQ_BLUE[2], lw=1.0, alpha=0.9,
                    label="unit contributions" if j == 0 else None)
        ax.plot(xs, target(xs), **P.truth_kw())
        ax.plot(xs, predict_original(net, ds, xs), color=P.color(1), label="network (sum + bias)")
        ax.set_title(f"{h} hidden unit{'s' if h > 1 else ''} · {net.n_params} params")
        P.note(ax, f"RMSE {fmt(m['rmse'])}", "lower left")
        ax.set_xlabel("x")
        ax.set_ylim(-1.6, 1.6)
    axes[0].set_ylabel("y")
    h_, l_ = axes[0].get_legend_handles_labels()
    f.legend(h_, l_, loc="outside upper right", ncol=3)
    f.suptitle("E0 · One hidden layer = weighted sum of shifted tanh curves (target sin 2πx)", x=0.01, ha="left")
    P.save(f, fig(EXP, "units"), "Levenberg-Marquardt, best of 4 initialisations. Light-blue: v_j·tanh(w_j x + b_j) of each unit, centred; with 2 units the pair has huge opposite weights and leaves the plot.")

    body = "### Gradient check (64 random 2-D points)\n\n" + md_table(
        ["model", "params", "loss-gradient error", "Jacobian error"],
        [[k, v["params"], f"{v['grad']:.1e}", f"{v['jac']:.1e}"] for k, v in gc.items()]) + (
        "\n\nGradient error = max |numerical − analytic| / max |analytic| over 200 random coordinates."
        " A ReLU network can exceed 1e-6 only when a ±1e-6 probe pushes a pre-activation across 0"
        " (the derivative does not exist there); smooth layers are not affected.\n\n"
        "### Universal approximation, target f1 = sin(2πx)\n\n") + md_table(
        ["hidden units", "params", "test RMSE", "max error"], rows)
    write_summary(EXP, "E0 · Engine sanity and universal approximation", body, profile)


if __name__ == "__main__":
    main()
