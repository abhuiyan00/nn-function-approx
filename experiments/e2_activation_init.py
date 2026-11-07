"""E2 - Activation function and initialisation.

Question: which neuron non-linearity suits which target, and how much does the
starting point (weight/bias initialisation) matter?

Variants (2 x 32 hidden): tanh, sigmoid, SiLU, ReLU (zero bias), ReLU (random kink
init), SIREN sin(w0 z) with w0 = 10 and 30. Targets: f1 smooth, f4 Runge, f6 kink |x|.
Extra: 1 x 32 ReLU with zero vs kink bias on f1 (all hinges at x = 0 vs spread out).
"""
from common import fig, fmt, md_table, parse_profile, run_many, run_one, save_rows, write_summary

import numpy as np

from funcprox import plotting as P

EXP = "e2_activation_init"
TARGETS = {"f1": "f1 · sin(2πx)", "f4": "f4 · Runge", "f6": "f6 · |x| (kink)"}
VARIANTS = [
    ("tanh", dict(act="tanh")),
    ("sigmoid", dict(act="sigmoid")),
    ("SiLU", dict(act="silu")),
    ("ReLU, zero bias", dict(act="relu")),
    ("ReLU, kink init", dict(act="relu", bias_init="kink")),
    ("SIREN w0=10", dict(act="sine", w0=10.0)),
    ("SIREN w0=30", dict(act="sine", w0=30.0)),
]


def main():
    profile, prof = parse_profile(__doc__)
    base = dict(n_train=prof["n_train"], n_val=prof["n_val"], epochs=prof["epochs"], patience=prof["patience"])
    specs = [dict(target=t, variant=name, seed=s, **kw, **base)
             for t in TARGETS for name, kw in VARIANTS for s in prof["seeds"]]
    stall = [dict(target="f1", hidden=(32,), act="relu", bias_init=b, variant=f"1x32 ReLU {b or 'zero'}",
                  seed=s, **base) for b in (None, "kink") for s in prof["seeds"]]
    rows = run_many(specs + stall, prof["workers"], label=EXP)
    save_rows(EXP, rows)
    main_rows = rows[: len(specs)]
    stall_rows = rows[len(specs):]

    # ---- dot plot: variant x target -------------------------------------------
    f, axes = P.plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    names = [n for n, _ in VARIANTS]
    for ax, t in zip(axes, TARGETS):
        for i, n in enumerate(names):
            vals = [r["rmse"] for r in main_rows if r["target"] == t and r["variant"] == n]
            ax.plot([min(vals), max(vals)], [i, i], color=P.color(i), lw=2, alpha=0.5)
            ax.plot(vals, [i] * len(vals), ls="none", **P.marker_kw(i))
            ax.text(max(vals) * 1.25, i, fmt(float(np.mean(vals))), va="center", fontsize=7.5, color=P.TEXT_2)
        ax.set_xscale("log")
        ax.set_title(TARGETS[t])
        ax.set_xlabel("test RMSE (dots = seeds)")
        ax.set_xlim(right=ax.get_xlim()[1] * 4)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    f.suptitle("E2 · Activation and initialisation, 2×32 network", x=0.01, ha="left")
    P.save(f, fig(EXP, "dotplot"), f"Adam lr 1e-3, batch 64, ≤{prof['epochs']} epochs; profile {profile}. Text = mean RMSE.")

    # ---- fits: ReLU hinge placement and the kink ------------------------------
    f, axes = P.plt.subplots(1, 3, figsize=(13, 3.6))
    seed = prof["seeds"][0]
    for i, b in enumerate((None, "kink")):
        r = run_one(dict(target="f1", hidden=(32,), act="relu", bias_init=b, seed=seed, return_pred=True, **base))
        axes[0].plot(r["_x"], r["_pred"], color=P.color(3 + i),
                     label=f"ReLU {'kink init' if b else 'zero bias'} · RMSE {fmt(r['rmse'])}")
    axes[0].plot(r["_x"], r["_true"], **P.truth_kw())
    axes[0].set_title("f1, 1×32 ReLU: where do the hinges start?")
    axes[0].legend(loc="lower left", fontsize=7.5)
    for j, (n, kw) in enumerate([("tanh", dict(act="tanh")), ("ReLU, kink init", dict(act="relu", bias_init="kink")),
                                 ("SIREN w0=10", dict(act="sine", w0=10.0))]):
        r = run_one(dict(target="f6", seed=seed, return_pred=True, **kw, **base))
        x, e = r["_x"].ravel(), (r["_pred"] - r["_true"]).ravel()
        sel = np.abs(x) < 0.15
        axes[1].plot(x[sel], r["_pred"].ravel()[sel], color=P.color([0, 4, 5][j]), label=n)
        axes[2].plot(x, np.abs(e), color=P.color([0, 4, 5][j]), lw=1.2, label=n)
    axes[1].plot(x[sel], np.abs(x[sel]), **P.truth_kw())
    axes[1].set_title("f6 = |x|, zoom at the kink")
    axes[1].legend(loc="upper center", fontsize=7.5)
    axes[2].set_yscale("log")
    axes[2].set_title("f6 absolute error along x")
    for ax in axes:
        ax.set_xlabel("x")
    f.suptitle("E2 · Piecewise-linear units fit kinks; smooth units round them off", x=0.01, ha="left")
    P.save(f, fig(EXP, "fits"))

    tbl = []
    for n in names:
        tbl.append([n] + [fmt(float(np.mean([r["rmse"] for r in main_rows if r["target"] == t and r["variant"] == n])))
                          for t in TARGETS])
    st = {b: [r for r in stall_rows if r["bias_init"] == b] for b in (None, "kink")}
    body = ("Mean test RMSE, 2×32 network.\n\n" + md_table(["variant"] + list(TARGETS), tbl)
            + "\n\nReLU 1×32 on f1 (hinge placement at initialisation):\n\n"
            + md_table(["bias init", "RMSE per seed", "R² per seed"],
                       [[b or "zero (all kinks at x=0)", ", ".join(fmt(r["rmse"]) for r in st[b]),
                         ", ".join(f"{r['r2']:.4f}" for r in st[b])] for b in st]))
    write_summary(EXP, "E2 · Activation and initialisation", body, profile)


if __name__ == "__main__":
    main()
