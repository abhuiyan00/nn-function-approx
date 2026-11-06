"""E3 - Training parameters: learning rate x batch size, and the optimiser.

Question: how sensitive is the result to the learning rate and the mini-batch size,
and how do SGD, momentum, Nesterov, Adam and Levenberg-Marquardt compare?
Network 2 x 32 tanh (heat map) / 2 x 16 tanh (optimisers), target f4 (Runge).
Fixed budget of epochs: a smaller batch means more parameter updates.
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import numpy as np

from fapprox import plotting as P

EXP = "e3_training"
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BATCHES = [16, 64, 256, 0]  # 0 = full batch
OPT_LRS = {"sgd": [1e-2, 3e-2, 1e-1], "momentum": [1e-3, 3e-3, 1e-2], "nesterov": [1e-3, 3e-3, 1e-2],
           "adam": [3e-4, 1e-3, 3e-3]}
OPT_NAMES = {"sgd": "SGD", "momentum": "SGD + momentum 0.9", "nesterov": "Nesterov 0.9", "adam": "Adam", "lm": "Levenberg-Marquardt"}


def main():
    profile, prof = parse_profile(__doc__)
    base = dict(target="f4", n_train=prof["n_train"], n_val=prof["n_val"], epochs=prof["epochs"],
                patience=prof["patience"])
    grid = [dict(lr=lr, batch=b, seed=s, **base) for lr in LRS for b in BATCHES for s in prof["seeds"]]
    opt = [dict(optimizer=o, lr=lr, hidden=(16, 16), seed=prof["seeds"][0], return_hist=True, **base)
           for o, lrs in OPT_LRS.items() for lr in lrs]
    opt.append(dict(optimizer="lm", hidden=(16, 16), seed=prof["seeds"][0], return_hist=True,
                    **{**base, "epochs": prof["lm_iters"], "patience": 30}))
    rows = run_many(grid + opt, prof["workers"], label=EXP)
    g_rows, o_rows = rows[: len(grid)], rows[len(grid):]
    save_rows(EXP, g_rows, "lr_batch")
    save_rows(EXP, o_rows, "optimizers")

    # ---- heat map ---------------------------------------------------------------
    M = np.array([[np.exp(np.mean(np.log([r["rmse"] for r in g_rows if r["lr"] == lr and r["batch"] == b])))
                   for lr in LRS] for b in BATCHES])
    U = np.array([[np.mean([r["updates"] for r in g_rows if r["lr"] == lr and r["batch"] == b]) for lr in LRS]
                  for b in BATCHES])
    f, ax = P.plt.subplots(figsize=(7.6, 3.6))
    im = ax.imshow(np.log10(M), cmap=P.SEQ, aspect="auto")
    for i in range(len(BATCHES)):
        for j in range(len(LRS)):
            dark = np.log10(M[i, j]) > np.log10(M).min() + 0.6 * np.ptp(np.log10(M))
            ax.text(j, i, f"{fmt(M[i, j])}\n{U[i, j] / 1000:.0f}k upd", ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark else P.TEXT)
    ax.set_xticks(range(len(LRS)), [f"{v:g}" for v in LRS])
    ax.set_yticks(range(len(BATCHES)), [str(b) if b else f"full ({prof['n_train']})" for b in BATCHES])
    ax.set_xlabel("learning rate (Adam)")
    ax.set_ylabel("batch size")
    ax.grid(False)
    f.colorbar(im, ax=ax, label="log10 test RMSE (geo-mean of seeds)")
    ax.set_title("E3 · Learning rate × batch size, f4 Runge, 2×32 tanh")
    P.save(f, fig(EXP, "lr_batch"), f"≤{prof['epochs']} epochs, patience {prof['patience']}; 'upd' = parameter updates actually made.")

    # ---- optimiser curves ---------------------------------------------------------
    best = {}
    for r in o_rows:
        o = r["optimizer"]
        if o not in best or r["rmse"] < best[o]["rmse"]:
            best[o] = r
    f, axes = P.plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for i, o in enumerate(["sgd", "momentum", "nesterov", "adam", "lm"]):
        r = best[o]
        h = r["_hist"]
        rm = np.sqrt(np.array(h["val_mse"]))
        lab = f"{OPT_NAMES[o]}" + (f" (lr {r['lr']:g})" if o != "lm" else "")
        axes[0].plot(np.maximum(h["updates"], 1), rm, color=P.color(i), label=lab)
        axes[1].plot(np.array(h["wall"]) + 1e-3, rm, color=P.color(i), label=lab)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_xlabel("parameter updates")
    axes[1].set_xlabel("wall-clock seconds")
    axes[0].set_ylabel("validation RMSE")
    axes[0].legend(loc="lower left", fontsize=7.5)
    f.suptitle("E3 · Optimisers on f4, 2×16 tanh (best learning rate of 3 for each)", x=0.01, ha="left")
    P.save(f, fig(EXP, "optimizers"), "Validation RMSE in original units (noise-free labels). LM: full batch, one update = one Gauss-Newton step.")

    body = ("Geometric-mean test RMSE (rows: batch size, columns: learning rate).\n\n"
            + md_table(["batch"] + [f"{v:g}" for v in LRS],
                       [[str(b) if b else "full"] + [fmt(v) for v in M[i]] for i, b in enumerate(BATCHES)])
            + "\n\nOptimisers (2×16 tanh, best lr of three):\n\n"
            + md_table(["optimiser", "lr", "test RMSE", "updates", "seconds", "stop"],
                       [[OPT_NAMES[o], "-" if o == "lm" else f"{best[o]['lr']:g}", fmt(best[o]["rmse"]),
                         best[o]["updates"], f"{best[o]['seconds']:.1f}", best[o]["stop"]]
                        for o in ["sgd", "momentum", "nesterov", "adam", "lm"]]))
    write_summary(EXP, "E3 · Learning rate, batch size and optimiser", body, profile)


if __name__ == "__main__":
    main()
