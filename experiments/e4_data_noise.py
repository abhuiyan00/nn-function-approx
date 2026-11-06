"""E4 - Training-set size and label noise.

Question: how does test error (vs the exact f) scale with the number of training
points N and the noise level sigma, and what does overfitting look like when we can
see the exact function?

Grid: N {100, 250, 500, 1000, 2000, 5000} x sigma {0, 0.05, 0.1, 0.2}, f1, 2 x 32 tanh.
Validation = N/4 noisy points. Equal budget of ~19k parameter updates for every N
(so small sets are not simply under-trained); patience scaled the same way.
Each cell is run with Adam (constant and cosine learning rate) and with Levenberg-Marquardt.
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_rows, write_summary

import numpy as np

from fapprox import plotting as P
from fapprox.data import make_dataset
from fapprox.functions import get_target
from fapprox.metrics import predict_original
from fapprox.model import build_mlp
from fapprox.train import TrainConfig, fit

EXP = "e4_data_noise"
NS = [100, 250, 500, 1000, 2000, 5000]
SIGMAS = [0.0, 0.05, 0.1, 0.2]
UPDATES = 19_200  # = 600 epochs x 32 batches of 64 at N = 2000


def main():
    profile, prof = parse_profile(__doc__)
    specs = []
    for n in NS:
        steps = int(np.ceil(n / 64))
        for sg in SIGMAS:
            for s in prof["seeds"]:
                for sch in ("cosine", "constant"):
                    specs.append(dict(target="f1", n_train=n, n_val=max(n // 4, 25), sigma=sg, seed=s,
                                      lr_schedule=sch, epochs=int(np.ceil(UPDATES * prof["scale"] / steps)),
                                      patience=int(np.ceil(prof["patience"] * 32 / steps))))
                specs.append(dict(target="f1", n_train=n, n_val=max(n // 4, 25), sigma=sg, seed=s,
                                  lr_schedule="lm", optimizer="lm", hidden=(16, 16), epochs=prof["lm_iters"],
                                  patience=20))
    rows = run_many(specs, prof["workers"], label=EXP)
    save_rows(EXP, rows)

    f, axes = P.plt.subplots(1, 2, figsize=(12.5, 4.0))
    ax = axes[0]
    def gm(n, sg, sch):
        return float(np.exp(np.mean(np.log([r["rmse"] for r in rows if r["n_train"] == n and r["sigma"] == sg
                                            and r["lr_schedule"] == sch]))))

    for i, sg in enumerate(SIGMAS):
        ax.plot(NS, [gm(n, sg, "lm") for n in NS], color=P.color(i), lw=1.3, ls=(0, (3, 2)))
        m = [gm(n, sg, "constant") for n in NS]
        ax.plot(NS, m, label=f"σ = {sg:g}", **P.marker_kw(i))
        if sg > 0:
            ax.text(NS[-1] * 1.12, m[-1], f"{m[-1] / sg:.2f} σ", fontsize=7.5, color=P.TEXT_2, va="center")
    ref = np.array(NS, dtype=float)
    ax.plot(ref, 0.1 * np.sqrt(ref[0] / ref) * 0.6, color=P.MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(ref[2], 0.1 * np.sqrt(ref[0] / ref[2]) * 0.6 * 0.72, "∝ N^-1/2", color=P.MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=NS[-1] * 2.2)
    ax.set_xlabel("training points N")
    ax.set_ylabel("test RMSE vs exact f")
    ax.set_title("Error vs data size and noise (f1, 2×32 tanh)")
    P.note(ax, "solid: Adam 2×32, batch 64 · dashed: Levenberg-Marquardt 2×16", "upper right")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))

    # ---- overfitting, seen through the exact function ----------------------------
    target = get_target("f1")
    ds = make_dataset(target, 100, 25, 0.2, data_seed=0)
    net = build_mlp(1, [128, 128], "tanh", seed=prof["seeds"][0])
    ep, te = [], []

    def cb(e, n):
        if e % 10 == 0:
            ep.append(e)
            te.append(float(np.mean((predict_original(n, ds, ds.x_test) - ds.y_test) ** 2)))

    h = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
            TrainConfig(epochs=6000, patience=0, restore_best=False, seed=prof["seeds"][0]), callback=cb)
    sd = ds.ys.std ** 2
    ax = axes[1]
    ax.plot(h.epoch, np.array(h.train_mse) * sd, color=P.color(0), lw=1.5, label="train MSE (noisy labels)")
    ax.plot(h.epoch, np.array(h.val_mse) * sd, color=P.color(1), lw=1.5, label="validation MSE (noisy labels)")
    ax.plot(ep, te, color=P.color(2), label="test MSE vs exact f")
    ax.axhline(0.04, color=P.MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(h.epoch[-1], 0.04 * 1.08, "noise variance σ² = 0.04", ha="right", color=P.TEXT_2, fontsize=8)
    ax.axvline(h.best_epoch, color=P.MUTED, lw=1)
    ax.text(h.best_epoch * 0.93, 0.97, f"best validation\nepoch {h.best_epoch}", transform=ax.get_xaxis_transform(),
            ha="right", va="top", color=P.TEXT_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title("Overfitting: N = 100, σ = 0.2, 2×128 tanh, no early stop")
    ax.legend(loc="lower left", fontsize=7.5)
    f.suptitle("E4 · Data size, label noise and overfitting", x=0.01, ha="left")
    P.save(f, fig(EXP, "n_sigma"), f"Equal update budget (~{UPDATES * prof['scale'] // 1000}k) for all N; validation = N/4 points; profile {profile}.")

    i_best = int(np.argmin(np.abs(np.array(ep) - h.best_epoch)))
    tbl = [[f"{sg:g}", sch] + [fmt(gm(n, sg, sch)) for n in NS] for sg in SIGMAS for sch in ("constant", "cosine", "lm")]
    body = ("Geometric-mean test RMSE vs exact f (rows σ and trainer, columns N). Trainer: Adam 2×32 batch 64 with"
            " constant or cosine-annealed learning rate, or Levenberg-Marquardt (lm) 2×16 full batch.\n\n"
            + md_table(["σ", "trainer"] + [f"N={n}" for n in NS], tbl)
            + "\n\nLM is full batch and uses at most 4,000 training points (N = 5000 is subsampled)."
            + f"\n\nOverfitting run (N=100, σ=0.2, 2×128): best validation epoch {h.best_epoch}; test MSE there "
              f"{fmt(te[i_best])}, at the last epoch {fmt(te[-1])} (×{te[-1] / te[i_best]:.1f}); final train MSE "
              f"{fmt(h.train_mse[-1] * sd)} vs noise variance 0.04.")
    write_summary(EXP, "E4 · Data size and noise", body, profile)


if __name__ == "__main__":
    main()
