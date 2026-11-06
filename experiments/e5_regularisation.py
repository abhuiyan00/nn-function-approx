"""E5 - Early stopping and L2 weight decay against overfitting.

Question: with few noisy points and a large network, which regulariser protects the
approximation of the *exact* function better, and how sensitive is it to its knob?

Setting: f1, N = 250 (validation 62), sigma = 0.1, 2 x 128 tanh (16.9k params),
budget 4,800 epochs (= 19.2k updates).
    early stopping: patience {5, 15, 50, 200, 1000} epochs, best weights restored
    L2 weight decay: lambda {0, 1e-5, 1e-4, 1e-3, 1e-2}, no early stopping
"""
from common import fig, fmt, md_table, parse_profile, run_many, run_one, save_rows, write_summary

import numpy as np

from fapprox import plotting as P

EXP = "e5_regularisation"
PATIENCE = [5, 15, 50, 200, 1000]
L2 = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]


def main():
    profile, prof = parse_profile(__doc__)
    base = dict(target="f1", n_train=250, n_val=62, sigma=0.1, hidden=(128, 128), epochs=4800 * prof["scale"])
    es = [dict(kind="patience", patience=p, restore_best=True, seed=s, **base) for p in PATIENCE for s in prof["seeds"]]
    wd = [dict(kind="l2", l2=l, patience=0, restore_best=False, seed=s, **base) for l in L2 for s in prof["seeds"]]
    rows = run_many(es + wd, prof["workers"], label=EXP)
    save_rows(EXP, rows)
    r_es, r_wd = rows[: len(es)], rows[len(es):]

    f, axes = P.plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    xs = np.arange(len(PATIENCE))
    for j, p in enumerate(PATIENCE):
        v = [r["rmse"] for r in r_es if r["patience"] == p]
        ax.plot([j] * len(v), v, ls="none", **P.marker_kw(0))
        ax.text(j + 0.12, np.mean(v), f"ep {int(np.mean([r['best_epoch'] for r in r_es if r['patience'] == p]))}",
                fontsize=7.5, color=P.TEXT_2, va="center")
    m_es = [np.mean([r["rmse"] for r in r_es if r["patience"] == p]) for p in PATIENCE]
    ax.plot(xs, m_es, color=P.color(0), lw=1.2)
    none = [r["rmse"] for r in r_wd if r["l2"] == 0.0]
    ax.axhline(np.mean(none), color=P.MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(len(PATIENCE) - 0.6, np.mean(none) * 1.04, "no regularisation", ha="right", va="bottom", fontsize=8, color=P.TEXT_2)
    ax.set_xticks(xs, [str(p) for p in PATIENCE])
    ax.set_xlabel("early-stopping patience (epochs); text = best epoch")
    ax.set_ylabel("test RMSE vs exact f")
    ax.set_title("Early stopping (best weights restored)")

    ax = axes[1]
    xs = np.arange(len(L2))
    for j, l in enumerate(L2):
        v = [r["rmse"] for r in r_wd if r["l2"] == l]
        ax.plot([j] * len(v), v, ls="none", **P.marker_kw(1))
    ax.plot(xs, [np.mean([r["rmse"] for r in r_wd if r["l2"] == l]) for l in L2], color=P.color(1), lw=1.2)
    ax.set_xticks(xs, ["0"] + [f"{l:g}" for l in L2[1:]])
    ax.set_xlabel("L2 weight decay λ (no early stopping)")
    ax.set_title("Weight decay")
    lo = min(axes[0].get_ylim()[0], ax.get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], ax.get_ylim()[1])
    for a in axes[:2]:
        a.set_ylim(lo, hi)

    best_es = min(PATIENCE, key=lambda p: np.mean([r["rmse"] for r in r_es if r["patience"] == p]))
    best_l2 = min(L2[1:], key=lambda l: np.mean([r["rmse"] for r in r_wd if r["l2"] == l]))
    ax = axes[2]
    seed = prof["seeds"][0]
    for i, (lab, kw) in enumerate([("no regularisation", dict(l2=0.0, patience=0, restore_best=False)),
                                   (f"early stop, patience {best_es}", dict(patience=best_es, restore_best=True)),
                                   (f"weight decay λ={best_l2:g}", dict(l2=best_l2, patience=0, restore_best=False))]):
        r = run_one(dict(seed=seed, return_pred=True, **{**base, **kw}))
        ax.plot(r["_x"], r["_pred"] - r["_true"], color=[P.MUTED, P.color(0), P.color(1)][i], lw=1.4,
                label=f"{lab} · RMSE {fmt(r['rmse'])}")
    ax.axhline(0, color=P.TEXT, lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("network − exact f")
    ax.set_title("Error curves (one seed)")
    ax.legend(loc="lower center", fontsize=7.5)
    f.suptitle("E5 · Regularisation: N = 250 noisy points (σ = 0.1), 2×128 tanh", x=0.01, ha="left")
    P.save(f, fig(EXP, "regularisation"), f"Adam lr 1e-3, batch 64, ≤{base['epochs']} epochs; profile {profile}.")

    body = (md_table(["patience", "mean test RMSE", "mean best epoch"],
                     [[p, fmt(np.mean([r["rmse"] for r in r_es if r["patience"] == p])),
                       int(np.mean([r["best_epoch"] for r in r_es if r["patience"] == p]))] for p in PATIENCE])
            + "\n\n" + md_table(["λ (no early stop)", "mean test RMSE"],
                                [[f"{l:g}", fmt(np.mean([r["rmse"] for r in r_wd if r["l2"] == l]))] for l in L2]))
    write_summary(EXP, "E5 · Early stopping and weight decay", body, profile)


if __name__ == "__main__":
    main()
