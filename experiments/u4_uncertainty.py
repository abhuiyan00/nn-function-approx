"""U4 (unique) - Knowing where the approximation can be trusted.

(a) Deep ensembles (Lakshminarayanan et al., 2017): 5 networks, different seeds,
    trained on f2 = 2x^2 inside [-2, 2], queried on [-3, 3]. Does the spread of the
    ensemble grow where the error grows (extrapolation)?
(b) Split conformal prediction (Vovk et al.; Lei et al., 2018): distribution-free
    intervals with guaranteed marginal coverage, here on f1 with heteroscedastic
    noise sigma(x) = 0.02 + 0.2 ((x + 1) / 2)^2.
      plain     : interval f_hat(x) +- q,          q from |residuals| on a calibration set
      weighted  : interval f_hat(x) +- q * g(x),   g = second network fitted to |residuals|
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_json, write_summary

import numpy as np

from funcprox import plotting as P
from funcprox.data import InputScaler, OutputScaler, make_dataset
from funcprox.functions import get_target
from funcprox.metrics import predict_original
from funcprox.model import build_mlp
from funcprox.train import TrainConfig, fit

EXP = "u4_uncertainty"


def ens_member(spec):
    t = get_target("f2")
    ds = make_dataset(t, spec["n_train"], spec["n_val"], 0.1, 0)
    net = build_mlp(1, [32, 32], spec["act"], seed=spec["seed"], bias_init="kink" if spec["act"] == "relu" else None)
    fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
        TrainConfig(epochs=spec["epochs"], patience=spec["patience"], seed=spec["seed"]))
    x = np.linspace(-3, 3, 1201).reshape(-1, 1)
    return dict(act=spec["act"], seed=spec["seed"], x=x.ravel(), y=predict_original(net, ds, x).ravel())


def noise_sd(x):
    return 0.02 + 0.2 * ((x[:, 0] + 1) / 2) ** 2


def conformal(prof, rng_seed=0):
    t = get_target("f1")
    rng = np.random.default_rng(rng_seed)
    def draw(n):
        x = rng.uniform(-1, 1, (n, 1))
        return x, t(x) + rng.normal(size=(n, 1)) * noise_sd(x).reshape(-1, 1)
    (xtr, ytr), (xva, yva), (xsc, ysc), (xpool, ypool) = draw(2000), draw(500), draw(1000), draw(6000)
    xs, ys = InputScaler.from_domain(t.domain), OutputScaler.fit(ytr)
    net = build_mlp(1, [32, 32], "tanh", seed=1)
    fit(net, xs.transform(xtr), ys.transform(ytr), xs.transform(xva), ys.transform(yva),
        TrainConfig(epochs=prof["epochs"], patience=prof["patience"]))
    fhat = lambda x: ys.inverse(net.forward(xs.transform(x))).ravel()
    # scale model g(x) ~ E|y - f(x)|, trained on a separate split
    r_sc = np.abs(ysc.ravel() - fhat(xsc))
    gs = OutputScaler.fit(r_sc)
    gnet = build_mlp(1, [16], "tanh", seed=2)
    k = int(0.8 * len(xsc))
    fit(gnet, xs.transform(xsc[:k]), gs.transform(r_sc[:k].reshape(-1, 1)), xs.transform(xsc[k:]),
        gs.transform(r_sc[k:].reshape(-1, 1)), TrainConfig(optimizer="lm", epochs=100, patience=10))
    g = lambda x: np.maximum(gs.inverse(gnet.forward(xs.transform(x))).ravel(), 1e-3)

    res_pool = np.abs(ypool.ravel() - fhat(xpool))
    g_pool = g(xpool)
    alphas = [0.5, 0.3, 0.2, 0.1, 0.05]
    cov = {m: {a: [] for a in alphas} for m in ("plain", "weighted")}
    width = {m: {a: [] for a in alphas} for m in ("plain", "weighted")}
    bins = np.linspace(-1, 1, 11)
    cond = {m: [] for m in ("plain", "weighted")}
    for rep in range(200):
        perm = rng.permutation(len(xpool))
        cal, te = perm[:1000], perm[1000:]
        for a in alphas:
            lvl = min(np.ceil((len(cal) + 1) * (1 - a)) / len(cal), 1.0)
            for m, s_cal, s_te, scale in (("plain", res_pool[cal], res_pool[te], 1.0),
                                          ("weighted", res_pool[cal] / g_pool[cal], res_pool[te] / g_pool[te], g_pool[te])):
                q = np.quantile(s_cal, lvl, method="higher")
                inside = s_te <= q
                cov[m][a].append(inside.mean())
                width[m][a].append(np.mean(2 * q * scale))
                if a == 0.1:
                    which = np.digitize(xpool[te, 0], bins) - 1
                    cond[m].append([inside[which == b].mean() for b in range(10)])
    # one illustrative band at alpha = 0.1
    xg = np.linspace(-1, 1, 400).reshape(-1, 1)
    cal = rng.permutation(len(xpool))[:1000]
    lvl = np.ceil(1001 * 0.9) / 1000
    q_p = np.quantile(res_pool[cal], lvl, method="higher")
    q_w = np.quantile(res_pool[cal] / g_pool[cal], lvl, method="higher")
    return dict(alphas=alphas, cov=cov, width=width, cond={m: np.array(v) for m, v in cond.items()}, bins=bins,
                xg=xg.ravel(), f=t(xg).ravel(), fh=fhat(xg), band_p=q_p, band_w=q_w * g(xg),
                sd=noise_sd(xg), xs=xpool[:400, 0], ys=ypool[:400, 0])


def main():
    profile, prof = parse_profile(__doc__)
    M = 5 if profile == "quick" else 10
    specs = [dict(act=a, seed=s, n_train=prof["n_train"], n_val=prof["n_val"], epochs=prof["epochs"],
                  patience=prof["patience"]) for a in ("tanh", "relu") for s in range(1, M + 1)]
    members = run_many(specs, prof["workers"], fn=ens_member, label=EXP + " ensembles")
    cp = conformal(prof)

    f, axes = P.plt.subplots(2, 2, figsize=(12.5, 7.6))
    t = get_target("f2")
    ens_stats = {}
    for ax, act, c in ((axes[0, 0], "tanh", 0), (axes[0, 1], "relu", 4)):
        Y = np.array([m["y"] for m in members if m["act"] == act])
        x = members[0]["x"]
        mu, sd = Y.mean(0), Y.std(0)
        ex = t(x.reshape(-1, 1)).ravel()
        inside = np.abs(x) <= 2
        err = np.abs(mu - ex)
        ens_stats[act] = dict(rmse_in=float(np.sqrt(np.mean(err[inside] ** 2))),
                              rmse_out=float(np.sqrt(np.mean(err[~inside] ** 2))),
                              sd_in=float(sd[inside].mean()), sd_out=float(sd[~inside].mean()),
                              covered_out=float(np.mean(err[~inside] <= 2 * sd[~inside])))
        ax.axvspan(-2, 2, color=P.BAND, lw=0, label="training box")
        ax.fill_between(x, mu - 2 * sd, mu + 2 * sd, color=P.color(c), alpha=0.25, lw=0, label="ensemble mean ± 2 sd")
        for y in Y:
            ax.plot(x, y, color=P.color(c), lw=0.7, alpha=0.6)
        ax.plot(x, mu, color=P.color(c), label=f"ensemble mean ({M} × 2×32 {act})")
        ax.plot(x, ex, **P.truth_kw())
        ax.set_ylim(-2, 20)
        ax.set_title(f"{act}: in-box RMSE {fmt(ens_stats[act]['rmse_in'])}, outside {fmt(ens_stats[act]['rmse_out'])}")
        ax.set_xlabel("x")
        ax.legend(loc="upper center", fontsize=7.5)
    axes[0, 0].set_ylabel("f2 = 2x²")

    ax = axes[1, 0]
    ax.plot(cp["xs"], cp["ys"], ls="none", marker="o", markersize=2.5, color=P.MUTED, alpha=0.6, label="noisy samples")
    ax.fill_between(cp["xg"], cp["fh"] - cp["band_p"], cp["fh"] + cp["band_p"], color=P.color(1), alpha=0.18, lw=0,
                    label="plain split conformal (90 %)")
    ax.plot(cp["xg"], cp["fh"] - cp["band_w"], color=P.color(2), lw=1.4, label="weighted conformal (90 %)")
    ax.plot(cp["xg"], cp["fh"] + cp["band_w"], color=P.color(2), lw=1.4)
    ax.plot(cp["xg"], cp["f"], **P.truth_kw())
    ax.set_xlabel("x")
    ax.set_title("90 % prediction intervals, noise sd 0.02 → 0.22 along x")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1, 1]
    centres = 0.5 * (cp["bins"][1:] + cp["bins"][:-1])
    for i, m in enumerate(("plain", "weighted")):
        c = cp["cond"][m]
        ax.fill_between(centres, np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0), color=P.color(1 + i),
                        alpha=0.18, lw=0)
        ax.plot(centres, c.mean(0), label=f"{m}: overall {np.mean(cp['cov'][m][0.1]):.3f}", **P.marker_kw(1 + i))
    ax.axhline(0.9, color=P.TEXT, lw=1, ls=(0, (3, 2)))
    ax.text(-0.98, 0.905, "nominal 0.90", fontsize=8, color=P.TEXT_2, va="bottom")
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("x (10 bins)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage per region (200 random calibration splits, band 5-95 %)")
    ax.legend(loc="lower left", fontsize=7.5)
    f.suptitle("U4 · Trust: ensemble spread outside the data, conformal intervals inside", x=0.01, ha="left")
    P.save(f, fig(EXP, "uncertainty"), "Top: f2 trained on [-2, 2] with σ = 0.1. Bottom: f1, heteroscedastic noise; "
           "1,000 calibration / 5,000 test points per split.")

    save_json(EXP, dict(ensembles=ens_stats,
                        coverage={m: {str(a): float(np.mean(v)) for a, v in cp["cov"][m].items()} for m in cp["cov"]},
                        width={m: {str(a): float(np.mean(v)) for a, v in cp["width"][m].items()} for m in cp["width"]}),
              "summary")
    body = ("Deep ensembles on f2 (trained on [-2, 2], evaluated on [-3, 3]):\n\n"
            + md_table(["members", "RMSE inside", "RMSE outside", "mean sd inside", "mean sd outside",
                        "outside: abs error ≤ 2 sd"],
                       [[f"{M} × {a}", fmt(s["rmse_in"]), fmt(s["rmse_out"]), fmt(s["sd_in"]), fmt(s["sd_out"]),
                         f"{100 * s['covered_out']:.0f} %"] for a, s in ens_stats.items()])
            + "\n\nSplit conformal on f1 with heteroscedastic noise (mean over 200 splits):\n\n"
            + md_table(["nominal", "plain coverage", "plain width", "weighted coverage", "weighted width"],
                       [[f"{1 - a:.2f}", f"{np.mean(cp['cov']['plain'][a]):.3f}", fmt(np.mean(cp['width']['plain'][a])),
                         f"{np.mean(cp['cov']['weighted'][a]):.3f}", fmt(np.mean(cp['width']['weighted'][a]))]
                        for a in cp["alphas"]])
            + "\n\nCoverage at nominal 0.90 in the lowest / highest-noise bin: plain "
              f"{cp['cond']['plain'].mean(0)[0]:.3f} / {cp['cond']['plain'].mean(0)[-1]:.3f}, weighted "
              f"{cp['cond']['weighted'].mean(0)[0]:.3f} / {cp['cond']['weighted'].mean(0)[-1]:.3f}.")
    write_summary(EXP, "U4 · Uncertainty: ensembles and conformal prediction", body, profile)


if __name__ == "__main__":
    main()
