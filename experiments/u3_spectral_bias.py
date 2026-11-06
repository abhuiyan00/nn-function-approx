"""U3 (unique) - Spectral bias: networks learn low frequencies first. Can we fix it?

Target f3(x) = sin(pi x) + 0.5 sin(4 pi x) + 0.25 sin(16 pi x): three known
frequencies, so after every training step we can measure how much of *each*
component the network has already learned (projection on sin(k pi x), k = 1, 4, 16).

Models (full-batch Adam, lr 1e-3, noise-free N = 1,024):
    tanh MLP 3x64 | ReLU MLP 3x64 (kink init) | Fourier features (m = 32, scale 4) + tanh 2x64
    SIREN 3x64 (w0 = 30) | KAN [1, 16, 16, 1], G = 10
"""
from common import fig, fmt, md_table, parse_profile, run_many, save_json, write_summary

import numpy as np

from fapprox import plotting as P
from fapprox.data import make_dataset
from fapprox.functions import get_target
from fapprox.metrics import evaluate, predict_original
from fapprox.model import build_kan, build_mlp
from fapprox.train import TrainConfig, fit

EXP = "u3_spectral_bias"
MODELS = {
    "tanh MLP": lambda s: build_mlp(1, [64, 64, 64], "tanh", seed=s),
    "ReLU MLP": lambda s: build_mlp(1, [64, 64, 64], "relu", seed=s, bias_init="kink"),
    "Fourier features + tanh": lambda s: build_mlp(1, [64, 64], "tanh", seed=s, fourier=(32, 4.0)),
    "SIREN": lambda s: build_mlp(1, [64, 64, 64], "sine", seed=s, w0=30.0),
    "KAN": lambda s: build_kan([1, 16, 16, 1], 10, seed=s),
}
KS = (1, 4, 16)
AMPS = (1.0, 0.5, 0.25)


def run_model(spec: dict) -> dict:
    t = get_target("f3")
    ds = make_dataset(t, 1024, 256, 0.0, 0)
    net = MODELS[spec["model"]](spec["seed"])
    xg = np.linspace(-1, 1, 2048, endpoint=False).reshape(-1, 1)  # periodic grid for projections / FFT
    basis = np.stack([np.sin(k * np.pi * xg[:, 0]) for k in KS], axis=1)
    fx = t(xg).ravel()
    snaps = dict(step=[], rel=[], spec=[])

    def cb(step, n):
        if step % spec["every"]:
            return
        p = predict_original(n, ds, xg).ravel()
        a = basis.T @ p / (len(xg) / 2)          # learned amplitude of sin(k pi x)
        snaps["step"].append(step)
        snaps["rel"].append([abs(a[j] - AMPS[j]) / AMPS[j] for j in range(len(KS))])
        snaps["spec"].append(np.abs(np.fft.rfft(fx - p))[:41] / (len(xg) / 2))

    cfg = TrainConfig(optimizer="adam", lr=1e-3, batch=0, epochs=spec["steps"], patience=0,
                      restore_best=False, seed=spec["seed"])
    fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, cfg, callback=cb)
    m = evaluate(net, ds)
    return dict(model=spec["model"], seed=spec["seed"], n_params=net.n_params, rmse=m["rmse"],
                step=snaps["step"], rel=np.array(snaps["rel"]), spec=np.array(snaps["spec"]),
                pred=predict_original(net, ds, xg).ravel(), x=xg.ravel(), f=fx)


def first_step_below(r, j, thr=0.1):
    idx = np.where(r["rel"][:, j] < thr)[0]
    return int(r["step"][idx[0]]) if len(idx) else None


def main():
    profile, prof = parse_profile(__doc__)
    steps = 3000 if profile == "quick" else 10000
    specs = [dict(model=m, seed=prof["seeds"][0], steps=steps, every=25) for m in MODELS]
    rows = run_many(specs, prof["workers"], fn=run_model, label=EXP)

    f, axes = P.plt.subplots(1, len(MODELS), figsize=(16, 3.6), sharey=True)
    for ax, r in zip(axes, rows):
        for j, k in enumerate(KS):
            ax.plot(np.maximum(r["step"], 1), np.maximum(r["rel"][:, j], 1e-6), color=P.color(j),
                    label=f"k = {k}  (amplitude {AMPS[j]:g})")
        ax.axhline(0.1, color=P.MUTED, lw=1, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{r['model']} · {r['n_params']} p")
        ax.set_xlabel("full-batch Adam step")
        P.note(ax, f"final RMSE {fmt(r['rmse'])}", "lower left")
    axes[0].set_ylabel("relative error of component sin(kπx)")
    h_, l_ = axes[0].get_legend_handles_labels()
    f.legend(h_, l_, loc="outside upper right", ncol=3)
    f.suptitle("U3 · Spectral bias: when does each frequency of f3 get learned?", x=0.01, ha="left")
    P.save(f, fig(EXP, "components"), "Dashed line: 10 % error. Plain tanh/ReLU learn k = 1 first and k = 16 last "
           "(or never); input encodings and KAN splines reach high frequencies early.")

    f, axes = P.plt.subplots(1, 3, figsize=(15, 3.7))
    for ax, name in zip(axes[:2], ["tanh MLP", "Fourier features + tanh"]):
        r = next(r for r in rows if r["model"] == name)
        S = np.log10(np.maximum(r["spec"].T, 1e-5))
        im = ax.pcolormesh(np.maximum(r["step"], 1), np.arange(S.shape[0]), S, cmap=P.SEQ, shading="auto",
                           vmin=-5, vmax=0)
        ax.set_xscale("log")
        ax.set_ylabel("frequency bin (cycles on [-1, 1))")
        ax.set_xlabel("full-batch Adam step")
        ax.set_title(f"residual spectrum · {name}")
        ax.grid(False)
        for k in KS:
            ax.axhline(k, color="#ffffff", lw=0.6, alpha=0.7)
    f.colorbar(im, ax=axes[:2], label="log10 |residual amplitude|")
    ax = axes[2]
    sel = (rows[0]["x"] > 0.2) & (rows[0]["x"] < 0.6)
    ax.plot(rows[0]["x"][sel], rows[0]["f"][sel], **P.truth_kw())
    for i, r in enumerate(rows):
        ax.plot(r["x"][sel], r["pred"][sel], color=P.color(i + 3), lw=1.4, label=r["model"])
    ax.set_title("final fits, zoom x ∈ [0.2, 0.6]")
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=7)
    f.suptitle("U3 · Residual spectrum during training (dark rows = frequencies not yet learned)", x=0.01, ha="left")
    P.save(f, fig(EXP, "spectrum"), "sin(kπx) completes k cycles on [-1, 1), so it sits in bin k (1, 4, 16; "
           "thin white lines). SIREN (w0 = 30) wiggles between training points.")

    tbl = [[r["model"], r["n_params"], fmt(r["rmse"])] + [str(first_step_below(r, j) or "never") for j in range(3)]
           for r in rows]
    save_json(EXP, {r["model"]: dict(rmse=r["rmse"], step=r["step"], rel=r["rel"].tolist()) for r in rows}, "curves")
    body = (f"Full-batch Adam, {steps} steps, lr 1e-3, one seed.\n\n"
            + md_table(["model", "params", "final RMSE", "step k=1 < 10 %", "step k=4 < 10 %", "step k=16 < 10 %"], tbl))
    write_summary(EXP, "U3 · Spectral bias", body, profile)


if __name__ == "__main__":
    main()
