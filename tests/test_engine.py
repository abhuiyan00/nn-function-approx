"""Correctness tests of the engine."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fapprox import bspline
from fapprox.baselines import ChebyshevLS, PolyInterpolant, SplineLS
from fapprox.data import make_dataset
from fapprox.functions import TARGETS, get_target
from fapprox.gradcheck import gradient_error, jacobian_error
from fapprox.model import build_kan, build_mlp, kan_param_count, mlp_param_count
from fapprox.train import History, TrainConfig, fit, mse

RNG = np.random.default_rng(123)
X1 = RNG.uniform(-1, 1, (40, 1))
X2 = RNG.uniform(-1, 1, (40, 2))


def _target(x):
    return np.sin(3 * x[:, :1]) + (x[:, 1:2] ** 2 if x.shape[1] > 1 else 0)


MLP_CASES = [
    dict(hidden=[8], activation="tanh"),
    dict(hidden=[6, 5], activation="tanh"),
    dict(hidden=[6, 5], activation="sigmoid"),
    dict(hidden=[7, 7], activation="silu"),
    dict(hidden=[7, 7], activation="relu", bias_init="kink"),
    dict(hidden=[8, 8], activation="sine", w0=5.0),
    dict(hidden=[8], activation="tanh", fourier=(4, 1.0)),
]


@pytest.mark.parametrize("case", MLP_CASES, ids=lambda c: f"{c['activation']}-{c['hidden']}")
@pytest.mark.parametrize("x", [X1, X2], ids=["1d", "2d"])
def test_mlp_gradient(case, x):
    net = build_mlp(x.shape[1], seed=3, **case)
    assert gradient_error(net, x, _target(x)) < 1e-6


@pytest.mark.parametrize("x", [X1, X2], ids=["1d", "2d"])
def test_kan_gradient(x):
    net = build_kan([x.shape[1], 4, 1], grid_size=5, seed=2, noise_scale=0.5)
    assert gradient_error(net, x, _target(x)) < 1e-6


@pytest.mark.parametrize("builder", [
    lambda d: build_mlp(d, [6, 5], "tanh", seed=1),
    lambda d: build_mlp(d, [6], "relu", seed=1, bias_init="kink"),
    lambda d: build_mlp(d, [6], "tanh", seed=1, fourier=(3, 1.0)),
    lambda d: build_kan([d, 3, 1], grid_size=4, seed=1, noise_scale=0.5),
], ids=["mlp-tanh", "mlp-relu", "mlp-fourier", "kan"])
def test_jacobian(builder):
    net = builder(2)
    assert jacobian_error(net, X2) < 1e-6


def test_param_counts():
    assert build_mlp(1, [32]).n_params == mlp_param_count(1, [32]) == 97
    assert build_mlp(1, [8, 8]).n_params == mlp_param_count(1, [8, 8]) == 97
    assert build_kan([2, 5, 1], 5).n_params == kan_param_count([2, 5, 1], 5)


def test_bspline_partition_of_unity():
    knots = bspline.make_knots(np.array([-1.0]), np.array([1.0]), 7, 3)
    x = np.linspace(-1, 0.999, 200).reshape(-1, 1)
    b = bspline.basis(x, knots, 3)
    assert b.shape == (200, 1, 10)
    np.testing.assert_allclose(b.sum(axis=2), 1.0, atol=1e-12)


def test_bspline_derivative():
    knots = bspline.make_knots(np.array([-1.0]), np.array([1.0]), 6, 3)
    x = np.linspace(-0.95, 0.95, 50).reshape(-1, 1)
    _, db = bspline.basis_and_derivative(x, knots, 3)
    h = 1e-6
    num = (bspline.basis(x + h, knots, 3) - bspline.basis(x - h, knots, 3)) / (2 * h)
    np.testing.assert_allclose(db, num, atol=1e-6)


def test_kan_refit_preserves_function():
    net = build_kan([1, 3, 1], grid_size=5, seed=0, noise_scale=1.0)
    x = np.linspace(-1, 1, 400).reshape(-1, 1)
    before = net.forward(x)
    net.kan_refit(x, new_grid=20)
    after = net.forward(x)
    assert net.layers[0].G == 20
    assert np.max(np.abs(after - before)) < 5e-3


def test_lm_beats_adam_on_small_problem():
    ds = make_dataset(get_target("f4"), n_train=300, n_val=100, data_seed=0)
    a = build_mlp(1, [10], "tanh", seed=1)
    b = a.copy()
    fit(a, ds.u_train, ds.z_train, ds.u_val, ds.z_val, TrainConfig(optimizer="adam", epochs=50, patience=0))
    h = fit(b, ds.u_train, ds.z_train, ds.u_val, ds.z_val, TrainConfig(optimizer="lm", epochs=50, patience=0))
    assert np.all(np.diff(h.train_mse) <= 1e-15)  # LM accepts only descent steps
    assert mse(b, ds.u_val, ds.z_val) < mse(a, ds.u_val, ds.z_val)


def test_baselines_exact_on_polynomials():
    u = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = 3 * u ** 3 - u
    assert np.max(np.abs(ChebyshevLS(3).fit(u, y).predict(u) - y)) < 1e-10
    assert np.max(np.abs(SplineLS(8).fit(u, y).predict(u) - y)) < 1e-8
    p = PolyInterpolant(3, "chebyshev").fit_function(lambda v: 3 * v ** 3 - v)
    assert np.max(np.abs(p.predict(u) - y)) < 1e-10


def test_targets_shapes():
    for t in TARGETS.values():
        ds = make_dataset(t, n_train=50, n_val=10, n_test=400)
        assert ds.y_train.shape == (50, 1) and ds.y_test.shape[1] == 1
        assert np.all(np.abs(ds.u_test) <= 1 + 1e-12)


@pytest.mark.parametrize("optimizer", ["adam", "lm"])
def test_fit_fills_a_caller_owned_history_live(optimizer):
    ds = make_dataset(get_target("f4"), n_train=100, n_val=30, data_seed=0)
    net = build_mlp(1, [6], "tanh", seed=1)
    live, seen = History(), []
    h = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
            TrainConfig(optimizer=optimizer, epochs=5, patience=0),
            callback=lambda ep, _n: seen.append(len(live.epoch)), history=live)
    assert h is live
    assert seen == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("optimizer", ["adam", "lm"])
def test_truthy_callback_stops_training_and_restores_best_weights(optimizer):
    ds = make_dataset(get_target("f4"), n_train=100, n_val=30, data_seed=0)
    net = build_mlp(1, [6], "tanh", seed=1)
    h = fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
            TrainConfig(optimizer=optimizer, epochs=50, patience=0),
            callback=lambda ep, _n: ep >= 3)
    assert len(h.epoch) == 3
    assert h.stop_reason == "stopped by callback"
    assert mse(net, ds.u_val, ds.z_val) == pytest.approx(h.best_val)


def test_importing_the_engine_pins_blas_to_one_thread():
    # measured on the lab's LM preset: spinning BLAS threads make small solves about 2x slower
    code = "import os, fapprox; print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])"
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.split() == ["1", "1"]
