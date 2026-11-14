"""Results from the literature and the lectures that the engine must reproduce.

Each test states one claim a reader of the report would expect, and checks it on a small
problem that runs in seconds. If one fails, either the engine is wrong or the claim does
not hold in this setting, and the report has to say which.
"""
import numpy as np
import pytest

from funcprox.baselines import ChebyshevLS, PolyInterpolant, RBFNet
from funcprox.custom import make_target
from funcprox.data import grid, make_dataset
from funcprox.functions import get_target
from funcprox.metrics import evaluate, predict_original
from funcprox.model import build_mlp
from funcprox.train import TrainConfig, fit

LM = dict(optimizer="lm", epochs=150, patience=30, seed=1)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.ravel(a) - np.ravel(b)) ** 2)))


def train(target, hidden=(16,), seed=1, n_train=300, n_val=100, sigma=0.0, **cfg):
    ds = make_dataset(target, n_train=n_train, n_val=n_val, sigma=sigma, data_seed=0, n_test=2000)
    net = build_mlp(target.dim, list(hidden), "tanh", seed=seed)
    fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val, TrainConfig(**{**LM, **cfg}))
    return ds, net


# ------------------------------------------------------------- classical theory
def test_runge_equispaced_interpolation_gets_worse_with_degree_and_chebyshev_nodes_cure_it():
    """Runge (1901): on 1/(1+25x^2), equispaced nodes diverge as the degree grows; Chebyshev nodes converge."""
    f = get_target("f4")
    u = np.linspace(-1, 1, 2001).reshape(-1, 1)
    error = {(nodes, n): rmse(PolyInterpolant(n, nodes).fit_function(f).predict(u), f(u))
             for nodes in ("equispaced", "chebyshev") for n in (10, 20)}
    assert error["equispaced", 20] > 10 * error["equispaced", 10]
    assert error["chebyshev", 20] < error["chebyshev", 10] / 5


def test_chebyshev_least_squares_converges_fast_on_a_smooth_function():
    """Trefethen, ATAP ch. 8: for an analytic function the error falls faster than any power of the degree."""
    f = get_target("f1")
    u = np.linspace(-1, 1, 400).reshape(-1, 1)
    errors = [rmse(ChebyshevLS(n).fit(u, f(u)).predict(u), f(u)) for n in (8, 16, 24)]
    assert errors[0] > 100 * errors[1] > 100 * 100 * errors[2]


@pytest.mark.parametrize("key", ["f4", "f6"])
def test_an_rbf_network_gets_closer_with_more_centres(key):
    """Park & Sandberg (1991): Gaussian RBF networks are universal approximators.

    On sin(2 pi x) the error stops near 1e-4 from about 12 centres on: the Gaussians grow so
    alike that least squares hits its conditioning floor. The peak and the kink keep improving.
    """
    f = get_target(key)
    u = np.random.default_rng(0).uniform(-1, 1, (400, 1))
    test = np.linspace(-1, 1, 500).reshape(-1, 1)
    errors = [rmse(RBFNet(k).fit(u, f(u)).predict(test), f(test)) for k in (4, 12, 30)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01


# -------------------------------------------------------- the network as a device
def test_more_hidden_units_approximate_better():
    """Cybenko (1989), Hornik (1991): one hidden layer is enough, given enough units."""
    f = get_target("f1")
    (ds2, narrow), (ds16, wide) = train(f, hidden=(2,)), train(f, hidden=(16,))
    assert evaluate(wide, ds16)["rmse"] < evaluate(narrow, ds2)["rmse"] / 10


def test_the_network_learns_the_function_and_not_the_noise():
    """With noisy labels the fit averages the noise out: the error against the exact f falls below sigma."""
    sigma = 0.1
    ds, net = train(get_target("f1"), n_train=2000, n_val=500, sigma=sigma)
    assert evaluate(net, ds)["rmse"] < sigma / 2


def test_a_network_does_not_extrapolate():
    """Haley & Soloway (1992): outside the training box a tanh network flattens out, whatever f does."""
    f = get_target("f1")
    ds, net = train(f)
    inside, outside = np.linspace(-1, 1, 200).reshape(-1, 1), np.linspace(1.25, 2, 200).reshape(-1, 1)
    assert rmse(predict_original(net, ds, outside), f(outside)) > 20 * rmse(predict_original(net, ds, inside), f(inside))


def test_a_classical_fit_beats_the_network_on_a_smooth_one_dimensional_function():
    """The project's own finding (E6): for smooth 1-D targets a polynomial of the same size is closer."""
    f = get_target("f1")
    ds, net = train(f, hidden=(16,))
    poly = ChebyshevLS(net.n_params // 2).fit(ds.u_train, ds.y_train)
    assert rmse(poly.predict(ds.xs.transform(ds.x_test)), ds.y_test) < evaluate(net, ds)["rmse"] / 10


# ---------------------------------------------------------------- the missing middle
@pytest.mark.parametrize("key", ["m1", "m2", "m3", "m4"])
def test_a_hole_always_costs_accuracy(key):
    """No data, no information (E7): inside the missing middle the error is larger than where there was data."""
    ds, net = train(get_target(key))
    m = evaluate(net, ds)
    assert m["rmse_gap"] > 2 * m["rmse_seen"]


def test_a_simple_shape_is_bridged_but_a_hidden_kink_comes_back_rounded():
    """A tanh network is smooth: it continues x^3 - x across a hole, but cannot put back the corner of |x|.

    The hidden peak of Runge's function (m2) is not in this test: with its own 20 % hole the shoulders
    pin the peak down to about 3 % of its height. E7 shows that a 30 % hole is wide enough to lose it.
    """
    cubic, cubic_net = train(get_target("m4"))
    kink, kink_net = train(get_target("m3"))
    assert evaluate(cubic_net, cubic)["rmse_gap"] < 0.03 * np.ptp(cubic.y_test)
    rounded = evaluate(kink_net, kink)
    assert rounded["max_err_gap"] > 0.05 > 50 * rounded["rmse_seen"]


def test_an_erased_hump_of_a_drawn_curve_is_never_seen_and_is_scored_on_its_own():
    """A student draws one period of a sine on the pad (120 columns across -1 to 1) and erases the second hump.

    The erased points must not reach training or validation, all of them must be scored, and the
    network, which never saw the hump, misses it by more than it misses the rest.
    """
    x = np.linspace(-1, 1, 120)
    target = make_target(points=list(zip(x, 0.8 * np.sin(np.pi * x))), gap=[(0.2, 0.8)])
    ds, net = train(target)
    erased = (x >= 0.2) & (x <= 0.8)
    assert not np.any((ds.x_train >= 0.2) & (ds.x_train <= 0.8))
    assert not np.any((ds.x_val >= 0.2) & (ds.x_val <= 0.8))
    assert np.sum((ds.x_test >= 0.2) & (ds.x_test <= 0.8)) == erased.sum()
    m = evaluate(net, ds)
    assert m["rmse_gap"] > m["rmse_seen"]


# ------------------------------------------------------------------ reproducibility
def test_the_same_seeds_give_the_same_network_bit_for_bit():
    a, b = (train(get_target("f4"), epochs=20)[1].get_flat() for _ in range(2))
    np.testing.assert_array_equal(a, b)


def test_the_model_seed_changes_only_the_network_and_never_the_data():
    """Data and model seeds are independent, so a seed study compares networks on the same points."""
    (ds1, net1), (ds2, net2) = train(get_target("f4"), seed=1, epochs=5), train(get_target("f4"), seed=2, epochs=5)
    np.testing.assert_array_equal(ds1.x_train, ds2.x_train)
    assert not np.array_equal(net1.get_flat(), net2.get_flat())


@pytest.mark.parametrize("key", ["f1", "f5"])
def test_the_test_grid_covers_the_whole_box(key):
    """Scores are taken on a dense grid of the whole range, so a model cannot hide a bad stretch."""
    target = get_target(key)
    g = grid(target.domain, 2000)
    for axis, (lo, hi) in enumerate(target.domain):
        assert g[:, axis].min() == lo and g[:, axis].max() == hi
