"""Functions a user brings: typed formulas, measured points, and a missing middle."""
import json
import math

import numpy as np
import pytest

from funcprox import cli
from funcprox.custom import make_target, saved_target
from funcprox.data import grid, make_dataset
from funcprox.functions import TARGETS, get_target
from funcprox.metrics import evaluate, regression_metrics
from funcprox.model import build_mlp
from funcprox.train import TrainConfig, fit


def at(text: str, *inputs: float, domain=None) -> float:
    target = make_target(formula=text, domain=domain or [(-5.0, 5.0)] * max(1, len(inputs)))
    return float(target(np.array([inputs]))[0, 0])


# --------------------------------------------------------------------- formulas
@pytest.mark.parametrize("text,x,expected", [
    ("sin(x)", 0.5, math.sin(0.5)),
    ("x^2 + 1", 2.0, 5.0),
    ("2*x**3", 2.0, 16.0),
    ("-x^2", 2.0, -4.0),
    ("exp(-x^2)", 1.0, math.exp(-1)),
    ("abs(x) + pi", -1.0, 1 + math.pi),
    ("min(x, 0.5)", 2.0, 0.5),
    ("max(x, 0.5)", 2.0, 2.0),
    ("ln(e) + log(e)", 1.0, 2.0),
    ("sqrt(x) / 2", 4.0, 1.0),
    ("step(x) * x", -1.0, 0.0),
    ("3", 1.0, 3.0),
])
def test_a_formula_means_what_it_says(text, x, expected):
    assert at(text, x) == pytest.approx(expected)


def test_a_formula_with_y_has_two_inputs():
    target = make_target(formula="x + 10*y", domain=[(0, 1), (0, 1)])
    assert target.dim == 2
    assert target(np.array([[1.0, 2.0]]))[0, 0] == 21.0


def test_a_formula_without_a_range_uses_minus_one_to_one():
    assert make_target(formula="x").domain == ((-1.0, 1.0),)
    assert make_target(formula="x*y").domain == ((-1.0, 1.0), (-1.0, 1.0))


def test_a_constant_formula_gives_one_value_per_input():
    assert make_target(formula="2")(np.zeros((5, 1))).shape == (5, 1)


@pytest.mark.parametrize("attack", [
    "__import__('os').system('whoami')",
    "x.__class__",
    "(lambda: 1)()",
    "open('secret')",
    "[1, 2]",
    "x if x else 1",
    "x < 1",
    "'text'",
    "True",
    "1j",
    "sin(*x)",
    "sin(x=1)",
    "exp(x)(x)",
])
def test_a_formula_cannot_run_code(attack):
    with pytest.raises(ValueError):
        make_target(formula=attack)


@pytest.mark.parametrize("text,hint", [
    ("", "Enter a formula"),
    ("1 +", "Could not read"),
    ("2x", "2*x"),
    ("w + 1", "'w'"),
    ("sin", "needs brackets"),
    ("sin()", "takes 1"),
    ("sin(x, x)", "takes 1"),
    ("min(x)", "takes 2"),
    ("x" * 201, "too long"),
])
def test_a_bad_formula_says_what_is_wrong(text, hint):
    with pytest.raises(ValueError, match=hint):
        make_target(formula=text)


@pytest.mark.parametrize("domain", [[(1, 1)], [(2, 1)], [(0, float("inf"))], [(0, float("nan"))]])
def test_a_bad_range_is_refused(domain):
    with pytest.raises(ValueError, match="range"):
        make_target(formula="x", domain=domain)


def test_the_number_of_ranges_must_fit_the_formula():
    with pytest.raises(ValueError, match="2 inputs"):
        make_target(formula="x*y", domain=[(0, 1)])


# --------------------------------------------------------------- missing middle
def test_no_training_or_validation_point_falls_in_the_missing_middle():
    ds = make_dataset(make_target(formula="x^3", domain=[(-2, 2)], gap=[(-0.5, 0.5)]), 500, 100)
    for x in (ds.x_train, ds.x_val):
        assert len(x) in (500, 100)
        assert not np.any(np.abs(x) <= 0.5)
    assert np.any(np.abs(ds.x_test) <= 0.5)  # the score still covers the hole


def test_a_two_dimensional_hole_keeps_the_rest_of_its_rows():
    ds = make_dataset(make_target(formula="x*y", gap=[(-0.3, 0.3), (-0.3, 0.3)]), 2000, 10)
    in_hole = np.all(np.abs(ds.x_train) <= 0.3, axis=1)
    assert not in_hole.any()
    assert np.any(np.abs(ds.x_train[:, 0]) <= 0.3)  # only the square is empty, not the whole band


def test_every_built_in_gap_target_really_has_a_hole():
    gapped = [t for t in TARGETS.values() if t.gap]
    assert len(gapped) >= 4 and any(t.dim == 2 for t in gapped)
    for t in gapped:
        assert not t.in_gap(make_dataset(t, 300, 50).x_train).any(), t.key


def test_sampling_is_unchanged_for_a_target_without_a_hole():
    # published results depend on this random stream
    ds = make_dataset(get_target("f1"), 50, 10)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(ds.x_train, rng.uniform([-1.0], [1.0], size=(50, 1)))
    np.testing.assert_array_equal(ds.x_val, rng.uniform([-1.0], [1.0], size=(10, 1)))
    assert len(ds.x_test) == 10_000


def test_a_built_in_target_can_get_or_lose_a_hole():
    assert make_target(target="f1", gap=[(-0.2, 0.2)]).gap == ((-0.2, 0.2),)
    assert TARGETS["f1"].gap is None  # the shared target stays as it was
    assert make_target(target="m1").gap == TARGETS["m1"].gap
    assert make_target(target="m1", gap=[]).gap is None


@pytest.mark.parametrize("gap,message", [
    ([(-2, 2)], "almost nothing"),         # hides the whole range
    ([(5, 6)], "outside"),
    ([(0.5, 0.1)], "range"),
    ([(0, 1), (0, 1)], "2 ranges|1 range"),
])
def test_a_useless_hole_is_refused(gap, message):
    with pytest.raises(ValueError, match=message):
        make_target(formula="x", domain=[(-1, 1)], gap=gap)


def test_the_error_is_split_between_the_hole_and_the_rest():
    y, y_hat = np.zeros((4, 1)), np.array([[0.0], [0.0], [3.0], [4.0]])
    m = regression_metrics(y_hat, y, hidden=np.array([False, False, True, True]))
    assert m["rmse_seen"] == 0.0
    assert m["rmse_gap"] == pytest.approx(math.sqrt(12.5))
    assert m["max_err_gap"] == 4.0
    assert "rmse_gap" not in regression_metrics(y_hat, y)


def _train(target, epochs=30, **data):
    ds = make_dataset(target, **data)
    net = build_mlp(target.dim, [8], "tanh", seed=1)
    fit(net, ds.u_train, ds.z_train, ds.u_val, ds.z_val,
        TrainConfig(optimizer="lm", epochs=epochs, patience=0, seed=1))
    return ds, net


def test_evaluate_reports_the_hole_on_its_own():
    ds, net = _train(get_target("m4"), n_train=200, n_val=50)
    m = evaluate(net, ds)
    assert m["rmse_gap"] > m["rmse_seen"] > 0  # the curve was fitted to one and has to guess the other


# ------------------------------------------------- functions undefined in places
def test_where_a_formula_is_undefined_nothing_is_sampled_or_scored():
    target = make_target(formula="sqrt(x^2 - 0.25)", domain=[(-1, 1)])
    ds, net = _train(target, n_train=200, n_val=50)
    assert not np.any(np.abs(ds.x_train) < 0.5)
    assert not np.any(np.abs(ds.x_test) < 0.5)
    assert np.isfinite(ds.y_test).all()
    assert math.isfinite(evaluate(net, ds)["rmse"])


def test_a_pole_counts_as_undefined():
    ds = make_dataset(make_target(formula="1/x", domain=[(-1, 1)]), 100, 10, n_test=101)
    assert np.isfinite(ds.y_test).all() and len(ds.y_test) == 100  # x = 0 is dropped


def test_a_formula_that_is_never_defined_is_refused():
    with pytest.raises(ValueError, match="almost nothing"):
        make_target(formula="sqrt(-1 - x^2)")


def test_no_warnings_escape_from_a_formula():
    with np.errstate(all="raise"):
        make_target(formula="log(x) + 1/x", domain=[(-1, 1)])(grid(((-1.0, 1.0),), 101))


# ----------------------------------------------------------------------- points
WAVE = [(x, math.sin(3 * x)) for x in np.linspace(-1, 1, 41)]


def test_points_are_split_and_never_invented():
    target = make_target(points=WAVE)
    assert target.dim == 1 and not target.known and target.domain == ((-1.0, 1.0),)
    ds = make_dataset(target, data_seed=3)
    assert len(ds.x_train) + len(ds.x_val) == 41 and len(ds.x_val) == 8
    assert not set(ds.x_train.ravel()) & set(ds.x_val.ravel())
    np.testing.assert_array_equal(np.sort(ds.x_test.ravel()), np.sort(ds.x_val.ravel()))  # scored on held-out points only


def test_points_in_the_hole_are_hidden_from_training_and_used_as_the_answer():
    ds, net = _train(make_target(points=WAVE, gap=[(-0.3, 0.3)]))
    assert not np.any(np.abs(ds.x_train) <= 0.3) and not np.any(np.abs(ds.x_val) <= 0.3)
    assert np.sum(np.abs(ds.x_test) <= 0.3) == sum(abs(x) <= 0.3 for x, _ in WAVE)
    m = evaluate(net, ds)
    assert m["rmse_gap"] > 0 and m["rmse_seen"] > 0


def test_points_need_no_order_and_set_their_own_range():
    target = make_target(points=[(3.0, 1.0), (-2.0, 0.0), *[(i * 0.1, i) for i in range(8)]])
    assert target.domain == ((-2.0, 3.0),)


@pytest.mark.parametrize("points,message", [
    ([], "At least 8"),
    ([(0, 0)] * 7, "At least 8"),
    ([(1.0, i) for i in range(9)], "same x"),
    ([(i, float("nan")) for i in range(9)], "finite"),
    ([(i, 1, 2) for i in range(9)], "x, y"),
])
def test_bad_points_are_refused(points, message):
    with pytest.raises(ValueError, match=message):
        make_target(points=points)


def test_too_many_points_in_the_hole_is_refused():
    with pytest.raises(ValueError, match="outside"):
        make_target(points=WAVE, gap=[(-0.9, 0.9)])


def test_a_formula_and_points_together_are_refused():
    with pytest.raises(ValueError, match="either"):
        make_target(formula="x", points=WAVE)


def test_an_unknown_target_is_a_value_error():
    with pytest.raises(ValueError, match="f99"):
        make_target(target="f99")


# ---------------------------------------------------------- saving and the CLI
def test_a_saved_description_rebuilds_the_function():
    assert saved_target({"target": "f4"}) is TARGETS["f4"]
    assert saved_target({}) is None
    assert saved_target({"target": "gone"}) is None
    rebuilt = saved_target({"target": "custom", "function": {"formula": "x^2", "domain": [[-2, 2]], "gap": [[-1, 1]]}})
    assert rebuilt.gap == ((-1.0, 1.0),) and rebuilt(np.array([[2.0]]))[0, 0] == 4.0


def _metrics(capsys) -> dict:
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


FAST = ["--hidden", "8", "--optimizer", "lm", "--epochs", "10", "--patience", "0", "--n-train", "100", "--n-val", "30"]


def test_the_command_line_trains_on_a_formula_with_a_hole(capsys, workdir):
    path = workdir / "m.npz"
    assert cli.main(["train", "--formula", "x^3 - x", "--domain=-1.5,1.5", "--gap=-0.5,0.5", *FAST, "--out", str(path)]) == 0
    m = _metrics(capsys)
    assert m["rmse_gap"] > 0
    assert cli.main(["predict", str(path), "1"]) == 0
    assert "exact=0" in capsys.readouterr().out


def test_the_command_line_trains_on_a_points_file(capsys, workdir):
    points, model = workdir / "p.csv", workdir / "p.npz"
    points.write_text("x,y\n" + "\n".join(f"{x},{y}" for x, y in WAVE), encoding="utf-8")
    assert cli.main(["train", "--points", str(points), *FAST, "--out", str(model)]) == 0
    assert _metrics(capsys)["rmse"] > 0
    assert cli.main(["predict", str(model), "0.5"]) == 0
    assert "exact" not in capsys.readouterr().out  # measured data has no exact function to quote


@pytest.mark.parametrize("args", [["--formula", "import os"], ["--points", "no-such-file.csv"], ["--formula", "x", "--gap=1"]])
def test_the_command_line_explains_bad_input(args, capsys):
    assert cli.main(["train", *args, *FAST]) == 2
    assert capsys.readouterr().err.strip()


def test_targets_lists_the_holes(capsys):
    cli.main(["targets"])
    assert "missing" in capsys.readouterr().out
