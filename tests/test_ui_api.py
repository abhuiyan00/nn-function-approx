"""HTTP contract of the visual interface (needs the [ui] extra; skipped without it)."""
import json
import math
import re
import sys
import time
import warnings

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

with warnings.catch_warnings():  # starlette's notice about its own httpx dependency
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient  # noqa: E402

from funcprox import cli  # noqa: E402
from funcprox.data import make_dataset  # noqa: E402
from funcprox.functions import get_target  # noqa: E402
from funcprox_ui import catalog, lab, search  # noqa: E402
from funcprox_ui.app import create_app  # noqa: E402
from funcprox_ui.jobs import JobQueue  # noqa: E402
from funcprox_ui.lab import LabRun, matched_baselines  # noqa: E402
from funcprox_ui.schemas import RunSpec  # noqa: E402

ORIGIN = "http://127.0.0.1"
FAST_LM = dict(target="f4", hidden=[6], optimizer="lm", epochs=15, patience=0, n_train=120, n_val=40)


@pytest.fixture
def client(workdir):
    with TestClient(create_app(models_dir=workdir), base_url=ORIGIN) as c:
        yield c


def wait_for(client, run_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/lab/runs/{run_id}").json()
        if run["status"] in ("done", "stopped", "failed"):
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def train(client, **spec):
    r = client.post("/api/lab/runs", json=spec)
    assert r.status_code == 201, r.text
    return wait_for(client, r.json()["id"])


# ---------------------------------------------------------------------- catalog
def test_targets_come_with_previews_of_the_right_shape(client):
    targets = {t["key"]: t for t in client.get("/api/targets").json()}
    assert sorted(targets) == ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "m1", "m2", "m3", "m4", "m5"]
    assert targets["m1"]["preview"]["gap"] == [[-0.25, 0.25]] and targets["f1"]["preview"]["gap"] is None
    assert targets["f1"]["preview"]["kind"] == "1d"
    assert len(targets["f1"]["preview"]["x"]) == len(targets["f1"]["preview"]["exact"])
    p = targets["f5"]["preview"]
    assert p["kind"] == "2d" and len(p["exact"]) == len(p["x"]) * len(p["y"])
    assert targets["f2"]["preview"]["box"] == [-2.0, 2.0]
    assert targets["f2"]["preview"]["x"][0] == -3.0  # shows the extrapolation zone


def test_every_experiment_is_described_from_the_docs(client):
    exps = client.get("/api/experiments").json()
    assert [e["id"] for e in exps] == ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "u1", "u2", "u3", "u4"]
    for e in exps:
        assert e["question"], e["id"]
        assert e["hypothesis"], e["id"]
        assert e["owner"] in ("A", "B")
        assert e["profile"] == "quick"
    detail = client.get("/api/experiments/e6").json()
    assert "Chebyshev" in detail["findings"]
    assert "Chebyshev" in detail["summary"]
    assert any(f["url"] == "/figures/e6_baselines_scaling.png" for f in detail["figures"])
    assert client.get(detail["figures"][0]["url"]).status_code == 200
    assert [d["name"] for d in detail["datasets"]] == ["runs"]


def test_dataset_rows_keep_unicode_and_infer_numbers(client):
    d = client.get("/api/datasets/e6/runs").json()
    assert {"MLP 2×H · LM", "Chebyshev LS"} <= {r["model"] for r in d["rows"]}
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["rmse"]["type"] == "number" and cols["model"]["type"] == "string"
    assert cols["data_seed"]["constant"] is True
    assert all(r["seed"] is None for r in d["rows"] if r["model"] == "Chebyshev LS")
    assert isinstance(d["rows"][0]["rmse"], float)


@pytest.mark.parametrize("exp,name,column", [("e0", "gradcheck", "error"), ("u3", "curves", "rel_error"),
                                             ("u4", "coverage", "coverage"), ("u4", "ensembles", "spread")])
def test_json_results_are_exposed_as_tidy_rows(client, exp, name, column):
    rows = client.get(f"/api/datasets/{exp}/{name}").json()["rows"]
    assert rows and all(column in r for r in rows)


@pytest.mark.parametrize("path", ["/api/datasets/e6/nope", "/api/datasets/zz/runs", "/api/experiments/zz",
                                  "/api/datasets/e6/..%2F..%2Fsecret", "/api/documents/nope", "/api/nope"])
def test_unknown_resources_are_json_404s(client, path):
    r = client.get(path)
    assert r.status_code == 404 and "detail" in r.json()


def test_overview_is_computed_from_the_result_files(client):
    o = client.get("/api/overview").json()
    assert o["profiles"] == {"quick": 12}
    assert o["run_count"] > 500
    t = o["highlights"]
    assert round(t["lm_vs_adam"]["f4"]) == 81 and round(t["lm_vs_adam"]["f5"]) == 54
    assert t["kan"]["n_params"] == 74 and t["kan"]["rmse"] < t["kan"]["best_mlp"]
    assert t["spectral"]["tanh_learned_at"] is None and t["spectral"]["fourier_learned_at"] == 100
    assert math.isclose(t["conformal"]["coverage"], 0.902, abs_tol=0.001)
    assert t["engine"]["worst_error"] < 1e-9
    verdicts = {v["target"]: v for v in o["verdicts"]}
    assert verdicts["f1"]["winner"] == "classical" and verdicts["f1"]["orders"] > 9
    assert verdicts["f6"]["winner"] == "network"
    assert verdicts["f5"]["winner"] == "tie"  # 1.3x is inside the 1.5x seed spread the project does not claim
    assert o["deadline"] == "2026-12-14"
    assert any(m["label"].startswith("Consultation") and m["milestone"] for m in o["timeline"])
    assert any(m["label"] == "Member A, E0-E5 analysis and figures" for m in o["timeline"])


def test_a_checkout_without_results_still_serves_every_page(client, workdir, monkeypatch):
    monkeypatch.setattr(catalog, "RESULTS", workdir)  # an empty folder: nothing has been run yet
    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["highlights"] == {} and overview.json()["verdicts"] == []
    experiments = client.get("/api/experiments").json()
    assert len(experiments) == 12 and experiments[0]["profile"] is None
    assert client.get("/api/experiments/e6").json()["datasets"] == []
    assert client.get("/api/datasets").json() == []


def test_the_images_of_the_readme_are_served(client):
    readme = (catalog.ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\((?!https?:)([^)]+)\)", readme)
    assert any("screenshots/" in image for image in images)
    for image in images:
        folder, name = image.split("/")[-2:]
        served = client.get(f"/{folder}/{name}")
        assert served.status_code == 200 and served.headers["content-type"].startswith("image/"), image  # not the page fallback


def test_documents_are_whitelisted_markdown(client):
    docs = {d["slug"]: d for d in client.get("/api/documents").json()}
    assert {"readme", "findings", "project_plan", "results"} <= set(docs)
    assert client.get("/api/documents/findings").json()["markdown"].startswith("# Findings")
    assert client.get("/api/documents/..%2F..%2Fpyproject").status_code == 404


# -------------------------------------------------------------------------- lab
@pytest.mark.parametrize("bad", [
    dict(target="f99"),
    dict(optimizer="magic"),
    dict(hidden=[8] * 7),
    dict(hidden=[0]),
    dict(extra_field=1),
    dict(optimizer="lm", hidden=[64, 64]),                # P x P solve: too large for interactive use
    dict(arch="kan", target="f7", widths=[1, 5, 1]),      # first width must be the input dimension
    dict(arch="kan", target="f1", widths=[1, 5, 2]),      # scalar output only
    dict(lr=0),
    dict(epochs=10 ** 9),
    dict(formula="__import__('os').system('whoami')"),
    dict(formula="x", points=[[i, i] for i in range(9)]),
    dict(target="custom"),
    dict(points=[[0, 0]]),
    dict(formula="x", gap=[[5, 6]]),
    dict(target="f1", gap=[[0, 1], [0, 1]]),
    dict(formula="x*y", arch="kan", widths=[1, 5, 1]),
])
def test_invalid_specs_are_rejected_before_training(client, bad):
    assert client.post("/api/lab/runs", json=bad).status_code == 422


def test_lab_run_reports_the_same_metrics_as_the_cli(client, capsys):
    run = train(client, seed=3, **FAST_LM)
    assert run["status"] == "done"
    cli.main(["train", "--target", "f4", "--hidden", "6", "--optimizer", "lm", "--epochs", "15", "--patience", "0",
              "--n-train", "120", "--n-val", "40", "--seed", "3"])
    out = capsys.readouterr().out
    expected = json.loads(out[out.index("{"):])
    assert run["final"]["metrics"]["rmse"] == pytest.approx(expected["rmse"], rel=1e-9)
    assert run["n_params"] == expected["n_params"]
    assert run["history"]["epoch"] == list(range(1, 16))
    assert len(run["history"]["mu"]) == 15


def test_finished_run_compares_against_classical_baselines(client):
    final = train(client, **FAST_LM)["final"]
    names = {b["model"]: b for b in final["baselines"]}
    assert set(names) == {"Chebyshev LS", "Cubic spline LS", "RBF network LS"}
    assert names["Chebyshev LS"]["n_coef"] <= final["metrics"]["n_params"]
    assert names["Chebyshev LS"]["rmse"] > 0


def test_stream_delivers_init_frames_and_done_in_order(client):
    rid = client.post("/api/lab/runs", json=dict(FAST_LM, epochs=60)).json()["id"]
    events = []
    with client.stream("GET", f"/api/lab/runs/{rid}/stream") as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        for line in s.iter_lines():
            if line.startswith("event: "):
                events.append(line[7:])
            if events and events[-1] == "done" and line.startswith("data: "):
                final = json.loads(line[6:])
                break
    assert events[0] == "init" and events[-1] == "done"
    assert final["final"]["metrics"]["rmse"] > 0


def test_stream_replays_a_finished_run(client):
    rid = train(client, **FAST_LM)["id"]
    with client.stream("GET", f"/api/lab/runs/{rid}/stream") as s:
        text = "\n".join(s.iter_lines())
    assert text.index("event: init") < text.index("event: done")
    init = json.loads(text.split("event: init\ndata: ")[1].split("\n")[0])
    assert init["history"]["epoch"][-1] == 15  # a late subscriber still gets the whole curve


def test_stop_ends_training_early_and_keeps_best_weights(client):
    rid = client.post("/api/lab/runs", json=dict(target="f1", hidden=[32, 32], epochs=50000, patience=0)).json()["id"]
    while not client.get(f"/api/lab/runs/{rid}").json()["history"]["epoch"]:
        time.sleep(0.01)
    assert client.post(f"/api/lab/runs/{rid}/stop").status_code == 200
    run = wait_for(client, rid)
    assert run["status"] == "stopped" and run["final"]["metrics"]["rmse"] > 0
    assert len(run["history"]["epoch"]) < 50000


def test_queued_run_can_be_stopped_before_it_starts(client):
    first = client.post("/api/lab/runs", json=dict(target="f1", epochs=50000, patience=0)).json()["id"]
    second = client.post("/api/lab/runs", json=dict(FAST_LM)).json()["id"]
    client.post(f"/api/lab/runs/{second}/stop")
    client.post(f"/api/lab/runs/{first}/stop")
    assert wait_for(client, second)["status"] == "stopped"
    assert wait_for(client, first)["status"] == "stopped"


def test_two_dimensional_run_returns_a_prediction_surface(client):
    run = train(client, target="f7", hidden=[6], optimizer="lm", epochs=5, patience=0, n_train=100, n_val=30)
    g = run["grid"]
    assert g["kind"] == "2d"
    n = len(g["x"]) * len(g["y"])
    assert len(g["exact"]) == n and len(run["final"]["pred"]) == n and len(run["final"]["residual"]) == n
    assert run["final"]["spectrum"] is None


KAN_F7 = dict(target="f7", arch="kan", widths=[2, 1, 1], grid=10, grid_updates=[25, 75], optimizer="lm",
              epochs=150, patience=0, n_train=1000, n_val=200, seed=1)


def test_kan_run_exposes_learned_edges_and_reads_the_formula_on_f7(client):
    run = train(client, **KAN_F7)
    edges = run["final"]["edges"]
    assert run["final"]["metrics"]["rmse"] < 5e-3
    assert [(e["layer"], e["i"], e["o"]) for e in edges] == [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    assert len(edges[0]["x"]) == len(edges[0]["y"])
    names = [e["match"]["name"] for e in edges]
    assert names[:2] == ["sin(πx)", "y²"] and re.fullmatch(r"exp\([\d.]+·s\)", names[2])
    assert all(e["match"]["r2"] > 0.99 for e in edges)


def test_kan_grid_updates_match_the_cli(client, capsys):
    run = train(client, **KAN_F7)
    cli.main(["train", "--target", "f7", "--arch", "kan", "--widths", "2,1,1", "--grid", "10", "--grid-updates", "25,75",
              "--optimizer", "lm", "--epochs", "150", "--patience", "0", "--n-train", "1000", "--n-val", "200", "--seed", "1"])
    out = capsys.readouterr().out
    assert run["final"]["metrics"]["rmse"] == pytest.approx(json.loads(out[out.index("{"):])["rmse"], rel=1e-9)


def test_grid_updates_only_apply_to_kans(client):
    assert client.post("/api/lab/runs", json=dict(grid_updates=[25])).status_code == 422


def test_f3_tracks_each_frequency_component(client):
    run = train(client, target="f3", hidden=[8], optimizer="lm", epochs=10, patience=0, n_train=200, n_val=50)
    tr = run["components"]
    assert tr["k"] == [1, 4, 16]
    assert len(tr["rel"]) == len(tr["epoch"]) >= 1 and len(tr["rel"][0]) == 3
    assert len(run["final"]["spectrum"]["pred"]) == len(run["grid"]["spectrum_exact"])


def test_diverged_training_is_reported_with_valid_json(client):
    run = train(client, target="f1", hidden=[8], optimizer="sgd", lr=10, batch=0, epochs=50, patience=0,
                n_train=100, n_val=30)
    assert run["status"] == "done"
    assert run["final"]["stop"] in ("diverged", "max epochs")


def test_restored_best_weights_do_not_fake_a_drop_on_the_exact_f_curve(monkeypatch):
    # a small noisy data set overfits: the best validation epoch comes long before the last one
    monkeypatch.setattr(lab, "MIN_SNAPSHOT_S", float("inf"))  # no timed snapshots: the clock must not decide this test
    run = LabRun(RunSpec(target="f1", hidden=[32, 32], sigma=0.5, n_train=30, n_val=10, epochs=400, patience=0, seed=1))
    run.execute()
    assert run.final["best_epoch"] < run.final["epochs_run"]
    assert run.test_epoch == [0]  # the snapshot of the restored weights stays off the curve


def test_the_network_view_gets_units_links_and_values(client):
    run = train(client, target="f5", hidden=[40, 3], fourier_m=4, epochs=2, patience=0, n_train=50, n_val=10)
    view = run["live"]["net"]
    assert view["names"] == ["input", "Fourier", "tanh", "tanh", "output"]
    assert view["sizes"] == [2, 8, 40, 3, 1] and view["shown"] == [2, 8, lab.VIEW_UNITS, 3, 1]
    assert [len(w) for w in view["w"]] == [a * b for a, b in zip(view["shown"], view["shown"][1:])]
    assert [len(a) for a in view["act"]] == [lab.VIEW_PROBES * n for n in view["shown"]]
    assert len(view["x"]) == 2 * lab.VIEW_PROBES and len(view["y"]) == lab.VIEW_PROBES
    kan = train(client, **KAN_F7)["live"]["net"]
    assert kan["kan"] and kan["sizes"] == [2, 1, 1] and all(w >= 0 for link in kan["w"] for w in link)


def test_baselines_for_a_large_network_stay_small_enough_to_solve():
    ds = make_dataset(get_target("f5"), 4000, 100)
    out = matched_baselines(ds, n_params=100_000)
    assert all(b["n_coef"] <= 441 for b in out)  # the largest 2-D fit of experiment E6


def _baseline(target, n_params, **data):
    return {b["model"]: b for b in matched_baselines(make_dataset(get_target(target), **data), n_params)}


def test_baselines_do_not_blow_up_at_high_degree():
    # degree 256 on 2,000 random points is numerically unstable: RMSE 2e3 on |x|
    assert _baseline("f6", 8513, n_train=2000, n_val=500)["Chebyshev LS"]["rmse"] < 0.01


def test_baselines_do_not_blow_up_on_few_noisy_points():
    # degree 49 through 100 noisy points: RMSE 28 on sin 2 pi x
    out = _baseline("f1", 16897, n_train=100, n_val=25, sigma=0.2)
    assert out["Chebyshev LS"]["rmse"] < 0.2 and out["Cubic spline LS"]["rmse"] < 0.2


def test_baselines_are_never_larger_than_a_small_network():
    out = _baseline("f4", 33, n_train=2000, n_val=500)
    assert all(b["n_coef"] <= 33 for b in out.values())


def test_closing_the_interface_stops_a_running_experiment(workdir):
    jobs = _queue("import time\nprint('up', flush=True)\ntime.sleep(60)")
    with TestClient(create_app(models_dir=workdir, jobs=jobs), base_url=ORIGIN) as c:
        job_id = c.post("/api/jobs", json=dict(experiment="e1", profile="quick")).json()["id"]
        while "up" not in c.get(f"/api/jobs/{job_id}").json()["log"]:
            time.sleep(0.02)
    job = jobs.jobs[job_id]
    deadline = time.monotonic() + 10
    while not job.over and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.status == "cancelled" and job.process.poll() is not None


def test_completed_runs_can_be_listed_and_removed(client):
    rid = train(client, **FAST_LM)["id"]
    listed = client.get("/api/lab/runs").json()
    assert listed[0]["id"] == rid and "history" not in listed[0]
    assert listed[0]["label"] == "MLP 1×6 tanh · LM"
    assert client.delete(f"/api/lab/runs/{rid}").status_code == 204
    assert client.get(f"/api/lab/runs/{rid}").status_code == 404


# ----------------------------------------------------------------------- models
def test_saved_model_can_be_listed_inspected_and_queried(client, workdir):
    rid = train(client, seed=2, **FAST_LM)["id"]
    assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name="runge-demo")).status_code == 201
    assert (workdir / "runge-demo.npz").exists()
    listed = client.get("/api/models").json()
    assert [m["name"] for m in listed] == ["runge-demo"]
    assert listed[0]["target"] == "f4" and listed[0]["rmse"] > 0
    detail = client.get("/api/models/runge-demo").json()
    assert detail["grid"]["kind"] == "1d" and len(detail["pred"]) == len(detail["grid"]["x"])
    q = client.post("/api/models/runge-demo/predict", json=dict(inputs=[[0.0], [0.5]])).json()
    assert q["exact"][0] == pytest.approx(1.0)
    assert q["abs_error"][0] == pytest.approx(abs(q["y_hat"][0] - 1.0))
    assert client.post("/api/models/runge-demo/predict", json=dict(inputs=[[0.0, 1.0]])).status_code == 422


def test_saving_never_overwrites_and_refuses_unsafe_names(client, workdir):
    rid = train(client, **FAST_LM)["id"]
    assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name="one")).status_code == 201
    assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name="one")).status_code == 409
    for bad in ("../evil", "a/b", "a\\b", ".hidden", "", "x" * 100):
        assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name=bad)).status_code == 422, bad
    assert client.get("/api/models/..%2Fevil").status_code == 404
    assert sorted(p.name for p in workdir.iterdir()) == ["one.npz"]


# ------------------------------------------------------------ your own function
FUNCTION = dict(formula="x^3 - x", domain=[[-1.5, 1.5]], gap=[[-0.5, 0.5]])
FAST = dict(hidden=[8], optimizer="lm", epochs=12, patience=0, n_train=120, n_val=40)
WAVE = [[-1 + i / 20, math.sin(3 * (-1 + i / 20))] for i in range(41)]


def preview(client, **function):
    r = client.post("/api/functions/preview", json=function)
    assert r.status_code == 200, r.text
    return r.json()


def test_preview_shows_the_function_its_hole_and_where_the_data_will_be(client):
    p = preview(client, **FUNCTION)
    assert p["key"] == "custom" and p["dim"] == 1 and p["formula"] == "x^3 - x"
    assert p["grid"]["gap"] == [[-0.5, 0.5]] and p["grid"]["known"] is True
    assert len(p["grid"]["x"]) == len(p["grid"]["exact"])
    assert len(p["samples"]["x"]) > 50 and all(abs(x) > 0.5 for x in p["samples"]["x"])


def test_preview_of_a_built_in_target_and_of_a_surface(client):
    assert preview(client, target="m1")["grid"]["gap"] == [[-0.25, 0.25]]
    surface = preview(client, formula="x*y", gap=[[-0.3, 0.3], [-0.3, 0.3]])
    assert surface["dim"] == 2 and surface["grid"]["kind"] == "2d"
    inside = [abs(x) <= 0.3 and abs(y) <= 0.3 for x, y in zip(surface["samples"]["x"], surface["samples"]["y"])]
    assert not any(inside)


def test_preview_marks_undefined_values_as_null(client):
    g = preview(client, formula="sqrt(x^2 - 0.25)")["grid"]
    middle = [v for x, v in zip(g["x"], g["exact"]) if abs(x) < 0.45]
    assert middle and all(v is None for v in middle)
    assert g["exact"][0] == pytest.approx(math.sqrt(0.75), rel=1e-3)


def test_preview_of_points_invents_no_exact_function(client):
    p = preview(client, points=WAVE, gap=[[-0.2, 0.2]])
    assert p["grid"]["known"] is False and all(v is None for v in p["grid"]["exact"])
    assert p["samples"]["x"] and all(abs(x) > 0.2 for x in p["samples"]["x"])


@pytest.mark.parametrize("function,message", [
    (dict(formula="2x"), "2*x"),
    (dict(formula="open('a')"), "open"),
    (dict(points=[[0, 0]]), "At least 8"),
    (dict(formula="x", gap=[[-9, 9]]), "almost nothing"),
    (dict(target="nope"), "nope"),
])
def test_preview_says_what_is_wrong(client, function, message):
    r = client.post("/api/functions/preview", json=function)
    assert r.status_code == 422 and message in r.json()["detail"][0]["msg"]


def test_a_formula_with_a_hole_trains_like_on_the_command_line(client, capsys):
    run = train(client, **FUNCTION, **FAST, seed=2)
    assert run["status"] == "done" and run["spec"]["target"] == "custom"
    assert run["formula"] == "x^3 - x" and run["grid"]["gap"] == [[-0.5, 0.5]]
    m = run["final"]["metrics"]
    cli.main(["train", "--formula", "x^3 - x", "--domain=-1.5,1.5", "--gap=-0.5,0.5", "--hidden", "8", "--optimizer", "lm",
              "--epochs", "12", "--patience", "0", "--n-train", "120", "--n-val", "40", "--seed", "2"])
    out = capsys.readouterr().out
    expected = json.loads(out[out.index("{"):])
    assert m["rmse"] == pytest.approx(expected["rmse"], rel=1e-9)
    assert m["rmse_gap"] == pytest.approx(expected["rmse_gap"], rel=1e-9) and m["rmse_gap"] > 0
    assert run["live"]["rmse_gap"] > 0


def test_the_baselines_are_split_at_the_hole_too(client):
    baselines = train(client, **FUNCTION, **FAST)["final"]["baselines"]
    assert baselines and all(b["rmse_gap"] > 0 and b["rmse_seen"] >= 0 for b in baselines)


def test_a_built_in_target_can_be_given_a_hole_in_the_lab(client):
    run = train(client, target="f4", gap=[[-0.2, 0.2]], **FAST)
    assert run["spec"]["target"] == "f4" and run["final"]["metrics"]["rmse_gap"] > 0


def test_measured_points_train_and_stay_valid_json(client):
    run = train(client, points=WAVE, gap=[[-0.2, 0.2]], **FAST)
    assert run["status"] == "done", run["error"]
    assert run["grid"]["known"] is False and all(v is None for v in run["final"]["residual"])
    assert all(v is not None for v in run["final"]["pred"])
    assert run["final"]["metrics"]["rmse_gap"] > 0 and run["final"]["spectrum"] is None
    assert run["live"]["rmse"] > 0 and run["history"]["test_mse"]
    assert all(abs(x) > 0.2 for x in run["samples"]["x"])


def test_a_function_undefined_in_the_middle_trains_and_stays_valid_json(client):
    run = train(client, formula="sqrt(x^2 - 0.25)", **FAST)
    assert run["status"] == "done", run["error"]
    assert run["final"]["metrics"]["rmse"] > 0 and run["live"]["rmse"] > 0
    assert run["grid"]["spectrum_exact"] is None
    middle = [p for x, p in zip(run["grid"]["x"], run["final"]["pred"]) if abs(x) < 0.45]
    assert all(p is not None for p in middle)  # the network still draws a curve there


def test_a_kan_takes_its_inputs_from_the_formula(client):
    r = client.post("/api/lab/runs", json=dict(formula="x*y", arch="kan", optimizer="lm", epochs=3, patience=0, n_train=40, n_val=10))
    assert r.status_code == 201 and r.json()["label"].startswith("KAN [2,5,1]")


def test_a_model_on_a_formula_is_rebuilt_from_its_file(client):
    rid = train(client, **FUNCTION, **FAST)["id"]
    assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name="cubic")).status_code == 201
    detail = client.get("/api/models/cubic").json()
    assert detail["target"] == "custom" and detail["formula"] == "x^3 - x" and detail["grid"]["gap"] == [[-0.5, 0.5]]
    q = client.post("/api/models/cubic/predict", json=dict(inputs=[[1.0], [0.0]])).json()
    assert q["exact"] == [0.0, 0.0] and q["in_gap"] == [False, True]


def test_a_model_on_points_has_no_exact_answer(client):
    rid = train(client, points=WAVE, **FAST)["id"]
    assert client.post(f"/api/lab/runs/{rid}/save", json=dict(name="wave")).status_code == 201
    assert client.get("/api/models/wave").json()["grid"]["known"] is False
    q = client.post("/api/models/wave/predict", json=dict(inputs=[[0.5]])).json()
    assert q["exact"] is None and q["abs_error"] is None and q["y_hat"][0] is not None


def test_a_finished_run_answers_questions(client):
    run = train(client, **FUNCTION, **FAST)
    q = client.post(f"/api/lab/runs/{run['id']}/predict", json=dict(inputs=[[1.0], [0.0], [9.0]]))
    assert q.status_code == 200
    assert q.json()["in_gap"] == [False, True, False] and q.json()["outside_box"] == [False, False, True]
    assert client.post(f"/api/lab/runs/{run['id']}/predict", json=dict(inputs=[[1.0, 2.0]])).status_code == 422


def test_a_running_run_does_not_answer_yet(client):
    rid = client.post("/api/lab/runs", json=dict(target="f1", epochs=50000, patience=0)).json()["id"]
    assert client.post(f"/api/lab/runs/{rid}/predict", json=dict(inputs=[[0.0]])).status_code == 409
    client.post(f"/api/lab/runs/{rid}/stop")
    wait_for(client, rid)


# ------------------------------------------------------------- the closest curve
@pytest.fixture
def fast_search(monkeypatch):
    monkeypatch.setitem(search.EFFORT, "quick", dict(epochs=20, patience=0, iters=6))


def find(client, timeout=120.0, **body):
    r = client.post("/api/search", json=body)
    assert r.status_code == 201, r.text
    state = r.json()
    deadline = time.monotonic() + timeout
    while state["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        state = client.get(f"/api/search/{state['id']}").json()
    assert state["status"] != "running", "the search did not finish"
    return state


def test_search_ranks_every_model_by_how_close_it_gets(client, fast_search):
    s = find(client, base=dict(target="m4", n_train=120, n_val=40))
    assert s["status"] == "done" and s["known"] is True and s["done"] == s["total"] == 6
    rmse = [r["rmse"] for r in s["rows"]]
    assert rmse == sorted(rmse) and s["winner"] == s["rows"][0]["key"]
    assert s["best_network"] == next(r["key"] for r in s["rows"] if r["kind"] == "network")
    assert {r["kind"] for r in s["rows"]} == {"network", "classical"}
    assert {"Chebyshev LS", "Cubic spline LS", "RBF network LS"} <= {r["key"] for r in s["rows"]}
    assert len(s["rows"]) == 9 and sorted(r["slot"] for r in s["rows"]) == list(range(9))
    for r in s["rows"]:
        assert len(r["pred"]) == len(s["grid"]["x"]), r["key"]
        assert r["rmse_gap"] > 0 and r["rmse_seen"] >= 0 and r["n_params"] > 0


def test_search_names_the_choice_made_without_seeing_the_function(client, fast_search):
    s = find(client, base=dict(target="m4", n_train=120, n_val=40))
    val = {r["key"]: r["val_mse"] for r in s["rows"]}
    assert s["blind_pick"] == min(val, key=val.get)


def test_every_network_of_a_search_can_be_opened_in_the_lab(client, fast_search):
    s = find(client, base=dict(target="m1", n_train=120, n_val=40))
    runs = [r for r in s["rows"] if r["kind"] == "network"]
    assert len(runs) == 6
    assert all(client.get(f"/api/lab/runs/{r['best_run']}").status_code == 200 for r in runs)
    assert all(r["best_run"] is None for r in s["rows"] if r["kind"] == "classical")


def test_several_seeds_show_how_much_the_starting_point_matters(client, fast_search):
    s = find(client, base=dict(target="m4", n_train=120, n_val=40), seeds=2)
    assert s["total"] == 12
    for r in (r for r in s["rows"] if r["kind"] == "network"):
        assert r["seeds"] == 2 and r["rmse_lo"] <= r["rmse"] <= r["rmse_hi"]
        assert all(a <= b for a, b in zip(r["lo"], r["hi"]))


def test_search_on_a_surface_sends_only_the_surfaces_it_names(client, fast_search):
    s = find(client, base=dict(target="m5", n_train=120, n_val=40))
    assert s["grid"]["kind"] == "2d" and s["status"] == "done"
    named = (s["winner"], s["best_network"])
    for r in s["rows"]:
        assert (r["pred"] is not None) == (r["key"] in named), r["key"]
    assert len(s["rows"][0]["pred"]) == len(s["grid"]["exact"])


def test_search_works_on_measured_points(client, fast_search):
    s = find(client, base=dict(points=WAVE, gap=[[-0.2, 0.2]]))
    assert s["known"] is False and s["status"] == "done"
    assert s["rows"][0]["rmse"] > 0 and s["rows"][0]["rmse_gap"] > 0


def test_search_keeps_the_function_data_and_noise_it_was_given(client, fast_search):
    s = find(client, base=dict(formula="sin(3*x)", domain=[[-2, 2]], sigma=0.1, n_train=120, n_val=40, hidden=[4], optimizer="sgd"))
    assert s["formula"] == "sin(3*x)"
    network = next(r for r in s["rows"] if r["kind"] == "network")
    run = client.get(f"/api/lab/runs/{network['best_run']}").json()
    assert run["spec"]["sigma"] == 0.1 and run["spec"]["domain"] == [[-2.0, 2.0]]


def test_a_search_can_be_stopped_and_keeps_what_finished(client):
    state = client.post("/api/search", json=dict(base=dict(target="m1"), effort="thorough")).json()
    assert state["status"] == "running"
    assert client.post(f"/api/search/{state['id']}/stop").status_code == 200
    deadline = time.monotonic() + 30
    while state["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        state = client.get(f"/api/search/{state['id']}").json()
    assert state["status"] == "stopped"


@pytest.mark.parametrize("body", [dict(base=dict(formula="import os")), dict(base=dict(target="m1"), seeds=0),
                                  dict(base=dict(target="m1"), seeds=99), dict(base=dict(target="m1"), effort="huge")])
def test_a_bad_search_is_refused(client, body):
    assert client.post("/api/search", json=body).status_code == 422


def test_an_unknown_search_is_a_404(client):
    assert client.get("/api/search/nope").status_code == 404
    assert client.post("/api/search/nope/stop").status_code == 404


# ------------------------------------------------------------------------- jobs
def _queue(code):
    return JobQueue(command=lambda exp, profile, workers: [sys.executable, "-c", code])


def _wait_job(client, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_runs_a_process_and_keeps_its_unicode_log(workdir):
    app = create_app(models_dir=workdir, jobs=_queue("print('σ line'); print('two')"))
    with TestClient(app, base_url=ORIGIN) as c:
        job = c.post("/api/jobs", json=dict(experiment="e1", profile="quick")).json()
        finished = _wait_job(c, job["id"])
        assert finished["status"] == "done" and finished["log"] == ["σ line", "two"]
        with c.stream("GET", f"/api/jobs/{job['id']}/stream") as s:
            assert "σ line" in "\n".join(s.iter_lines())


def test_failing_job_is_marked_failed(workdir):
    app = create_app(models_dir=workdir, jobs=_queue("import sys; sys.exit(3)"))
    with TestClient(app, base_url=ORIGIN) as c:
        job = c.post("/api/jobs", json=dict(experiment="e1", profile="quick")).json()
        finished = _wait_job(c, job["id"])
        assert finished["status"] == "failed" and finished["returncode"] == 3


def test_running_job_can_be_cancelled(workdir):
    app = create_app(models_dir=workdir, jobs=_queue("import time\nprint('up', flush=True)\ntime.sleep(60)"))
    with TestClient(app, base_url=ORIGIN) as c:
        job = c.post("/api/jobs", json=dict(experiment="e1", profile="quick")).json()
        while "up" not in c.get(f"/api/jobs/{job['id']}").json()["log"]:
            time.sleep(0.02)
        assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        started = time.monotonic()
        assert _wait_job(c, job["id"])["status"] == "cancelled"
        assert time.monotonic() - started < 10


@pytest.mark.parametrize("bad", [dict(experiment="rm -rf", profile="quick"), dict(experiment="e1", profile="huge"),
                                 dict(experiment="../e1", profile="quick")])
def test_jobs_only_run_known_experiments(client, bad):
    assert client.post("/api/jobs", json=bad).status_code == 422


# --------------------------------------------------------------------- security
def test_requests_for_other_hosts_are_refused(workdir):
    # DNS rebinding: a page on evil.example resolving to 127.0.0.1 must not reach the API
    with TestClient(create_app(models_dir=workdir), base_url="http://evil.example") as c:
        assert c.get("/api/targets").status_code == 400


def test_cross_site_writes_are_refused(client):
    headers = {"Origin": "https://evil.example"}
    assert client.post("/api/lab/runs", json=FAST_LM, headers=headers).status_code == 403
    assert client.post("/api/lab/runs", json=FAST_LM, headers={"Origin": ORIGIN}).status_code == 201


# ------------------------------------------------------------------- frontend
def test_any_page_url_falls_through_to_the_app_shell(client):
    r = client.get("/experiments/e6")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


# ------------------------------------------------- analysis of one run, on request
def test_units_of_the_last_layer_add_up_to_the_answer(client):
    run = train(client, target="f1", hidden=[20], epochs=3, patience=0, n_train=60, n_val=20)
    units, pred = run["final"]["units"], run["final"]["pred"]
    assert len(units["curves"]) == lab.UNIT_CURVES and len(run["grid"]["units_x"]) == len(units["curves"][0])
    total = [sum(c[i] for c in units["curves"]) + units["rest"][i] + units["offset"] for i in range(len(units["rest"]))]
    assert total == pytest.approx(pred[::4], abs=2e-3)  # every curve is rounded to 4 digits
    assert train(client, target="f5", **FAST)["final"]["units"] is None  # one input only


def test_the_slope_of_the_network_follows_the_slope_of_the_function(client):
    run = train(client, target="f2", hidden=[16, 16], optimizer="lm", epochs=60, patience=0, n_train=300, n_val=60)
    slope = run["final"]["slope"]
    assert len(slope["pred"]) == len(run["grid"]["slope_exact"]) == len(run["grid"]["x"])
    assert slope["rmse"] < 0.5  # f2 = 2x^2 has slopes up to 8 inside the box
    i = run["grid"]["x"].index(min(run["grid"]["x"], key=lambda v: abs(v - 1.0)))
    assert run["grid"]["slope_exact"][i] == pytest.approx(4.0, abs=0.05)


def test_the_trust_band_covers_fresh_noisy_points(client):
    spec = dict(target="f1", hidden=[16], optimizer="lm", epochs=30, patience=0, n_train=300, n_val=60, sigma=0.2)
    band = train(client, **spec)["final"]["band"]
    assert band["level"] == 0.9 and band["n"] == lab.CAL_POINTS
    assert 0.2 < band["half"] < 0.6  # 1.645 sigma = 0.33 when only the noise is left
    assert train(client, points=WAVE, **FAST)["final"]["band"] is None  # measured data has no fresh points


def test_the_replay_keeps_evenly_spread_curves_and_ends_on_the_device(client):
    run = train(client, **FAST_LM)
    reel = client.get(f"/api/lab/runs/{run['id']}/replay").json()
    assert reel["epoch"][0] == 0 and len(reel["pred"]) == len(reel["epoch"])
    assert reel["pred"][-1] == run["final"]["pred"]
    reel = lab.Reel(4)
    for i in range(100):
        reel.add(lambda i=i: i)
    assert len(reel.items) <= 4 and reel.items[0] == 0 and len(set(b - a for a, b in zip(reel.items, reel.items[1:]))) == 1


def test_the_loss_landscape_shows_that_training_went_downhill(client):
    run = train(client, target="f4", hidden=[8, 8], optimizer="lm", epochs=40, patience=0, n_train=200, n_val=40)
    d = client.get(f"/api/lab/runs/{run['id']}/landscape").json()
    n = lab.LANDSCAPE_N
    assert len(d["alpha"]) == len(d["beta"]) == n and len(d["loss"]) == n * n
    assert d["path"]["epoch"][0] == 0 and len(d["path"]["a"]) == len(d["path"]["b"]) == len(d["path"]["epoch"])
    assert 0 < d["explained"] <= 1 + 1e-9

    def loss_at(a, b):
        i = min(range(n), key=lambda k: abs(d["alpha"][k] - a))
        j = min(range(n), key=lambda k: abs(d["beta"][k] - b))
        return d["loss"][i * n + j]

    assert loss_at(0, 0) < loss_at(d["path"]["a"][0], d["path"]["b"][0])
    queued = client.post("/api/lab/runs", json=dict(epochs=50_000, patience=0)).json()["id"]
    assert client.get(f"/api/lab/runs/{queued}/landscape").status_code == 409
    client.post(f"/api/lab/runs/{queued}/stop")


@pytest.mark.parametrize("spec", [
    dict(target="f4", hidden=[6], act="silu"),
    dict(target="f5", hidden=[6], act="sine", w0=10, fourier_m=3),
    dict(target="f7", arch="kan", widths=[2, 2, 1], grid=4),
])
def test_an_exported_device_answers_like_the_run(client, spec):
    run = train(client, **spec, optimizer="lm", epochs=5, patience=0, n_train=80, n_val=20)
    r = client.get(f"/api/lab/runs/{run['id']}/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert not re.search(r"^(import|from) funcprox", r.text, re.M)  # the file stands alone
    scope: dict = {}
    exec(r.text, scope)
    rows = [[0.1], [0.7]] if spec["target"] == "f4" else [[0.1, 0.2], [0.7, 0.4]]
    want = client.post(f"/api/lab/runs/{run['id']}/predict", json=dict(inputs=rows)).json()["y_hat"]
    assert scope["predict"](rows).tolist() == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_a_saved_model_exports_too(client):
    run = train(client, **FAST_LM)
    client.post(f"/api/lab/runs/{run['id']}/save", json=dict(name="device"))
    r = client.get("/api/models/device/export")
    assert r.status_code == 200 and "def predict(x)" in r.text
    assert client.get("/api/models/nothing/export").status_code == 404


# ------------------------------------------------------------- three inputs, RBF
def test_a_formula_of_three_inputs_trains_and_is_drawn_as_a_slice(client):
    run = train(client, formula="x*y + sin(pi*z)", domain=[[-1, 1], [0, 2], [0, 1]], gap=[[-0.2, 0.2], [0.8, 1.2], [0.3, 0.7]],
                hidden=[16, 16], optimizer="lm", epochs=30, patience=0, n_train=500, n_val=100)
    assert run["status"] == "done" and run["final"]["metrics"]["rmse"] < 0.1
    grid = run["grid"]
    assert grid["kind"] == "2d" and grid["fixed"] == [0.5] and grid["box"] == [[-1, 1], [0, 2]]
    assert grid["gap"] == [[-0.2, 0.2], [0.8, 1.2]]  # the slice at z = 0.5 passes through the hole
    assert [b["model"] for b in run["final"]["baselines"]] == ["RBF network LS"]  # the tensor grids stop at two inputs
    assert run["live"]["net"]["sizes"][0] == 3
    answer = client.post(f"/api/lab/runs/{run['id']}/predict", json=dict(inputs=[[0.5, 0.5, 0.25]])).json()
    assert answer["exact"][0] == pytest.approx(0.25 + math.sin(math.pi / 4))
    assert client.post("/api/lab/runs", json=dict(formula="x + w")).status_code == 422


def test_the_slice_drops_a_hole_it_does_not_cross(client):
    body = dict(formula="x + y + z", gap=[[-0.2, 0.2], [-0.2, 0.2], [0.5, 0.9]])
    assert client.post("/api/functions/preview", json=body).json()["grid"]["gap"] is None


def test_the_rbf_network_fits_a_smooth_function_in_any_dimension():
    from funcprox.baselines import RBFNet
    for key, limit in (("f4", 1e-3), ("f5", 5e-2)):
        ds = make_dataset(get_target(key), 1000, 100)
        model = RBFNet(48).fit(ds.u_train, ds.y_train)
        assert model.n_coef == 48 * (ds.target.dim + 1) + 1
        assert float(np.sqrt(np.mean((model.predict(ds.u_test) - ds.y_test) ** 2))) < limit


@pytest.mark.parametrize("name, f", [("x", lambda x: 3 * x - 1), ("x²", lambda x: x * x), ("log(x)", np.log), ("√x", np.sqrt),
                                     ("|x|", lambda x: 2 * abs(x))])
def test_an_edge_is_named_after_its_closest_simple_form(name, f):
    x = np.linspace(0.5, 3.0, 64) if name in ("log(x)", "√x") else np.linspace(-1.0, 2.0, 64)
    match = lab.closest_form(x, f(x), "x")
    assert match["name"] == name and match["r2"] > 0.9999
    assert lab.closest_form(x, np.full(64, 2.0), "x") is None  # a flat edge has no shape


def test_a_wave_edge_reports_its_frequency_and_shift():
    x = np.linspace(-1.0, 1.0, 64)
    assert lab.closest_form(x, np.sin(2 * np.pi * x), "y")["name"] == "sin(2πy)"
    match = lab.closest_form(x, np.sin(2.3 * x + 0.4), "x")
    assert re.fullmatch(r"sin\(2\.[23]\d?·x \+0\.[34]\d?\)", match["name"]) and match["r2"] > 0.9999

