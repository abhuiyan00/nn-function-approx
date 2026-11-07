"""The closest curve: train a panel of models on one function and rank how close each one gets.

Every network is an ordinary lab run, so any of them can be opened, saved or queried later.
Ranking uses all the truth there is: the exact function over the whole range, hole included,
or the held-out points for measured data. The pick a real user could make, by validation
error alone, is reported next to it, because on a hole the two can differ.
"""
from __future__ import annotations

import math
import statistics
import threading
import uuid
from collections import OrderedDict

import numpy as np

from funcprox.data import make_dataset

from .lab import LabManager, LabRun, clean, make_grid, matched_baselines, round_array
from .schemas import RunSpec, SearchRequest

MAX_KEPT = 10
EFFORT = {"quick": dict(epochs=500, patience=60, iters=120), "thorough": dict(epochs=2000, patience=200, iters=400)}
MODEL_DEFAULTS = dict(arch="mlp", hidden=[32, 32], act="tanh", bias_init=None, fourier_m=0, fourier_scale=2.0, w0=30.0,
                      widths=None, grid=5, grid_updates=[], lr=1e-3, batch=64, l2=0.0)


def panel(dim: int, effort: str) -> list[dict]:
    """One model of each kind the engine has. Levenberg-Marquardt gets the sizes it can solve quickly."""
    e = EFFORT[effort]
    adam = dict(optimizer="adam", epochs=e["epochs"], patience=e["patience"])
    lm = dict(optimizer="lm", epochs=e["iters"], patience=30)
    return [
        dict(hidden=[32, 32], act="tanh", **adam),
        dict(hidden=[16, 16], act="tanh", **lm),
        dict(hidden=[32, 32], act="relu", bias_init="kink", **adam),
        dict(hidden=[32, 32], act="tanh", fourier_m=16, **adam),
        dict(hidden=[32, 32], act="sine", w0=10.0, **adam),
        dict(arch="kan", widths=[dim, 5, 1], grid=10, grid_updates=[25, 75], **lm),
    ]


def _usable(value) -> bool:
    return value is not None and math.isfinite(value)


def _median(values: list) -> float | None:
    values = [v for v in values if _usable(v)]
    return statistics.median(values) if values else None


class Search:
    def __init__(self, request: SearchRequest, lab: LabManager):
        self.id = uuid.uuid4().hex[:10]
        base = request.base
        self.target = base.resolved
        self.stop_requested = False
        self.classical: list[dict] = []
        self.classical_done = False
        self.grid_x, self.grid = make_grid(self.target)
        specs = [RunSpec(**{**base.model_dump(), **MODEL_DEFAULTS, **model, "seed": seed})
                 for model in panel(self.target.dim, request.effort) for seed in range(1, request.seeds + 1)]
        self.runs: list[LabRun] = [lab.submit(spec) for spec in specs]
        self.samples = self.runs[0].samples
        lab.run_after_queue(lambda: self._fit_classical(base))

    def _fit_classical(self, base: RunSpec) -> None:
        try:
            if not self.stop_requested:
                ds = make_dataset(self.target, base.n_train, base.n_val, base.sigma, base.data_seed)
                self.classical = matched_baselines(ds, max(r.net.n_params for r in self.runs), ds.xs.transform(self.grid_x))
        finally:  # a search must still end when the classical fits fail
            self.classical_done = True

    def stop(self) -> None:
        self.stop_requested = True
        for run in self.runs:
            run.stop_requested = True

    # ------------------------------------------------------------------- payload
    def _network_row(self, slot: int, label: str, runs: list[LabRun]) -> dict:
        finished = [r for r in runs if r.final and _usable(r.final["metrics"]["rmse"])]
        row = dict(key=label, slot=slot, kind="network", seeds=len(runs), n_params=runs[0].net.n_params, best_run=None,
                   status="done" if len(finished) == len(runs) else next((r.status for r in runs if not r.finished), runs[0].status))
        if finished:
            metrics = [r.final["metrics"] for r in finished]
            best = min(finished, key=lambda r: r.final["metrics"]["rmse"])
            rmse = [m["rmse"] for m in metrics]
            row.update(rmse=_median(rmse), rmse_lo=min(rmse), rmse_hi=max(rmse), best_run=best.id, pred=best.final["pred"],
                       rmse_gap=_median([m.get("rmse_gap") for m in metrics]), rmse_seen=_median([m.get("rmse_seen") for m in metrics]),
                       max_err=_median([m["max_err"] for m in metrics]), val_mse=_median([m["val_mse_noisy"] for m in metrics]),
                       seconds=sum(r.final["seconds"] for r in finished))
            curves = np.array([r.final["pred"] for r in finished], dtype=float)
            if len(finished) > 1 and np.isfinite(curves).all():
                row.update(lo=round_array(curves.min(axis=0)), hi=round_array(curves.max(axis=0)))
        else:
            live = next((r.live for r in runs if r.live), None)
            if live:
                row.update(rmse=live["rmse"], rmse_gap=live["rmse_gap"], pred=live["pred"], best_run=runs[0].id)
        return row

    def _classical_row(self, slot: int, fit: dict) -> dict:
        return dict(key=fit["model"], slot=slot, kind="classical", seeds=1, n_params=fit["n_coef"], best_run=None, status="done",
                    rmse=fit["rmse"], rmse_lo=fit["rmse"], rmse_hi=fit["rmse"], rmse_gap=fit["rmse_gap"], rmse_seen=fit["rmse_seen"],
                    max_err=fit["max_err"], val_mse=fit["val_mse"], pred=fit["pred"], seconds=None)

    def state(self) -> dict:
        groups: dict[str, list[LabRun]] = {}
        for run in self.runs:
            groups.setdefault(run.spec.describe(), []).append(run)
        # A fixed colour slot per model: the chart must not repaint a curve when the ranking changes
        rows = [self._network_row(i, label, runs) for i, (label, runs) in enumerate(groups.items())]
        rows += [self._classical_row(len(groups) + i, fit) for i, fit in enumerate(self.classical)]
        rows.sort(key=lambda r: r["rmse"] if _usable(r.get("rmse")) else math.inf)
        complete = [r for r in rows if r["status"] == "done" and _usable(r.get("rmse"))]
        winner = complete[0]["key"] if complete else None
        best_network = next((r["key"] for r in complete if r["kind"] == "network"), None)
        priced = [r for r in complete if _usable(r.get("val_mse"))]
        if self.grid["kind"] == "2d":  # a surface is large and only the two named below are drawn
            for row in rows:
                if row["key"] not in (winner, best_network):
                    row["pred"] = row["lo"] = row["hi"] = None
        over = all(r.finished for r in self.runs) and self.classical_done
        return clean(dict(id=self.id, status="running" if not over else "stopped" if self.stop_requested else "done",
                          done=sum(r.finished for r in self.runs), total=len(self.runs), formula=self.target.formula,
                          dim=self.target.dim, known=self.target.known, grid=self.grid, samples=self.samples, rows=rows, winner=winner,
                          best_network=best_network, blind_pick=min(priced, key=lambda r: r["val_mse"])["key"] if priced else None))


class SearchManager:
    def __init__(self, lab: LabManager):
        self.lab = lab
        self.searches: OrderedDict[str, Search] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, request: SearchRequest) -> Search:
        search = Search(request, self.lab)
        with self._lock:
            self.searches[search.id] = search
            while len(self.searches) > MAX_KEPT:
                self.searches.popitem(last=False)
        return search
