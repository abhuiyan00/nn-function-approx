"""Sensitivity study: one setting takes several values, everything else stays, and each value is trained on several seeds.

The seeds give every value a band, so an effect can be told from the luck of the starting
weights: the study reports how far the error moves across the values next to how far it
moves between seeds at one value.
"""
from __future__ import annotations

import uuid

from .lab import LabManager, LabRun, clean
from .schemas import SweepRequest
from .search import _median, _usable


class Sweep:
    def __init__(self, request: SweepRequest, lab: LabManager):
        self.id = uuid.uuid4().hex[:10]
        self.param = request.param
        self.target = request.base.resolved
        self.label = request.base.describe()
        self.stop_requested = False
        self.values = sorted(request.values)
        self.runs: list[list[LabRun]] = [[lab.submit(spec) for spec in specs] for specs in request._specs]

    def stop(self) -> None:
        self.stop_requested = True
        for runs in self.runs:
            for run in runs:
                run.stop_requested = True

    def _row(self, value: float, runs: list[LabRun]) -> dict:
        finished = [r for r in runs if r.final and _usable(r.final["metrics"]["rmse"])]
        row = dict(value=value, n_params=runs[0].net.n_params, seeds=len(runs), done=sum(r.finished for r in runs),
                   status="done" if all(r.finished for r in runs) else next(r.status for r in runs if not r.finished),
                   best_run=None, rmse=None, rmse_lo=None, rmse_hi=None, rmse_gap=None, seconds=None, epochs=None)
        if finished:
            rmse = [r.final["metrics"]["rmse"] for r in finished]
            row.update(rmse=_median(rmse), rmse_lo=min(rmse), rmse_hi=max(rmse),
                       rmse_gap=_median([r.final["metrics"].get("rmse_gap") for r in finished]),
                       seconds=_median([r.final["seconds"] for r in finished]),
                       epochs=_median([r.final["epochs_run"] for r in finished]),
                       best_run=min(finished, key=lambda r: r.final["metrics"]["rmse"]).id)
        return row

    def state(self) -> dict:
        rows = [self._row(value, runs) for value, runs in zip(self.values, self.runs)]
        flat = [r for runs in self.runs for r in runs]
        over = all(r.finished for r in flat)
        scored = [r for r in rows if r["rmse"]]
        best = min(scored, key=lambda r: r["rmse"]) if scored else None
        return clean(dict(
            id=self.id, param=self.param, label=self.label, target=self.target.key, formula=self.target.formula,
            known=self.target.known, status="running" if not over else "stopped" if self.stop_requested else "done",
            done=sum(r.finished for r in flat), total=len(flat), rows=rows, best=best["value"] if best else None,
            # how far the error moves across the values, and between seeds at the best value
            effect=max(r["rmse"] for r in scored) / best["rmse"] if best else None,
            spread=best["rmse_hi"] / best["rmse_lo"] if best and best["rmse_lo"] else None))
